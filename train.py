#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

import pathlib
import scipy
import matplotlib.pyplot as plt
import datetime
from callbacks import log_speed_callback
from datasets import files_dataset


AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Using tensorflow version {}".format(tf.__version__))

BATCH_SIZE = 32
IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# set up our datasets

val_dir = "/home/alex/inat/tf2_classification/records/val"
train_dir = "/home/alex/inat/tf2_classification/records/train"

train_glob = "{}/*/*".format(train_dir)
train_list_ds = tf.data.Dataset.list_files(train_glob)
num_train = tf.data.experimental.cardinality(train_list_ds).numpy()
CLASS_NAMES = np.array([item.name for item in pathlib.Path(val_dir).glob('*')])
STEPS_PER_EPOCH = np.ceil(num_train/BATCH_SIZE)

# some image augmentations, originally from
# https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
# but updated for tf2

def rotate(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    # Rotate 0, 90, 180, 270 degrees
    rotate_amt = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, rotate_amt)
    return x, y

def flip(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y

# todo: look at how we're already doing this in Grant's code, match it
def color(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y

# redo this, we just need a random crop not this zoom
def zoom(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
    return x, y

# Add augmentations
augmentations = [flip, color, rotate]


val_ds = files_dataset.dataset_from_directory(val_dir, IMG_SIZE)
val_ds = files_dataset.prepare_dataset(
    val_ds, 
    cache=False, 
    augment_images=False
)

train_ds = files_dataset.dataset_from_directory(train_dir, IMG_SIZE)
train_ds = files_dataset.prepare_dataset(
    train_ds,
    cache=False, 
    augment_images=True, 
    augmentations=augmentations
)

image_batch, label_batch = next(iter(train_ds))

# let's make our model

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# create the base model from the pre-trained model Inception V3
base_model = tf.keras.applications.InceptionV3(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

feature_batch = base_model(image_batch)
base_model.trainable = True

# add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)

# append the classification head to the base model
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 20
validation_steps = 200

loss0,accuracy0 = model.evaluate(train_ds, steps = validation_steps)

print("initial loss: {:.4f}".format(loss0))
print("initial accuracy: {:.4f}".format(accuracy0))
print("expected initial accuracy: {:.4f}".format(1/len(CLASS_NAMES)))

# setup keras training callbacks

# update_freq seems to be per example, so if we want
# to update the logs every 100 steps, this is 
# BATCH_SIZE * 100 * NUM_GPUS.

# set profile_batch to 0 to workaround bug: 
# https://github.com/tensorflow/tensorboard/issues/2084
update_batch_freq = 100 * 1
update_examples_freq = BATCH_SIZE * 100 * 1
log_dir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-decompose_callbacks"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1, 
    update_freq=update_examples_freq,
    profile_batch=0
)

# tensorboard stuff
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()

step_sec_callback = log_speed_callback.LogStepsPerSecCallback(
    update_batch_freq, 
    STEPS_PER_EPOCH
)

def scheduler(epoch):
    # picking numbers out of a hat
    if epoch < 1:
        return 0.00015 
    else:
        return (0.0001 * tf.math.exp(0.1 * (1 - epoch))).numpy()

adjust_lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_path = log_dir + "/" + "checkpoint-{epoch:02d}.hdf5"
save_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1
)

callbacks = [
    step_sec_callback,
    adjust_lr_callback,
    tensorboard_callback,
    save_checkpoint_callback
]

history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks
)



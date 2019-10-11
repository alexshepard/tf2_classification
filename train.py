#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

import pathlib
import scipy
import matplotlib.pyplot as plt
import datetime

from callbacks import log_speed_callback
from datasets import files_dataset, augments
from nets import inat_inception


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

# training ugmentations
augmentations = [augments.flip, augments.color, augments.rotate]

# make datasets
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

# pull a batch from our training dataset
image_batch, label_batch = next(iter(train_ds))

# let's make our model
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
model = inat_inception.compiled_model(
    img_shape=IMG_SHAPE,
    image_batch=image_batch,
    num_classes=len(CLASS_NAMES)
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
log_dir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-decompose_dataset"
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



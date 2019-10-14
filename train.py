#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

import pathlib
import datetime
import argparse
import os

from callbacks import log_speed_callback
from datasets import files_dataset, augments
from nets import inat_inception, inat_nasnet

AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Using tensorflow version {}".format(tf.__version__))

NUM_GPUS=1
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

def callbacks_for_training(log_dir, update_batch_freq, batch_size, steps_per_epoch, lr_scheduler_fn):
    checkpoint_path = os.path.join(log_dir, "checkpoint-{epoch:02d}.hdf5")

    save_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1
    )

    step_sec_callback = log_speed_callback.LogStepsPerSecCallback(
        update_batch_freq,
        steps_per_epoch
    )
    
    adjust_lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_fn)

    # set profile_batch to 0 to workaround bug: 
    # https://github.com/tensorflow/tensorboard/issues/2084
    # tensorboar callback's update_freq is exppressed in terms
    # of examples, not batches
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1, 
        update_freq=update_batch_freq*batch_size,
        profile_batch=0
    )

    return [ 
        save_checkpoint_callback,
        step_sec_callback,
        adjust_lr_callback,
        tensorboard_callback
    ]
   


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        dest='train_dir',
        help='path to directory of training images',
        required=True,
        type=str
    )

    parser.add_argument(
        '--val_dir',
        dest='val_dir',
        help='path to directory of val images',
        required=True,
        type=str
    )

    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='number of training epochs',
        required=False,
        type=int,
        default=20
    )

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch size for training',
        required=False,
        type=int,
        default=32
    )

    parser.add_argument(
        '--val_steps',
        dest='val_steps',
        help='val steps to run per epoch',
        required=False,
        type=int,
        default=200
    )

    parser.add_argument(
        '--experiment_name',
        dest='experiment_name',
        help='a name for this training run',
        required=True,
        type=str
    )

    parser.add_argument(
        '--experiments_dir',
        dest='experiments_dir',
        help='the directory where your experiments live',
        required=True,
        type=str
    )

    parser.add_argument(
        '--train_update_freq',
        dest='train_update_freq',
        help='the number of steps between train updates',
        required=False,
        type=int,
        default=100
    )

    return parser.parse_args()

def main():
    args = parse_args()

    train_glob = os.path.join(*[args.train_dir, '*', '*'])
    train_list_ds = tf.data.Dataset.list_files(train_glob)
    num_train = tf.data.experimental.cardinality(train_list_ds).numpy()
    CLASS_NAMES = np.array([item.name for item in pathlib.Path(args.val_dir).glob('*')])
    STEPS_PER_EPOCH = np.ceil(num_train/args.batch_size)

    # training ugmentations
    augmentations = [augments.flip, augments.color, augments.rotate]

    # make datasets
    val_ds = files_dataset.dataset_from_directory(args.val_dir, IMG_SIZE)
    val_ds = files_dataset.prepare_dataset(
        val_ds, 
        cache=False, 
        augment_images=False
    )

    train_ds = files_dataset.dataset_from_directory(args.train_dir, IMG_SIZE)
    train_ds = files_dataset.prepare_dataset(
        train_ds,
        cache=False, 
        augment_images=True, 
        augmentations=augmentations
    )

    # let's make our model
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
    model = inat_nasnet.compiled_model(
        img_shape=IMG_SHAPE,
        num_classes=len(CLASS_NAMES)
    )

    print("evaluating model")
    evaluate_output = model.evaluate(train_ds, steps = args.val_steps)
    (loss0, head1_loss, head2_loss, head1_accuracy, head2_accuracy) = evaluate_output

    print()
    print("initial total loss: {:.4f}".format(loss0))
    print("initial head1 loss: {:.4f}".format(head1_loss))
    print("initial head2 loss: {:.4f}".format(head2_loss))
    print()
    print("initial head1 accuracy: {:.4f}".format(head1_accuracy))
    print("initial head2 accuracy: {:.4f}".format(head2_accuracy))
    print("expected initial accuracy: {:.4f}".format(1/len(CLASS_NAMES)))
    print()

    update_batch_freq = 100 * NUM_GPUS
    experiment_dir = "{}-{}".format(
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            args.experiment_name
    )
    log_dir = os.path.join(args.experiments_dir, experiment_dir)

    # tensorboard stuff
    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()

    # learning rate scheduler helper function
    def lr_scheduler(epoch):
        # picking numbers out of a hat
        if epoch < 1:
            return 0.00015 
        else:
            return (0.0001 * tf.math.exp(0.1 * (1 - epoch))).numpy()

    # keras fit callbacks
    callbacks = callbacks_for_training(
        log_dir = log_dir,
        update_batch_freq = update_batch_freq,
        batch_size = args.batch_size,
        steps_per_epoch = STEPS_PER_EPOCH,
        lr_scheduler_fn = lr_scheduler
    )
 
    history = model.fit(
        train_ds,
        epochs=args.num_epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=args.val_steps,
        callbacks=callbacks
    )

   

if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np

# some image augments, originally from
# https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
# but updated for tf2 and restructured to take and return both
# an image and a label

def rotate(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    # rotate 0, 90, 180, 270 degrees
    rotate_amt = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, rotate_amt)
    return x, y

def flip(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y

def color(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y

def _center_crop(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    height = x.shape[0]
    width = x.shape[1]
    x = tf.image.central_crop(x, 0.875)
    x = tf.image.resize(x, size=(height,width))
    return x, y

def _random_crop(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    height = x.shape[0]
    width = x.shape[1]

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1,1,4])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(x),
        bounding_boxes = bbox,
        area_range=(0.05,1.0),
        aspect_ratio_range=(0.75, 1.33),
        max_attempts=100,
        min_object_covered=0.1
    )
    x = tf.slice(x, begin, size)
    x = tf.image.resize(x, size=(height, width))

    return x, y

def crop(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    return _random_crop(x, y)
    #return _center_crop(x, y)



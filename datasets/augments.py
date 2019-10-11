import tensorflow as tf

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

def zoom(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    raise NotImplementedError

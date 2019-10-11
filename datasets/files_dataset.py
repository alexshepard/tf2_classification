import tensorflow as tf
import numpy as np
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

def dataset_from_directory(directory, IMG_SIZE):
    glob = "{}/*/*".format(directory)
    list_ds = tf.data.Dataset.list_files(glob)
    num_examples = tf.data.experimental.cardinality(list_ds).numpy()
    CLASS_NAMES = np.array([item.name for item in pathlib.Path(directory).glob('*')])
    return list_ds.map(lambda x: _process_path(x, CLASS_NAMES, IMG_SIZE), num_parallel_calls=AUTOTUNE)
    #return list_ds.map(process_path, num_parallel_calls = AUTOTUNE)

def _get_label(file_path, CLASS_NAMES):
    parts = tf.strings.split(file_path, "/")
    class_id = int(parts[-2])
    # one hot encoding
    return CLASS_NAMES==class_id

def _decode_img(img, IMG_SIZE):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, IMG_SIZE)

def _process_path(file_path, CLASS_NAMES, IMG_SIZE):
    label = _get_label(file_path, CLASS_NAMES)
    img = tf.io.read_file(file_path)
    img = _decode_img(img, IMG_SIZE)
    return img, label

def prepare_dataset(ds, cache=True, batch_size=32, shuffle_buffer_size=100, augment_images=True, augmentations=[]):
    def do_image_augmentations(ds, augmentations):
        for f in augmentations:
            ds = ds.map(f, num_parallel_calls=AUTOTUNE)
        # make sure the values are still in [0,1]
        ds = ds.map(lambda x,y: (tf.clip_by_value(x, 0, 1), y), num_parallel_calls=AUTOTUNE)
        return ds
        
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if augment_images:
        ds = do_image_augmentations(ds, augmentations)

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.repeat()

    ds = ds.batch(batch_size)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# import os
import numpy as np
import tensorflow as tf


def _rotate(img, lbl):
    # Rotate 0, 90, 180, 270 degrees
    img = tf.image.rot90(img, tf.random_uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32))
    return img, lbl


def _flip(img, lbl):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, lbl


def _make_numpy_data_from_hdf5(file_name):
    '''astro image datasets'''
    from tfcae.hdf5_readers import AstroHDF5Reader as HDF5Reader
    reader = HDF5Reader(file_name)
    nevents = reader.openf(make_data_dict=True)
    features = reader.data_dict['images']
    if len(features.shape) < 4:
        # need (batch, H, W, depth)
        features = np.expand_dims(features, axis=-1)
    labels = reader.data_dict['oh_labels']
    reader.closef()
    return nevents, features, labels


def _make_astro_generator_fn(file_name, batch_size, imghw, imgdepth):
    """
    make a generator function that we can query for batches
    """
    from tfcae.hdf5_readers import AstroHDF5Reader as HDF5Reader
    reader = HDF5Reader(file_name, imghw, imgdepth)
    nevents = reader.openf()

    def example_generator_fn():
        start_idx, stop_idx = 0, batch_size
        while True:
            if start_idx >= nevents:
                reader.closef()
                return
            yield reader.get_examples(start_idx, stop_idx)
            start_idx, stop_idx = start_idx + batch_size, stop_idx + batch_size

    return example_generator_fn


def make_astro_dset(
    file_name, batch_size, num_epochs=1, shuffle=False, in_memory=True,
    imghw=64, imgdepth=1, augment=True
):
    '''
    only need to supply imghw and imgdepth if not using an in_memory dataset.
    (TODO - if we go to "big" datasets, probably worth doing TFRecord
    conversion and avoiding from_generator)
    '''
    augmentations = [_flip, _rotate]
    if in_memory:
        _, features, targets = _make_numpy_data_from_hdf5(file_name)
        ds = tf.data.Dataset.from_tensor_slices((
            features.astype(np.float32), targets
        ))
        if augment:
            for f in augmentations:
                ds = ds.map(f, num_parallel_calls=4)
        if shuffle:
            ds = ds.shuffle(10000)
        ds = ds.repeat(num_epochs)
        ds = ds.batch(batch_size)
        return ds
    else:
        raise NotImplementedError('Not using from_generator anymore.')
    return None


def make_astro_iterators(
    file_name, batch_size, num_epochs=1, shuffle=False, in_memory=True,
    imghw=64, imgdepth=1
):
    ds = make_astro_dset(
        file_name, batch_size, num_epochs, shuffle, in_memory, imghw, imgdepth
    )

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, labs = itrtr.get_next()
    return feats, labs


def read_encoded_hdf5(encoded_filename, n_floats=64):
    from tfcae.hdf5_readers import EncodedHDF5Reader as HDF5Reader
    reader = HDF5Reader(encoded_filename, n_floats)
    _ = reader.openf()
    images = reader.data_dict['images']
    labels = reader.data_dict['labels']
    reader.closef()
    return images, labels


def make_encoded_numpy(encoded_filename, n_floats=64):
    '''encoded (compressed) data'''
    images, labels = read_encoded_hdf5(encoded_filename, n_floats)
    ones_idx = np.where(labels == 1)
    zeros_idx = np.where(labels == 0)
    ones = images[ones_idx]
    zeros = images[zeros_idx]
    return ones, zeros


def make_encoded_dataset(encoded_filename, n_floats=64, batch_size=50):
    ones, zeros = make_encoded_numpy(encoded_filename, n_floats)
    ones_ds = tf.data.Dataset.from_tensor_slices(ones)
    zeros_ds = tf.data.Dataset.from_tensor_slices(zeros)
    ones_ds = ones_ds.batch(batch_size)
    zeros_ds = zeros_ds.batch(batch_size)
    return ones_ds, zeros_ds

# def get_data_files_dict(path='path_to_data', tfrecord=False):
#     data_dict = {}
#     if tfrecord:
#         data_dict['train'] = os.path.join(path, 'fashion_train.tfrecord.gz')
#         data_dict['test'] = os.path.join(path, 'fashion_test.tfrecord.gz')
#     else:
#         data_dict['train'] = os.path.join(path, 'fashion_train.hdf5')
#         data_dict['test'] = os.path.join(path, 'fashion_test.hdf5')
#     return data_dict

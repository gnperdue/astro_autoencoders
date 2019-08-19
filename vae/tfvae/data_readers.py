from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# import os
import numpy as np
import tensorflow as tf


def _rotate(img, lbl):
    # Rotate 0, 90, 180, 270 degrees
    img = tf.image.rot90(img, tf.compat.v2.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32))
    return img, lbl


def _flip(img, lbl):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, lbl


def _rotate_img_only(img):
    # Rotate 0, 90, 180, 270 degrees
    img = tf.image.rot90(img, tf.compat.v2.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32))
    return img


def _flip_img_only(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


def _make_numpy_data_from_hdf5(file_name):
    '''astro image datasets'''
    from tfvae.hdf5_readers import AstroHDF5Reader as HDF5Reader
    reader = HDF5Reader(file_name)
    nevents = reader.openf(make_data_dict=True)
    features = reader.data_dict['images']
    if len(features.shape) < 4:
        # need (batch, H, W, depth)
        features = np.expand_dims(features, axis=-1)
    labels = reader.data_dict['oh_labels']
    reader.closef()
    return nevents, features, labels


def make_astro_dset(
    file_name, batch_size, num_epochs=1, shuffle=False, mode='plain',
    imghw=64, imgdepth=1, augment=True
):
    '''
    only need to supply imghw and imgdepth if not using an in_memory dataset.
    (TODO - if we go to "big" datasets, probably worth doing TFRecord
    conversion and avoiding from_generator)
    '''
    augmentations = []
    _, features, targets = _make_numpy_data_from_hdf5(file_name)
    if mode == 'with_labels':
        augmentations.extend([_flip, _rotate])
        ds = tf.data.Dataset.from_tensor_slices((
            features.astype(np.float32), targets
        ))
    elif mode == 'plain':
        augmentations.extend([_flip_img_only, _rotate_img_only])
        ds = tf.data.Dataset.from_tensor_slices(features.astype(np.float32))
    else:
        raise ValueError('unknown mode for make_astro_dset')
    if augment:
        for f in augmentations:
            ds = ds.map(f, num_parallel_calls=4)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size)

    return ds


def make_astro_iterators(
    file_name, batch_size, num_epochs=1, shuffle=False, mode='plain',
    imghw=64, imgdepth=1
):
    ds = make_astro_dset(
        file_name, batch_size, num_epochs, shuffle, mode, imghw, imgdepth
    )

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, labs = itrtr.get_next()
    return feats, labs


def read_encoded_hdf5(encoded_filename, n_floats=64):
    from tfvae.hdf5_readers import EncodedHDF5Reader as HDF5Reader
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

'''
HDF5 structure:

* stargalaxy_real.h5
    * imageset - (N, H, W, C) = (N, 48, 48, 3), dtype = uint8
    * catalog - (N,), dtype = int32
* stargalaxy_sim_20190214.h5
    * imageset - (N, H, W) = (N, 64, 64), dtype = float(32)
    * catalog - (N,), dtype = float(32)
* strong_lensing_spacebased.h5
    * imageset - (N, H, W) = (N, 101, 101), dtype = float32
    * catalog - (N,), dtype = float(32)

All 'catalog' values are either 0 or 1.
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import h5py
import numpy as np


class AstroHDF5Reader(object):
    """
    user should call `openf()` and `closef()` to start/finish.

    two modes of operation - when opening, pass `make_data_dict=True` to read
    the whole HDF5 file into numpy arrays (convenient for direct conversion
    to tf.data.Dataset object); otherwise just open the HDF5 and call the
    various `get_X` methods for use with a tf.data.Dataset using
    `from_generator`.
    """

    def __init__(self, hdf5_file, imghw=64, imgdepth=1, tofloat=False):
        '''
        assume image height and width are the same `imghw`, imgdepth is `None`
        if the image is just (H, W), or D for (H, W, D) - img dimensions only
        matter here if one is not creating a 'data directory' when calling
        `openf()`.
        '''
        self._file = hdf5_file
        self._f = None
        self._nlabels = 2
        self._tofloat = True
        self._imghw = imghw
        self._imgdepth = imgdepth

    def openf(self, make_data_dict=False):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['catalog'].shape[0]
        self.data_dict = {}
        if make_data_dict:
            self.data_dict['images'] = self._f['imageset'][:]
            self.data_dict['labels'] = self._f['catalog'][:]
            labels = self._f['catalog'][:].reshape([-1]).astype('int')
            oh_labels = np.zeros((labels.size, self._nlabels), dtype=np.uint8)
            oh_labels[np.arange(labels.size), labels] = 1
            self.data_dict['oh_labels'] = oh_labels
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_example(self, idx):
        image = self._f['imageset'][idx]
        if len(image.shape) < 3:
            # need (H, W, depth)
            image = np.expand_dims(image, axis=-1)
        label = self._f['catalog'][idx].reshape([-1])
        oh_label = np.zeros((1, self._nlabels), dtype=np.uint8)
        oh_label[0, label] = 1
        if self._tofloat:
            return image.astype(np.float32), \
                oh_label.reshape(self._nlabels,).astype(np.float32)
        return image, oh_label.reshape(self._nlabels,)

    def get_flat_example(self, idx):
        image, label = self.get_example(idx)
        image = np.reshape(image, (np.prod(image.shape)))
        return image, label

    def get_examples(self, start_idx, stop_idx):
        image = self._f['imageset'][start_idx: stop_idx]
        if len(image.shape) < 3:
            # need (H, W, depth)
            image = np.expand_dims(image, axis=-1)
        label = self._f['catalog'][start_idx: stop_idx].reshape([-1])
        oh_label = np.zeros((label.size, self._nlabels), dtype=np.uint8)
        oh_label[np.arange(label.size), label] = 1
        if self._tofloat:
            return image.astype(np.float32), oh_label.astype(np.float32)
        return image, oh_label

    def get_flat_examples(self, start_idx, stop_idx):
        image, label = self.get_examples(start_idx, stop_idx)
        image = np.reshape(image, (-1, np.prod(image.shape[1:])))
        return image, label


class EncodedHDF5Reader(object):
    """
    user should call `openf()` and `closef()` to start/finish.

    here, we assume the dataset is small enough to fit in memory. therefore,
    we only read the entire dataset into memory.
    """

    def __init__(self, hdf5_file, n_floats=64):
        self._file = hdf5_file
        self._f = None
        self._nlabels = 2
        self._nfloats = n_floats

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['encoded/labels'].shape[0]
        self.data_dict = {}
        # labels are already one-hot in the file
        self.data_dict['images'] = self._f['encoded/images'][:]
        self.data_dict['oh_labels'] = self._f['encoded/labels'][:]
        self.data_dict['labels'] = np.ones(self._nevents)
        idx = np.where(self.data_dict['oh_labels'][:, 0] == 1)
        self.data_dict['labels'][idx] = 0
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import unittest
import os

import tensorflow as tf
from tfcae.data_readers import make_astro_dset

tfe = tf.contrib.eager
tf.enable_eager_execution()

# Get path to data
SGREAL = os.path.join(
    os.environ['HOME'],
    'Dropbox/Quantum_Computing/hep-qml/data/stargalaxy_real.h5'
)
SGSIM = os.path.join(
    os.environ['HOME'],
    'Dropbox/Quantum_Computing/hep-qml/data/stargalaxy_sim_20190214.h5'
)


class TestDataReadersEager(unittest.TestCase):

    def test_eager_one_shot_iterator_read(
        self
    ):
        for h5file, shape in zip([SGREAL, SGSIM], [(48, 48, 3), (64, 64, 1)]):
            batch_size = 25
            num_epochs = 1

            targets_and_labels = make_astro_dset(
                h5file, batch_size, num_epochs,
                imghw=shape[0], imgdepth=shape[2]
            )

            for i, (fs, ls) in enumerate(tfe.Iterator(targets_and_labels)):
                self.assertTrue(fs.shape == (
                    batch_size, shape[0], shape[1], shape[2]
                ))
                self.assertTrue(fs.dtype == tf.float32)
                self.assertTrue(ls.shape == (batch_size, 2))
                self.assertTrue(ls.dtype == tf.uint8)
                if i > 2:
                    break


if __name__ == '__main__':
    unittest.main()

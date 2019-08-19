from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import unittest
import os

import tensorflow as tf
from tfcae.data_readers import make_astro_iterators

# Get path to data
SGREAL = os.path.join(
    os.environ['HOME'],
    'Dropbox/Quantum_Computing/hep-qml/data/stargalaxy_real.h5'
)
SGSIM = os.path.join(
    os.environ['HOME'],
    'Dropbox/Quantum_Computing/hep-qml/data/stargalaxy_sim_20190214.h5'
)


class TestDataReadersGraph(unittest.TestCase):

    def test_graph_one_shot_iterator_read(self):
        for h5file, shape in zip([SGREAL, SGSIM], [(48, 48, 3), (64, 64, 1)]):
            batch_size = 25
            num_epochs = 1

            feats, labs = make_astro_iterators(
                h5file, batch_size, num_epochs,
                imghw=shape[0], imgdepth=shape[2]
            )
            counter = 0

            with tf.Session() as sess:
                try:
                    while True:
                        fs, ls = sess.run([feats, labs])
                        self.assertTrue(fs.shape == (
                            batch_size, shape[0], shape[1], shape[2]
                        ))
                        self.assertTrue(fs.dtype == tf.float32)
                        self.assertTrue(ls.shape == (batch_size, 2))
                        self.assertTrue(ls.dtype == tf.uint8)
                        counter += 1
                        if counter > 2:
                            break

                except tf.errors.OutOfRangeError:
                    pass
                except Exception as e:
                    print(e)


if __name__ == '__main__':
    unittest.main()

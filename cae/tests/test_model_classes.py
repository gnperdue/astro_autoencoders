from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import unittest
# import os

import numpy as np
import tensorflow as tf
import tfcae.tf_model_classes


class TestAstroAutoencoder(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.model = tfcae.tf_model_classes.AstroAutoencoder(0.001)

    def tearDown(self):
        pass

    def test_build_network(self):
        with tf.Graph().as_default() as g:
            with tf.Session(graph=g) as sess:
                features = tf.zeros((1, 64, 64, 1))
                self.model.build_network(features)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                loss, encoded = sess.run([self.model.loss, self.model.encoded])
                self.assertEqual(loss, 0.0)
                self.assertEqual(np.sum(encoded), 0.0)
                print(loss, encoded)

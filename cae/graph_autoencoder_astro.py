"""
Autoencoder using the TF graph APIs

TODO - make this a class, not a script...
"""
import time
import os
import argparse
import logging

# import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tfcae.data_readers import make_astro_iterators
from tfcae.tf_model_classes import AstroAutoencoder
from tfcae.hdf5_utils import setup_hdf5
from tfcae.hdf5_utils import add_batch_to_hdf5
from tfcae.misc_fns import get_expected_image_shape
from tfcae.misc_fns import get_logging_level
from tfcae.misc_fns import get_number_of_trainable_parameters
from tfcae.misc_fns import log_function_args


def main(batch_size, data_dir, data_type, learning_rate, log_level, model_dir,
         num_epochs, train_steps):
    logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] \
        + str(int(time.time())) + '.txt'
    logging.basicConfig(
        filename=logfilename, level=get_logging_level(log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting...")
    logger.info(__file__)
    log_function_args(vars(), logger)
    test_file = os.path.join(data_dir, data_type + '_cae_test.h5')
    train_file = os.path.join(data_dir, data_type + '_cae_train.h5')
    imgshp = get_expected_image_shape(train_file)
    logger.info('Train / test files = {} / {}'.format(train_file, test_file))
    train(batch_size, imgshp, learning_rate, logger, model_dir, num_epochs,
          train_file, train_steps)
    n_encoded, n_labels = test(imgshp, logger, model_dir, test_file)
    encode(test_file, data_type + 'encoded_test.hdf5', imgshp, logger,
           model_dir, n_encoded, n_labels)


def train(batch_size, imgshp, learning_rate, logger, model_dir, num_epochs,
          train_file, train_steps):
    log_function_args(vars(), logger)
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'
    run_dest_dir = model_dir + '/%d' % time.time()
    n_steps = train_steps or 1000000000

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, _ = make_astro_iterators(
                train_file, batch_size, num_epochs, shuffle=True,
                imghw=imgshp[0], imgdepth=imgshp[2]
            )
            model = AstroAutoencoder(learning_rate=learning_rate)
            model.build_network(features)
            n_model_params = get_number_of_trainable_parameters()
            logger.info('Number of model params = {}'.format(n_model_params))

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            writer = tf.summary.FileWriter(
                logdir=run_dest_dir, graph=sess.graph
            )
            saver = tf.train.Saver(save_relative_paths=True)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                logger.info('Restored session from {}'.format(chkpt_dir))

            writer.add_graph(sess.graph)
            initial_step = model.global_step.eval()
            logger.info('initial step = {}'.format(initial_step))

            try:
                for b_num in range(initial_step, initial_step + n_steps):
                    _, loss_batch, encoded, summary_t = sess.run(
                        [model.optimizer,
                         model.loss,
                         model.encoded,
                         model.train_summary_op]
                    )
                    if (b_num + 1) % 50 == 0:
                        logger.info(
                            ' Loss @step {}: {:2.5f}'.format(b_num, loss_batch)
                        )
                        logger.debug(str(encoded))
                        saver.save(sess, chkpt_dir, b_num)
                        writer.add_summary(summary_t, global_step=b_num)

            except tf.errors.OutOfRangeError:
                logger.info('Training stopped - queue is empty.')

            saver.save(sess, chkpt_dir, b_num)
            writer.add_summary(summary_t, global_step=b_num)

        writer.close()


def test(imgshp, logger, model_dir, test_file):
    log_function_args(vars(), logger)
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, labels = make_astro_iterators(
                test_file, batch_size=1, num_epochs=1, shuffle=False,
                imghw=imgshp[0], imgdepth=imgshp[2]
            )
            model = AstroAutoencoder()
            model.build_network(features)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                logger.info('Restored session from {}'.format(chkpt_dir))

            initial_step = model.global_step.eval()
            logger.info('initial step = {}'.format(initial_step))

            try:
                for i in range(20):
                    loss_batch, encoded_batch, labels_batch, input, recon = \
                        sess.run(
                            [model.loss,
                             model.encoded,
                             labels,
                             model.X,
                             model.Y]
                        )
                    print(loss_batch, encoded_batch.shape, recon.shape)
                    n_encoded = encoded_batch.shape[1]
                    n_labels = labels_batch.shape[1]

                    if imgshp[-1] == 1:
                        imgshp = imgshp[:-1]
                    fig = plt.figure()
                    gs = plt.GridSpec(1, 3)
                    ax1 = plt.subplot(gs[0])
                    ax1.imshow(recon[0].reshape(*imgshp))
                    ax2 = plt.subplot(gs[1])
                    ax2.imshow(input[0].reshape(*imgshp))
                    plt.title(np.argmax(labels_batch[0]))
                    ax3 = plt.subplot(gs[2])
                    ax3.imshow(encoded_batch[0].reshape(8, 8))
                    figname = 'image_{:04d}.pdf'.format(i)
                    plt.savefig(figname, bbox_inches='tight')
                    plt.close()

            except tf.errors.OutOfRangeError:
                logger.info('Testing stopped - queue is empty.')

    return n_encoded, n_labels


def encode(data_file, encoded_file_name, imgshp, logger, model_dir,
           n_encoded, n_labels):
    log_function_args(vars(), logger)
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, labels = make_astro_iterators(
                data_file, batch_size=50, num_epochs=1, shuffle=False,
                imghw=imgshp[0], imgdepth=imgshp[2]
            )
            model = AstroAutoencoder()
            model.build_network(features)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                logger.info('Restored session from {}'.format(chkpt_dir))

            initial_step = model.global_step.eval()
            logger.info('initial step = {}'.format(initial_step))

            f = setup_hdf5(encoded_file_name, n_encoded, n_labels)

            try:
                for i in range(1000000000):
                    encoded_batch, labels_batch = sess.run([
                        model.encoded, labels
                    ])
                    add_batch_to_hdf5(f, encoded_batch, labels_batch)
            except tf.errors.OutOfRangeError:
                logger.info('Testing stopped - queue is empty.')

            f.close()


def decode():
    # don't know if we _really_ need this
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='HDF5 data directory.'
    )
    parser.add_argument(
        '--data-type', type=str, required=True,
        help='stargalaxy_real|stargalaxy_sim_20190214|'
        'strong_lensing_spacebased'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.0001,
        help='Learning rate'
    )
    parser.add_argument(
        '--log-level', default='INFO', type=str,
        help='log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)'
    )
    parser.add_argument(
        '--model-dir', type=str, default='chkpts/mnist_graph',
        help='Model directory'
    )
    parser.add_argument(
        '--num-epochs', type=int, default=1,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--train-steps', type=int, default=None,
        help='Number of training steps'
    )
    args = parser.parse_args()

    main(**vars(args))

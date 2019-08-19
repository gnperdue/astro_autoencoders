import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tfcae.data_readers import make_encoded_dataset

tf.enable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=50, type=int,
                    help='Batch size')
parser.add_argument('--maxplots', default=10, type=int,
                    help='Max batches to plot')
parser.add_argument('--nfloats', default=64, type=int,
                    help='Number of floats in the data compression')
parser.add_argument('--prefix', default='stargalaxy_sim_20190214', type=str,
                    help='Datafile prefix')


def vis_data(example, reshape_shape, maxval=1,
             title=None, plotname=None):
    fig = plt.figure(figsize=(8, 2))
    ax = plt.gca()
    ax.axis('on')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(np.reshape(example, reshape_shape),
               interpolation='none', vmin=0, vmax=maxval)
    if title is not None:
        plt.xlabel(np.sum(example))
        plt.ylabel(title)
    if plotname is not None:
        plt.savefig(plotname, bbox_inches='tight')


def main(batch_size, maxplots, nfloats, prefix):
    datafile = prefix + 'encoded_test.hdf5'
    ones, zeros = make_encoded_dataset(datafile, nfloats, batch_size)
    for cat, dset in zip([0, 1], [zeros, ones]):
        for i, images in enumerate(dset):
            plotname = '{}_batch{:03d}_cat{}.pdf'.format(prefix, i, cat)
            vis_data(images.numpy(), reshape_shape=(-1, nfloats), title=cat,
                     plotname=plotname)
            if i > maxplots:
                break


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

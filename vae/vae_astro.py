#!/usr/bin/env python
# coding: utf-8
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import tf2
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()

from tfvae.data_readers import make_astro_dset
from tfvae.hdf5_utils import setup_hdf5
from tfvae.hdf5_utils import add_batch_to_hdf5
from tfvae.misc_fns import generate_and_save_images
from tfvae.misc_fns import get_expected_image_shape
from tfvae.train_fns import apply_gradients
from tfvae.train_fns import compute_gradients
from tfvae.train_fns import compute_loss
from tfvae.tfk_models import CVAE


data_dir = "/Users/perdue/Dropbox/Quantum_Computing/hep-qml/data/cae_splits"
data_type = "stargalaxy_sim_20190214"
test_file = os.path.join(data_dir, data_type + '_cae_test.h5')
train_file = os.path.join(data_dir, data_type + '_cae_train.h5')
train_dataset = make_astro_dset(train_file, batch_size=32, shuffle=True)
test_dataset = make_astro_dset(test_file, batch_size=1, mode='with_labels',
                               augment=False)

input_shape = get_expected_image_shape(train_file)
print('input shape for image type {} is {}'.format(data_type, input_shape))

optimizer = tf.keras.optimizers.Adam(5e-5)
epochs = 10
latent_dim = 49
num_exmaples_to_generate = 16

# keep the random vector constant for generation to track improvements
random_vector_for_generation = tf.random.normal(
    shape=[num_exmaples_to_generate, latent_dim]
)

model = CVAE(latent_dim, input_shape=input_shape)
generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x, _ in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('epoch: {}, test set ELBO: {}, time elapsed {}'.format(
            epoch, elbo, end_time - start_time
        ))
        generate_and_save_images(model, epoch, random_vector_for_generation)

print('encoding output hdf5')
encoded_file_name = 'stargalaxy_sim_20190214encoded_test.hdf5'
n_encoded, n_labels = latent_dim, 1
f = setup_hdf5(encoded_file_name, n_encoded, n_labels)
for x, y in test_dataset:
    xmean, xlogvar = model.encode(x)
    z = model.reparameterize(xmean, xlogvar)
    add_batch_to_hdf5(f, z.numpy(),
                      np.argmax(y.numpy(), axis=1).reshape(-1, 1))
print('finished encoding')

for i, (x, y) in enumerate(test_dataset.take(10)):
    xmean, xlogvar = model.encode(x)
    z = model.reparameterize(xmean, xlogvar)
    print(z.shape)
    x_decoded = model.decode(z)
    print(x_decoded.shape)

    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = plt.subplot(gs[0])
    ax1.imshow(x_decoded.numpy().reshape((input_shape[0], input_shape[1])))
    ax2 = plt.subplot(gs[1])
    ax2.imshow(x.numpy().reshape((input_shape[0], input_shape[1])))
    plt.title(np.argmax(y.numpy(), axis=1).reshape(-1, 1))
    ax3 = plt.subplot(gs[2])
    ax3.imshow(z.numpy().reshape(7, 7))
    figname = 'image_{:04d}.pdf'.format(i)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

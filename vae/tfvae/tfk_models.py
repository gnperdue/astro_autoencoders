import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


class CVAE(tf.keras.Model):

    def __init__(self, latent_dim, input_shape):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tfk.Sequential([
            tfkl.InputLayer(input_shape=input_shape),
            tfkl.Conv2D(filters=32, kernel_size=3,
                        strides=(2, 2), activation='relu'),
            tfkl.Conv2D(filters=64, kernel_size=3,
                        strides=(2, 2), activation='relu'),
            tfkl.Flatten(),
            tfkl.Dense(latent_dim + latent_dim)

        ])
        # TODO: will need to condition based on input_shape
        self.generative_net = tfk.Sequential([
            tfkl.InputLayer(input_shape=(latent_dim,)),
            tfkl.Dense(units=16*16*6, activation=tf.nn.relu),
            tfkl.Reshape(target_shape=(16, 16, 6)),
            tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2),
                                 padding='SAME', activation='relu'),
            tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2),
                                 padding='SAME', activation='relu'),
            tfkl.Conv2DTranspose(filters=1, kernel_size=3,
                                 strides=(1, 1), padding='SAME'),
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x),
                                num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

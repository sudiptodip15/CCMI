import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, y_dim = 1, z_dim = 20, h_dim = 256):
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.name = 'cgan/mlp/d_net'

    def __call__(self, y, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            inp = tf.concat([y, z], 1)
            fc1 = tc.layers.fully_connected(
                inp, self.h_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #fc1 = leaky_relu(tc.layers.batch_norm(fc1))
            fc1 = leaky_relu(fc1)
            fc1 = tf.concat([fc1, z], 1)

            fc2 = tc.layers.fully_connected(
                fc1, self.h_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            #fc2 = leaky_relu(tc.layers.batch_norm(fc2))
            fc2 = leaky_relu(fc2)
            fc2 = tf.concat([fc2, z], 1)

            fc3 = tc.layers.fully_connected(fc2, 1,
                                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                            activation_fn=tf.identity
                                            )
            return tf.nn.sigmoid(fc3), fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, s_dim = 10, z_dim=20, y_dim = 1, h_dim =256):
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.name = 'cgan/mlp/g_net'

    def __call__(self, s, z):
        with tf.variable_scope(self.name) as vs:

            inp = tf.concat([s, z], 1)
            fc1 = tcl.fully_connected(
                inp, self.h_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #fc1 = leaky_relu(tc.layers.batch_norm(fc1))
            fc1 = leaky_relu(fc1)
            fc1 = tf.concat([fc1, z],1)

            fc2 = tcl.fully_connected(
                fc1, self.h_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #fc2 = leaky_relu(tc.layers.batch_norm(fc2))
            fc2 = leaky_relu(fc2)
            fc2 = tf.concat([fc2, z], 1)

            fc3 = tc.layers.fully_connected(
                fc2, self.y_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
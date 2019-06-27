import tensorflow as tf
import numpy as np

class CGAN_mimic(object):
    def __init__(self, g_net, d_net, Z_train, Y_train, x_dim, y_dim, z_dim, batch_size, bn = False):

        self.g_net = g_net
        self.d_net = d_net

        self.Z_train = Z_train
        self.Y_train = Y_train
        self.data_size = len(Z_train)
        self.batch_size = batch_size
        self.bn = bn
        self.model_name = 'mimic_cgan'

        self.s_dim = self.g_net.s_dim
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim

        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name='s')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_fake = self.g_net(self.s, self.z)

        self.d_prob, self.d_logits = self.d_net(self.y, self.z,  reuse=False)
        self.d_fake_prob, self.d_fake_logits = self.d_net(self.y_fake, self.z)

        d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_fake_prob), logits=self.d_fake_logits))

        d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_prob), logits=self.d_logits))
        g_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake_prob), logits=self.d_fake_logits))

        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = g_fake_loss

        self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.d_loss, var_list=self.d_net.vars)
        self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.g_loss, var_list=self.g_net.vars)


    def sample_train(self, batch_size):
        indx = np.random.randint(low=0, high=self.data_size, size=batch_size)
        return self.Z_train[indx, :], self.Y_train[indx]

    def sample_noise(self, batch_size):
        return np.random.uniform(-1.0, 1.0, [batch_size, self.s_dim])

    def train(self, Z_marginal):

        max_ep = 100
        batch_size = self.batch_size
        max_it = max_ep * self.data_size // batch_size

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.33
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:

            sess.run(tf.global_variables_initializer())

            for t in range(0, max_it):

                bz, by = self.sample_train(batch_size)
                bs = self.sample_noise(batch_size)
                sess.run(self.d_adam, feed_dict={self.y: by, self.z: bz, self.s : bs})

                sess.run(self.g_adam, feed_dict={self.z: bz, self.s : bs})

            Y_marginal = self.gen_cond_samples(sess, Z_marginal)
        return Y_marginal


    def gen_cond_samples(self, sess, Z_marginal):

        B = Z_marginal.shape[0]

        if self.bn == 'False':
            bs = self.sample_noise(B)
            y_gen = sess.run(self.y_fake, feed_dict={self.z: Z_marginal, self.s: bs})
        else:
            num_batches = (B // self.batch_size)
            y_gen = np.zeros(((num_batches + 1) * self.batch_size, self.y_dim))
            for b in range(num_batches):
                bs = self.sample_noise(self.batch_size)
                y_gen_batch = sess.run(self.y_fake, feed_dict={self.z: Z_marginal[b* self.batch_size:(b+1)*self.batch_size,:], self.s: bs})
                y_gen[b*self.batch_size:(b+1)*self.batch_size,:] = y_gen_batch

            bs = self.sample_noise(self.batch_size)
            y_gen_batch = sess.run(self.y_fake, feed_dict={self.z: Z_marginal[-self.batch_size:, :], self.s: bs})
            y_gen[num_batches * self.batch_size:(num_batches + 1) * self.batch_size, :] = y_gen_batch
            y_gen = y_gen[0:B, :]

        print('Generated CGAN Conditional samples !')
        return y_gen


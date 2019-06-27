import tensorflow as tf
import numpy as np

class CVAE_mimic(object):
    def __init__(self, enc_net, dec_net, Z_train, Y_train, x_dim, y_dim, z_dim, batch_size, bn = False):

        self.enc_net = enc_net
        self.dec_net = dec_net

        self.Z_train = Z_train
        self.Y_train = Y_train
        self.data_size = len(Z_train)
        self.batch_size = batch_size
        self.bn = bn
        self.model_name = 'mimic_cvae'

        self.s_dim = self.enc_net.s_dim
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim


        self.s_infer = tf.placeholder(tf.float32, [None, self.s_dim], name='s')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        mu_z, log_sigma_sq_z = self.enc_net(self.y, self.z, self.keep_prob)
        noise_z = tf.random_normal(tf.shape(mu_z), 0, 1, dtype=tf.float32)
        s_train = mu_z + tf.exp(0.5 * log_sigma_sq_z) * noise_z

        mu_y_hat, log_sigma_sq_y_hat = self.dec_net(s_train, self.z, self.keep_prob, reuse=False)

        self.recon_loss = 0.5 * tf.reduce_sum(
            tf.square(self.y - mu_y_hat) / tf.exp(log_sigma_sq_y_hat) + log_sigma_sq_y_hat, axis=1)

        self.KL_loss = 0.5 * tf.reduce_sum(tf.exp(log_sigma_sq_z) + tf.square(mu_z) - 1. - log_sigma_sq_z, axis=1)

        self.ELBO = tf.reduce_mean(self.recon_loss + self.KL_loss)

        self.opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.ELBO)

        mu_y_gen, log_sigma_sq_y_gen = self.dec_net(self.s_infer, self.z, 1.0)
        noise_y_gen = tf.random_normal(tf.shape(mu_y_gen), 0, 1, dtype=tf.float32)
        self.y_gen = mu_y_gen + tf.exp(0.5 * log_sigma_sq_y_gen) * noise_y_gen


    def sample_train(self, batch_size):
        indx = np.random.randint(low=0, high=self.data_size, size=batch_size)
        return self.Z_train[indx, :], self.Y_train[indx]

    def train(self, Z_marginal):

        max_ep = 20
        batch_size = self.batch_size
        max_it = max_ep * self.data_size // batch_size

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.33
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:

            sess.run(tf.global_variables_initializer())

            for t in range(0, max_it):

                bz, by = self.sample_train(batch_size)
                sess.run(self.opt, feed_dict={self.y: by, self.z: bz, self.keep_prob: 0.9})

            Y_marginal = self.gen_cond_samples(sess, Z_marginal)

        return Y_marginal


    def gen_cond_samples(self, sess, Z_marginal):

        B = Z_marginal.shape[0]
        S = np.random.normal(loc=0.0, scale=1.0, size=(B, self.s_dim))

        if self.bn == 'False':
            y_gen = sess.run(self.y_gen, feed_dict={self.z: Z_marginal, self.s_infer: S})

        else:
            num_batches = (B // self.batch_size)
            y_gen = np.zeros(((num_batches + 1) * self.batch_size, self.y_dim))
            for b in range(num_batches):
                bs = S[b* self.batch_size:(b+1)*self.batch_size,:]
                bz = Z_marginal[b* self.batch_size:(b+1)*self.batch_size,:]
                y_gen_batch = sess.run(self.y_gen, feed_dict={self.z: bz, self.s_infer: bs})
                y_gen[b*self.batch_size:(b+1)*self.batch_size,:] = y_gen_batch

            bs = S[-self.batch_size:, :]
            bz = Z_marginal[-self.batch_size:, :]
            y_gen_batch = sess.run(self.y_gen, feed_dict={self.z: bz, self.s_infer: bs})
            y_gen[num_batches * self.batch_size:(num_batches + 1) * self.batch_size, :] = y_gen_batch
            y_gen = y_gen[0:B, :]

        print('Generated CVAE Conditional samples !')
        return y_gen


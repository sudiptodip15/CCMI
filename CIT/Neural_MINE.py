import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from cit_util import *

class Neural_MINE(object):

    def __init__(self, data_train, data_eval, dx, h_dim = 64, actv = tf.nn.relu, batch_size = 128,
                 optimizer = 'adam', lr = 0.0001, max_ep = 200, mon_freq = 5000, metric = 'f_divergence'):

        self.dim_x = dx
        self.data_dim = data_train.shape[1]
        self.X = data_train[:, 0:dx]
        self.Y = data_train[:, dx:]
        self.train_size = len(data_train)

        self.X_eval = data_eval[:, 0:dx]
        self.Y_eval = data_eval[:, dx:]
        self.eval_size = len(data_eval)

        # Hyper-parameters of statistical network
        self.h_dim = h_dim
        self.actv = actv

        # Hyper-parameters of training process
        self.batch_size = batch_size
        self.max_iter = int(max_ep * self.train_size / batch_size)
        self.optimizer = optimizer
        self.lr = lr
        self.mon_freq = mon_freq
        self.metric = metric
        self.tol = 1e-4

    def sample_p_finite(self, batch_size):
        index = np.random.randint(low = 0, high = self.train_size, size=batch_size)
        return np.hstack((self.X[index, :], self.Y[index, :]))

    def stat_net(self, inp, reuse=False):
        with tf.variable_scope('func_approx') as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tcl.fully_connected(inp, num_outputs=self.h_dim, activation_fn=self.actv,
                                      weights_initializer=tf.orthogonal_initializer)
            out = tcl.fully_connected(fc1, num_outputs=1, activation_fn=tf.identity,
                                      weights_initializer=tf.orthogonal_initializer)
            return out

    def get_div(self, stat_inp_p, stat_inp_q):

        if self.metric == 'donsker_varadhan':
            return log_mean_exp_tf(stat_inp_q) - tf.reduce_mean(stat_inp_p)
        elif self.metric == 'f_divergence':
            return tf.reduce_mean(tf.exp(stat_inp_q - 1)) - tf.reduce_mean(stat_inp_p)
        else:
            raise NotImplementedError


    def train(self):

        # Define nodes for training process
        Inp_p = tf.placeholder(dtype=tf.float32, shape=[None, self.data_dim], name='Inp_p')
        finp_p = self.stat_net(Inp_p)

        Inp_q = tf.placeholder(dtype=tf.float32, shape=[None, self.data_dim], name='Inp_q')
        finp_q = self.stat_net(Inp_q, reuse=True)

        loss = self.get_div(finp_p, finp_q)
        mi_t = -loss

        if self.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(loss)
        elif self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1 = 0.5, beta2 = 0.999).minimize(loss)

        mi_with_iter = []
        #print('Estimating MI with metric = {}, opt = {}, lr = {}'.format(self.metric, self.optimizer, self.lr))
        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.33
        run_config.gpu_options.allow_growth = True
        with tf.Session(config = run_config) as sess:

            sess.run(tf.global_variables_initializer())

            eval_inp_p = np.hstack((self.X_eval, self.Y_eval))
            eval_inp_q = shuffle(eval_inp_p, self.dim_x)
            mi_est = -np.inf
            for it in range(self.max_iter):

                batch_inp_p = self.sample_p_finite(self.batch_size)
                batch_inp_q = shuffle(batch_inp_p, self.dim_x)

                MI, _ = sess.run([mi_t, opt], feed_dict={Inp_p: batch_inp_p, Inp_q: batch_inp_q})

                if ((it + 1) % self.mon_freq == 0 or (it + 1) == self.max_iter):

                    prev_est = mi_est
                    mi_est = sess.run(mi_t, feed_dict={Inp_p: eval_inp_p, Inp_q: eval_inp_q})
                    mi_with_iter.append(mi_est)

                    #print('Iter [%8d] : MI_est = %.4f' % (it + 1, mi_est))

                    if abs(prev_est - mi_est) < self.tol:
                        break

            # mi_with_iter = smooth_ma(mi_with_iter)
            # mi_est = mi_with_iter[-1]

            return mi_est, mi_with_iter



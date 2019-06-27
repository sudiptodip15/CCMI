import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from cit_util import *

class Neural_MINE(object):

    def __init__(self, data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal,
                 dx, h_dim = 64, actv = tf.nn.relu, batch_size = 128,
                 optimizer = 'adam', lr = 0.0001, max_ep = 200, mon_freq = 5000, metric = 'f_divergence'):

        self.dim_x = dx
        self.data_dim = data_train_joint.shape[1]
        self.train_size = len(data_train_joint)
        self.eval_size = len(data_eval_joint)

        self.data_train_joint = data_train_joint
        self.data_train_marginal = data_train_marginal
        self.data_eval_joint = data_eval_joint
        self.data_eval_marginal = data_eval_marginal

        # Hyper-parameters of statistical network
        self.h_dim = h_dim
        self.actv = actv

        # Hyper-parameters of training process
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = int(max_ep * self.train_size / batch_size)
        self.mon_freq = mon_freq
        self.metric = metric
        self.tol = 1e-4

    def sample_pq_finite(self, batch_size):
        index = np.random.randint(low = 0, high = self.train_size, size=batch_size)
        return self.data_train_joint[index, :], self.data_train_marginal[index, :]

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
        div_t = -loss

        if self.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(loss)
        elif self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1 = 0.5, beta2 = 0.999).minimize(loss)

        div_with_iter = []
        # print('Estimating MI with metric = {}, opt = {}, lr = {}'.format(self.metric, self.optimizer, self.lr))

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.33
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:

            sess.run(tf.global_variables_initializer())

            eval_inp_p = self.data_eval_joint
            eval_inp_q = self.data_eval_marginal
            div_est = -np.inf
            for it in range(self.max_iter):

                batch_inp_p, batch_inp_q  = self.sample_pq_finite(self.batch_size)

                D_KL, _ = sess.run([div_t, opt], feed_dict={Inp_p: batch_inp_p, Inp_q: batch_inp_q})

                if ((it + 1) % self.mon_freq == 0 or (it + 1) == self.max_iter):

                    prev_est = div_est
                    div_est = sess.run(div_t, feed_dict={Inp_p: eval_inp_p, Inp_q: eval_inp_q})
                    div_with_iter.append(div_est)

                    if abs(prev_est - div_est) < self.tol:
                        break

            return div_est



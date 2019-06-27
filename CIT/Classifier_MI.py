import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from cit_util import *


class Classifier_MI(object):

    def __init__(self, data_train, data_eval, dx, h_dim = 256, actv = tf.nn.relu, batch_size = 64,
                 optimizer='adam', lr=0.001, max_ep = 20, mon_freq = 5000,  metric = 'donsker_varadhan'):

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
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = int(max_ep * self.train_size / batch_size)
        self.mon_freq = mon_freq
        self.metric = metric
        self.tol = 1e-4
        self.eps = 1e-8


        #Post Non-linear Cosine Data-set : h_dim = (256, 256), actv = tf.nn.relu, batch_size = 64, adam, lr = 0.001 (default momentum's), max_ep = 20, num_boot_iter = 10 
        #Flow-Cytometry Data-set : h_dim = (64, 64), actv = tf.nn.relu, batch_size = 64, adam, lr = 0.001 (default momentum's), max_ep = 10, num_boot_iter = 20.

        self.reg_coeff = 1e-3

    def sample_p_finite(self, batch_size):
        index = np.random.randint(low = 0, high = self.train_size, size=batch_size)
        return np.hstack((self.X[index, :], self.Y[index, :]))


    def classifier(self, inp, reuse = False):

        with tf.variable_scope('func_approx') as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tcl.fully_connected(inp, num_outputs = self.h_dim, activation_fn=self.actv,
                                      weights_regularizer=tcl.l2_regularizer(self.reg_coeff))
            fc2 = tcl.fully_connected(fc1, num_outputs = self.h_dim, activation_fn=self.actv,
                                      weights_regularizer=tcl.l2_regularizer(self.reg_coeff))
            logit = tcl.fully_connected(fc2, num_outputs = 1, activation_fn=None,
                                        weights_regularizer=tcl.l2_regularizer(self.reg_coeff))
            prob = tf.nn.sigmoid(logit)

            return logit, prob

    def train_classifier_MLP(self):

        # Define tensorflow nodes for classifier
        Inp = tf.placeholder(dtype=tf.float32, shape=[None, self.data_dim], name='Inp')
        label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='label')

        logit, y_prob = self.classifier(Inp)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
        l2_loss = tf.losses.get_regularization_loss()
        cost = tf.reduce_mean(cross_entropy) + l2_loss

        y_hat = tf.round(y_prob)
        correct_pred = tf.equal(y_hat, label)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if self.optimizer == 'sgd':
            opt_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)
        elif self.optimizer == 'adam':
            opt_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        #print('Training MLP classifier on Two-sample, opt = {}, lr = {}'.format(self.optimizer, self.lr))
        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.33
        run_config.gpu_options.allow_growth = True
        with tf.Session(config = run_config) as sess:

            sess.run(tf.global_variables_initializer())

            eval_inp_p = np.hstack((self.X_eval, self.Y_eval))
            eval_inp_q = shuffle(eval_inp_p, self.dim_x)
            B = len(eval_inp_p)

            for it in range(self.max_iter):

                batch_inp_p = self.sample_p_finite(self.batch_size)
                batch_inp_q = shuffle(batch_inp_p, self.dim_x)

                batch_inp = np.vstack((batch_inp_p, batch_inp_q))
                by = np.vstack((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))))
                batch_index = np.random.permutation(2*self.batch_size)
                batch_inp = batch_inp[batch_index]
                by = by[batch_index]

                L, _ = sess.run([cost, opt_step], feed_dict={Inp: batch_inp, label: by})

                if ((it + 1) % self.mon_freq == 0):

                    eval_inp = np.vstack((eval_inp_p, eval_inp_q))
                    eval_y = np.vstack((np.ones((B, 1)), np.zeros((B, 1))))
                    eval_acc = sess.run(accuracy, feed_dict={Inp: eval_inp, label: eval_y})
                    print('Iteraion = {}, Test accuracy = {}'.format(it+1, eval_acc))

            pos_label_pred_p = sess.run(y_prob, feed_dict={Inp: eval_inp_p})
            rn_est_p = (pos_label_pred_p+self.eps)/(1-pos_label_pred_p-self.eps)
            finp_p = np.log(np.abs(rn_est_p))

            pos_label_pred_q = sess.run(y_prob, feed_dict={Inp: eval_inp_q})
            rn_est_q = (pos_label_pred_q + self.eps) / (1 - pos_label_pred_q - self.eps)
            finp_q = np.log(np.abs(rn_est_q))

            #mi_est = np.mean(finp_p) - np.log(np.mean(np.exp(finp_q)))
            mi_est = np.mean(finp_p) - log_mean_exp_numpy(finp_q)

        return mi_est

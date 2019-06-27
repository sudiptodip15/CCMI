import numpy as np
import tensorflow as tf
import random
from Neural_MINE import Neural_MINE
from Classifier_MI import Classifier_MI


class CCMI(object):
    def __init__(self, X, Y, Z, tester, metric, num_boot_iter, h_dim, max_ep):

        self.dim_x = X.shape[1]
        self.dim_y = Y.shape[1]
        self.dim_z = Z.shape[1]
        self.data_xyz = np.hstack((X, Y, Z))
        self.data_xz = np.hstack((X, Z))
        self.threshold = 1e-4

        self.tester = tester
        self.metric = metric
        self.num_boot_iter = num_boot_iter
        self.h_dim = h_dim
        self.max_ep = max_ep

    def split_train_test(self, data):
        total_size = data.shape[0]
        train_size = int(2*total_size/3)
        data_train = data[0:train_size,:]
        data_test = data[train_size:, :]
        return data_train, data_test

    def gen_bootstrap(self, data):
        np.random.seed()
        random.seed()
        num_samp = data.shape[0]
        #I = np.random.choice(num_samp, size=num_samp, replace=True)
        I = np.random.permutation(num_samp)
        data_new = data[I, :]
        return data_new

    def get_cmi_est(self):

        if self.tester == 'Neural':
            print('Tester = {}, metric = {}'.format(self.tester, self.metric))
            if self.metric == 'donsker_varadhan':
                batch_size = 512
            else:
                batch_size = 128
            I_xyz_list = []
            for t in range(self.num_boot_iter):
                tf.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xyz)
                data_xyz_train, data_xyz_eval = self.split_train_test(data_t)
                neurMINE_xyz = Neural_MINE(data_xyz_train, data_xyz_eval, self.dim_x,
                                           metric= self.metric, batch_size = batch_size)
                I_xyz_t, _ = neurMINE_xyz.train()
                I_xyz_list.append(I_xyz_t)

            I_xyz_list = np.array(I_xyz_list)
            I_xyz = np.mean(I_xyz_list)

            I_xz_list = []
            for i in range(self.num_boot_iter):
                tf.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xz)
                data_xz_train, data_xz_eval = self.split_train_test(data_t)
                neurMINE_xz = Neural_MINE(data_xz_train, data_xz_eval, self.dim_x,
                                          metric= self.metric, batch_size = batch_size)
                I_xz_t, _ = neurMINE_xz.train()
                I_xz_list.append(I_xz_t)

            I_xz_list = np.array(I_xz_list)
            I_xz = np.mean(I_xz_list)
            cmi_est = I_xyz - I_xz

        elif self.tester == 'Classifier':
            print('Tester = {}, metric = {}'.format(self.tester, self.metric))
            I_xyz_list = []
            for t in range(self.num_boot_iter):
                tf.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xyz)
                data_xyz_train, data_xyz_eval = self.split_train_test(data_t)
                classMINE_xyz = Classifier_MI(data_xyz_train, data_xyz_eval, self.dim_x,
                                              h_dim = self.h_dim, max_ep = self.max_ep)
                I_xyz_t = classMINE_xyz.train_classifier_MLP()
                I_xyz_list.append(I_xyz_t)

            I_xyz_list = np.array(I_xyz_list)
            I_xyz = np.mean(I_xyz_list)

            I_xz_list = []
            for i in range(self.num_boot_iter):
                tf.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xz)
                data_xz_train, data_xz_eval = self.split_train_test(data_t)
                classMINE_xz = Classifier_MI(data_xz_train, data_xz_eval, self.dim_x,
                                              h_dim = self.h_dim, max_ep = self.max_ep)
                I_xz_t = classMINE_xz.train_classifier_MLP()
                I_xz_list.append(I_xz_t)

            I_xz_list = np.array(I_xz_list)
            I_xz = np.mean(I_xz_list)
            cmi_est = I_xyz - I_xz
        else:
            raise NotImplementedError

        return cmi_est

    def is_indp(self, cmi_est):
          if max(0, cmi_est) < self.threshold:
              return True
          else:
              return False





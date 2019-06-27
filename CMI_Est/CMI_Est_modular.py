# Obtain samples from the joint p = p(x, y, z)  and the marginals q = p(x, z) p(y | z). Then compute divergence.

import numpy as np
import random
import tensorflow as tf
import importlib

from sklearn.neighbors import NearestNeighbors

eps = 1e-8

from CVAE_mimic import CVAE_mimic
from CGAN_mimic import CGAN_mimic
from Neural_MINE import Neural_MINE
from Classifier_MI import Classifier_MI


def split_XYZ(data, dx, dy):
    X = data[:, 0:dx]
    Y = data[:, dx:dx+dy]
    Z = data[:, dx+dy:]
    return X, Y, Z

def split_train_test(data):
    total_size = data.shape[0]
    train_size = int(2 * total_size / 3)
    data_train = data[0:train_size, :]
    data_test = data[train_size:, :]
    return data_train, data_test


def normalize_data(data):
    data_norm = (data - np.mean(data, axis = 0))/(np.std(data, axis = 0))
    return data_norm

def gen_bootstrap(data):

    np.random.seed()
    random.seed()
    num_samp = data.shape[0]
    I = np.random.permutation(num_samp)
    data_new = data[I, :]
    return data_new

def mimic_cvae(data_mimic, dx, dy, dz, Z_marginal):
    s_dim = 20
    h_dim = 256
    _, Y_train, Z_train = split_XYZ(data_mimic, dx, dy)
    model = importlib.import_module('model' + '.' + 'cvae_mlp')
    enc_net = model.Encoder(y_dim=dy, z_dim=dz, s_dim=s_dim, h_dim = h_dim)
    dec_net = model.Decoder(s_dim=s_dim, z_dim=dz, y_dim=dy, h_dim = h_dim)
    cvae = CVAE_mimic(enc_net, dec_net, Z_train, Y_train, dx, dy, dz, batch_size = 128)
    Y_marginal = cvae.train(Z_marginal)
    return Y_marginal


def mimic_cgan(data_mimic, dx, dy, dz, Z_marginal):
    s_dim = 20
    h_dim = 256
    _, Y_train, Z_train = split_XYZ(data_mimic, dx, dy)
    model = importlib.import_module('model' + '.' + 'cgan_mlp')
    d_net = model.Discriminator(y_dim=dy, z_dim=dz, h_dim = h_dim)
    g_net = model.Generator(s_dim=s_dim, z_dim=dz, y_dim=dy, h_dim = h_dim)
    cgan = CGAN_mimic(g_net, d_net, Z_train, Y_train, dx, dy, dz, batch_size = 128)
    Y_marginal = cgan.train(Z_marginal)
    return Y_marginal

def mimic_knn(data_mimic, dx, dy, dz, Z_marginal):

    _, Y_train, Z_train  = split_XYZ(data_mimic, dx, dy)
    nbrs = NearestNeighbors(n_neighbors=1).fit(Z_train)
    indx = nbrs.kneighbors(Z_marginal, return_distance=False).flatten()
    Y_marginal = Y_train[indx, :]
    print('Generated 1-NN Conditional samples !')
    return Y_marginal


def shuffle_y(data, dx):
    X = data[:,0:dx]
    Y = data[:,dx:]
    Y = np.random.permutation(Y)
    return np.hstack((X, Y))


def CMI_Est_modular(data, x_dim, y_dim, z_dim, mimic, tester, metric, cmi_true, outer_boot_iter, inner_dup_iter):

    data = normalize_data(data)
    cmi_est_it = []

    for i in range(outer_boot_iter):
        data = gen_bootstrap(data)

        #Compute CMI as Divergence if mimic \in {knn, cgan, cvae}
        if mimic == 'mi_diff':
            data_mine = data
            data_marginal = shuffle_y(data, x_dim)

            I_xyz_list = []
            I_xz_list = []
            for j in range(inner_dup_iter):

                data_train_joint, data_eval_joint = split_train_test(data_mine)
                data_train_marginal, data_eval_marginal = split_train_test(data_marginal)

                tf.reset_default_graph()
                if tester == 'Neural':
                    neur_mine_xyz = Neural_MINE(data_train_joint, data_eval_joint,
                                                data_train_marginal, data_eval_marginal, x_dim, metric=metric)
                    I_xyz_t = neur_mine_xyz.train()
                elif tester == 'Classifier':
                    class_mlp_mi_xyz = Classifier_MI(data_train_joint, data_eval_joint,
                                                     data_train_marginal, data_eval_marginal, x_dim, metric=metric)
                    I_xyz_t = class_mlp_mi_xyz.train_classifier_MLP()

                else:
                    raise NotImplementedError

                I_xyz_list.append(I_xyz_t)

                # De-bias the Estimate
                X, Y, Z = split_XYZ(data_mine, x_dim, y_dim)
                data_mine_xz = np.hstack((X, Z))
                data_marginal_xz = shuffle_y(data_mine_xz, x_dim)

                data_train_joint_xz, data_eval_joint_xz = split_train_test(data_mine_xz)
                data_train_marginal_xz, data_eval_marginal_xz = split_train_test(data_marginal_xz)

                tf.reset_default_graph()
                if tester == 'Neural':
                    neur_mine_xz = Neural_MINE(data_train_joint_xz, data_eval_joint_xz,
                                               data_train_marginal_xz, data_eval_marginal_xz, y_dim, metric=metric)
                    I_xz_t = neur_mine_xz.train()
                elif tester == 'Classifier':
                    class_mlp_mi_xz = Classifier_MI(data_train_joint_xz, data_eval_joint_xz,
                                                    data_train_marginal_xz, data_eval_marginal_xz, y_dim, metric=metric)
                    I_xz_t = class_mlp_mi_xz.train_classifier_MLP()

                else:
                    raise NotImplementedError
                I_xz_list.append(I_xz_t)
                print('I(X;YZ) = {}, I(X;Z) = {}'.format(I_xyz_t, I_xz_t))

            I_xyz = np.mean(np.array(I_xyz_list))
            I_xz = np.mean(np.array(I_xz_list))

            cmi_est_t = I_xyz - I_xz
            cmi_est_it.append(cmi_est_t)

        else:
            # Split data into two parts for Mimic and Mine.
            mimic_size = int(len(data)/2)
            data_mimic = data[0:mimic_size,:]
            data_mine = data[mimic_size:,:]
            X, Y, Z = split_XYZ(data_mine, x_dim, y_dim)

            # Train Mimic block to obtain q(y|z)
            debias = False
            if mimic == 'cvae':
                Y_marginal = mimic_cvae(data_mimic, x_dim, y_dim, z_dim, Z)
            elif mimic == 'cgan':
                Y_marginal = mimic_cgan(data_mimic, x_dim, y_dim, z_dim, Z)
            elif mimic == 'knn':
                Y_marginal = mimic_knn(data_mimic, x_dim, y_dim, z_dim, Z)
            else:
                raise NotImplementedError

            data_marginal = np.hstack((X, Y_marginal, Z))

            div_xyz_list = []
            div_yz_list = []
            for j in range(inner_dup_iter):

                data_train_joint, data_eval_joint = split_train_test(data_mine)
                data_train_marginal, data_eval_marginal = split_train_test(data_marginal)

                tf.reset_default_graph()
                if tester == 'Neural':
                    neur_mine_xyz = Neural_MINE(data_train_joint, data_eval_joint,
                                            data_train_marginal, data_eval_marginal, x_dim, metric = metric)
                    div_xyz_t = neur_mine_xyz.train()
                elif tester == 'Classifier':
                    class_mlp_mi_xyz = Classifier_MI(data_train_joint, data_eval_joint,
                                            data_train_marginal, data_eval_marginal, x_dim, metric=metric)
                    div_xyz_t = class_mlp_mi_xyz.train_classifier_MLP()

                else:
                    raise NotImplementedError

                div_xyz_list.append(div_xyz_t)

                # De-bias the Estimate
                if debias :
                    data_mine_yz = data_mine[:, x_dim:]
                    data_marginal_yz = data_marginal[:, x_dim:]

                    data_train_joint_yz, data_eval_joint_yz = split_train_test(data_mine_yz)
                    data_train_marginal_yz, data_eval_marginal_yz = split_train_test(data_marginal_yz)

                    tf.reset_default_graph()
                    if tester == 'Neural':
                        neur_mine_yz = Neural_MINE(data_train_joint_yz, data_eval_joint_yz,
                                                data_train_marginal_yz, data_eval_marginal_yz, y_dim, metric=metric)
                        div_yz_t = neur_mine_yz.train()
                    elif tester == 'Classifier':
                        class_mlp_mi_yz = Classifier_MI(data_train_joint_yz, data_eval_joint_yz,
                                                     data_train_marginal_yz, data_eval_marginal_yz, y_dim, metric=metric)
                        div_yz_t = class_mlp_mi_yz.train_classifier_MLP()

                    else:
                        raise NotImplementedError

                    div_yz_list.append(div_yz_t)
                    print('D_KL(XYZ) = {}, D_KL(YZ) = {}'.format(div_xyz_t, div_yz_t))

                else:
                    div_yz_list.append(0.0)
                    print('D_KL(XYZ) = {}'.format(div_xyz_t))


            div_xyz = np.mean(np.array(div_xyz_list))
            div_yz = np.mean(np.array(div_yz_list))

            cmi_est_t = div_xyz - div_yz
            cmi_est_it.append(cmi_est_t)


    cmi_est_it = np.array(cmi_est_it)
    cmi_est_mu = np.mean(cmi_est_it)
    cmi_est_std = np.sqrt(np.mean((cmi_est_it - cmi_est_mu)**2))
    return cmi_est_mu, cmi_est_std







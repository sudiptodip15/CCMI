import argparse
import numpy as np
import NPEET.entropy_estimators as ee
from math import log
import os

from func_def import *

def save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed):
    data = np.hstack((x, y, z))
    save_dir = './cat{}'.format(syn_file_cat)
    if not os.path.exists(save_dir):
       os.makedirs(save_dir)
    np.save('./cat{}/data.{}k.dz{}.seed{}.npy'.format(syn_file_cat, num_th, z_dim, seed), data)

def save_ksg_est_fz(x, y, fz, syn_file_cat, z_dim):
    cmi_est_base2 = ee.cmi(x, y, fz)
    cmi_est = cmi_est_base2 * log(2)
    cmi_est = np.abs(cmi_est)
    print('CMI = {}'.format(cmi_est))
    save_dir = './cat{}'.format(syn_file_cat)
    if not os.path.exists(save_dir):
       os.makedirs(save_dir)
    np.save('./cat{}/ksg_gt.dz{}.npy'.format(syn_file_cat, z_dim), np.array([cmi_est]))


def gen_cmi_data_var_gauss_channel(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # X ~ N(0, 1)
    # Z ~ U(0.5, 1.5)
    # Y ~ X + n, where n ~ N(0, z^2)
    # Category A.

    print('Category A')

    x = np.random.normal(loc = 0, scale = 1, size = (N, x_dim))
    z = np.random.uniform(low = 0.5, high = 1.5, size = (N, z_dim))
    noise = np.zeros((N, y_dim))
    for i in range(N):
         temp = np.random.normal(loc = 0.0, scale = z[i,:])
         noise[i,:] = temp
    y = x + noise

    save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed)

    save_ksg_est_fz(x, y, z, syn_file_cat, z_dim)



def gen_cmi_data_var_sparse_dependency(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # X ~ N(0, 1)
    # Z ~ [U(0.5, 1.5)]^d_z
    # Y ~ X + n, where n ~ N(0, z_1^2)
    # Category B.

    print('Category B')

    x = np.random.normal(loc=0, scale=1, size=(N, x_dim))
    z = np.random.uniform(low=0.5, high=1.5, size=(N, z_dim))

    noise = np.zeros((N, y_dim))
    for i in range(N):
        temp = np.random.normal(loc=0.0, scale=z[i, 0])
        noise[i, :] = temp
    y = x + noise

    save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed)

    save_ksg_est_fz(x, y, z[:, 0].reshape(-1, 1), syn_file_cat, z_dim)


def gen_cmi_data_var_lin_comb_Z(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # X ~ N(0, 1)
    # Z ~ U(0.5, 1.5)^d_z
    # U = Z w, w is a unit norm vector
    # Y ~ X + n, where n ~ N(0, U^2)
    # Category C.

    print('Category C')

    w = np.random.uniform(low = -1, high = 1, size = (z_dim, 1))
    w = w / np.linalg.norm(w)

    x = np.random.normal(loc=0, scale=1, size=(N, x_dim))
    #z = np.random.normal(loc=0, scale=1, size=(N, z_dim))
    z = np.random.uniform(low = 0.5, high = 1.5, size=(N, z_dim))
    u = np.dot(z, w)
    noise = np.zeros((N, y_dim))
    for i in range(N):
        temp = np.random.normal(loc=0.0, scale=np.abs(u[i]))
        noise[i, :] = temp
    y = x + noise

    save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed)

    save_ksg_est_fz(x, y, u.reshape(-1, 1), syn_file_cat, z_dim)


def gen_cmi_data_cond_indp1(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # n1, n2 ~ N(0, 1)
    # Z ~ N(0, 1)^d_z
    # X = w_x z + n1    || w_x || = 1
    # Y = w_y z + n2  , || w_y || = 1
    # Category D.

    print('Category D')


    w_x = np.random.uniform(low = -1, high = 1, size = (z_dim, 1))
    w_x = w_x / np.linalg.norm(w_x)

    w_y = np.random.uniform(low=-1, high=1, size=(z_dim, 1))
    w_y = w_y / np.linalg.norm(w_y)

    z = np.random.normal(loc=0, scale=1, size=(N, z_dim))

    n1 = np.random.normal(loc=0, scale=1, size=(N, x_dim))
    x = np.dot(z, w_x) + n1

    n2 = np.random.normal(loc=0, scale=1, size=(N, x_dim))
    y = np.dot(z, w_y) + n2

    save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed)

    cmi_est = 0.0
    print('CMI = {}'.format(cmi_est))
    np.save('./cat{}/ksg_gt.dz{}.npy'.format(syn_file_cat, z_dim), np.array([cmi_est]))


def gen_cmi_data_mean_gauss_channel(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # X ~ N(0, 1)
    # Z ~ U(-0.5, 0.5)
    # Y ~ X + n, where n ~ N(z, 0.1)
    # Category A.

    print('Category E')

    x = np.random.normal(loc = 0, scale = 1, size = (N, x_dim))
    z = np.random.uniform(low = -0.5, high = 0.5, size = (N, z_dim))
    noise = np.zeros((N, y_dim))
    for i in range(N):
         temp = np.random.normal(loc = z[i, :], scale = 0.1)
         noise[i,:] = temp
    y = x + noise

    save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed)

    save_ksg_est_fz(x, y, z, syn_file_cat, z_dim)



def gen_cmi_data_mean_sparse_dependency(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # X ~ N(0, 1)
    # Z ~ [U(-0.5, 0.5)]^d_z
    # Y ~ X + n, where n ~ N(z_1, 0.1)
    # Category F.

    print('Category F')
    num_samples = N + 50000

    x = np.random.normal(loc=0, scale=1, size=(num_samples, x_dim))
    z = np.random.uniform(low=-0.5, high=0.5, size=(num_samples, z_dim))

    noise = np.zeros((num_samples, y_dim))
    for i in range(num_samples):
        temp = np.random.normal(loc= z[i, 0], scale = 0.1)
        noise[i, :] = temp
    y = x + noise

    save_joint_data(x[0:N,:], y[0:N,:], z[0:N,:], syn_file_cat, num_th, z_dim, seed)

    save_ksg_est_fz(x[N:,:], y[N:,:], z[N:, 0].reshape(-1, 1), syn_file_cat, z_dim)


def gen_cmi_data_mean_lin_comb_Z(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # X ~ N(0, 1)
    # Z ~ N(0, 1)^d_z
    # U = Z w, w is a unit norm vector
    # Y ~ X + n, where n ~ N(U, 0.1)
    # Category G.

    print('Category G')
    num_samples = N + 50000

    w = np.random.uniform(low = -1, high = 1, size = (z_dim, 1))
    w = w / np.linalg.norm(w)

    x = np.random.normal(loc=0, scale=1, size=(num_samples, x_dim))
    z = np.random.normal(loc=0, scale=1, size=(num_samples, z_dim))
    u = np.dot(z, w)
    noise = np.zeros((num_samples, y_dim))
    for i in range(num_samples):
        temp = np.random.normal(loc=u[i], scale=0.1)
        noise[i, :] = temp
    y = x + noise

    save_joint_data(x[0:N,:], y[0:N,:], z[0:N,:], syn_file_cat, num_th, z_dim, seed)

    u = u.reshape(-1,1)
    save_ksg_est_fz(x[N:,:], y[N:,:], u[N:,:], syn_file_cat, z_dim)


def gen_cmi_data_cond_indp2(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # n1, n2 ~ N(0, 0.1)
    # Z ~ U(-1.0, 1.0)^d_z
    # X = sin(w_x z) + n1    || w_x || = 1
    # Y = cos(w_y z) + n2  , || w_y || = 1
    # Category H.

    print('Category H')


    w_x = np.random.uniform(low = -1, high = 1, size = (z_dim, 1))
    w_x = w_x / np.linalg.norm(w_x)

    w_y = np.random.uniform(low=-1, high=1, size=(z_dim, 1))
    w_y = w_y / np.linalg.norm(w_y)

    z = np.random.uniform(low=-1.0, high=1.0, size=(N, z_dim))

    n1 = np.random.normal(loc=0, scale=1.0, size=(N, x_dim))
    x = np.sin(np.dot(z, w_x)) + n1

    n2 = np.random.normal(loc=0, scale=1.0, size=(N, x_dim))
    y = np.cos(np.dot(z, w_y)) + n2

    save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed)

    cmi_est = 0.0
    print('CMI = {}'.format(cmi_est))
    np.save('./cat{}/ksg_gt.dz{}.npy'.format(syn_file_cat, z_dim), np.array([cmi_est]))



def gen_cmi_data_mixed_bsc_awgn(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed):

    # beta = 0.2, alpha = 0.3,
    # Z ~ U[0, 1]
    # (1) If Z < beta, then
    #        X ~ N(0,1), n ~ N(0, 0,01), Y = X + n
    # (2) Z' = min(alpha, Z) >= beta
    #        X ~ Bern(0.5), E ~ Bern(Z')
    #        Y = X (XOR) E
    # Category K.
    # Z = (Z, Z^2, Z^3)

    print('Category K')

    beta = 0.2
    alpha = 0.3
    p = 0.5

    x = np.zeros((N, x_dim))
    z = np.random.uniform(low = 0.0, high = 1.0, size=(N, z_dim))
    y = np.zeros((N, y_dim))

    for i in range(N):
        if z[i][0] < beta:
            x_i = np.random.normal(loc=0, scale=1, size=(1, x_dim))
            x[i,:] = x_i
            noise =  np.random.normal(loc=0, scale=0.1, size=(1, x_dim))
            y[i,:] = x_i + noise
        else:
            z_tilde = min(alpha, z[i][0])
            x_i = np.random.binomial(1, p, (1, x_dim))
            x[i,:] = x_i
            noise = np.random.binomial(1, z_tilde, (1, x_dim))
            y[i,:] = float(np.logical_xor(x_i, noise))

    for j in range(1, z_dim):
        z[:,j] = z[:,0]**(j+1)

    save_joint_data(x, y, z, syn_file_cat, num_th, z_dim, seed)

    save_ksg_est_fz(x, y, z, syn_file_cat, z_dim)

def gen_cmi_data_nonlin_NI(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed, f1, f2):

    # Z ~ N(I, var_z * I_d_z)
    # X ~ f1(n1)
    # Y ~ f2(Ay*Z + 2*X + n2)
    # Category Non-Lin-NI.

    print('Category {}'.format(syn_file_cat))

    #Data for KSG-estimate is also computed.
    num_samples = N + 50000
     
    # Bounded non-linear functions
    function_list = [tanh, negexp, cosine]
    if f1 == None:
        I1 = np.random.randint(low=0, high=3)
        f1 = function_list[I1]

    if f2 == None :
        I2 = np.random.randint(low=0, high=3)
        f2 = function_list[I2]

    print(f1)
    print(f2)
    # Normalize cross-correlation coefficients to have unit L1-norm
    A_zy = np.random.randn(z_dim, y_dim)
    for j in range(y_dim):
        A_zy[:, j] = A_zy[:, j] / np.linalg.norm(A_zy[:, j], ord=1)

    var_n = 0.1
    var_z = 1.0
    var_x = 0.1
    axy = 2.0
    mu_z = np.ones(z_dim)
    cov_z = var_z * np.eye(z_dim)
    Z = np.random.multivariate_normal(mu_z, cov_z, num_samples)
    X = f1(np.random.multivariate_normal(np.zeros(x_dim), var_x * np.eye(x_dim), num_samples))
    Y = f2(axy*X + np.dot(Z, A_zy) +
           np.random.multivariate_normal(np.zeros(y_dim), var_n*np.eye(y_dim), num_samples))


    save_joint_data(X[0:N,:], Y[0:N,:], Z[0:N,:], syn_file_cat, num_th, z_dim, seed)

    U = np.dot(Z[N:,:], A_zy)
    save_ksg_est_fz(X[N:, :], Y[N:, :], U, syn_file_cat, z_dim)

def gen_cmi_data_nonlin_CI(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed, f1, f2):

    # Z ~ N(0, var_z * I_d_z)
    # X ~ f1(n1)
    # Y ~ f2(Ay*Z + n2)
    # Category Non-Lin-CI.

    print('Category {}'.format(syn_file_cat))

    function_list = [tanh, negexp, cosine]
    if f1 == None :
        I1 = np.random.randint(low=0, high=3)
        f1 = function_list[I1]

    if f2 == None :
        I2 = np.random.randint(low=0, high=3)
        f2 = function_list[I2]


    # Normalize cross-correlation coefficients to have unit L1-norm
    A_zy = np.random.randn(z_dim, y_dim)
    for j in range(y_dim):
        A_zy[:, j] = A_zy[:, j] / np.linalg.norm(A_zy[:, j], ord=1)

    var_n = 0.1
    var_z = 1.0
    var_x = 0.1
    mu_z = np.ones(z_dim)
    cov_z = var_z * np.eye(z_dim)
    Z = np.random.multivariate_normal(mu_z, cov_z, N)
    X = f1(np.random.multivariate_normal(np.zeros(x_dim), var_x * np.eye(x_dim),N))
    Y = f2(np.dot(Z, A_zy) + np.random.multivariate_normal(np.zeros(y_dim), var_n * np.eye(y_dim), N))


    save_joint_data(X, Y, Z, syn_file_cat, num_th, z_dim, seed)

    cmi_est = 0.0
    print('CMI = {}'.format(cmi_est))
    np.save('./cat{}/ksg_gt.dz{}.npy'.format(syn_file_cat, z_dim), np.array([cmi_est]))


if __name__=='__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--cat', type=str, default='Non-lin-CI_25')
    parser.add_argument('--num_th', type=int, default=-1)
    parser.add_argument('--dz', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()



    f1 = None
    f2 = None
    x_dim = 1
    y_dim = 1
    z_dim = args.dz
    syn_file_cat = args.cat
    num_th = args.num_th
    N = num_th * 1000  # Number of samples in the dataset
    seed = args.seed  # Random seed for the dataset

    if syn_file_cat == 'A':
        gen_cmi_data_var_gauss_channel(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'B':
        gen_cmi_data_var_sparse_dependency(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'C':
        gen_cmi_data_var_lin_comb_Z(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'D':
        gen_cmi_data_cond_indp1(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'E':
        gen_cmi_data_mean_gauss_channel(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'F':
        gen_cmi_data_mean_sparse_dependency(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'G':
        gen_cmi_data_mean_lin_comb_Z(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'H':
        gen_cmi_data_cond_indp2(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat == 'K':
        gen_cmi_data_mixed_bsc_awgn(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed)
    elif syn_file_cat[0:10] == 'Non-lin-NI':
        gen_cmi_data_nonlin_NI(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed, f1, f2)
    elif syn_file_cat[0:10] == 'Non-lin-CI':
        gen_cmi_data_nonlin_CI(N, x_dim, y_dim, z_dim, syn_file_cat, num_th, seed, f1, f2)


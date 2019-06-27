import numpy as np
import NPEET.entropy_estimators as ee
import argparse
import importlib
from math import log
import Util

import dateutil.tz
import datetime

def split_XYZ(data, dx, dy):
    X = data[:, 0:dx]
    Y = data[:, dx:dx+dy]
    Z = data[:, dx+dy:]
    return X, Y, Z

def normalize_data(data):
    data_norm = (data - np.mean(data, axis = 0))/(np.std(data, axis = 0))
    return data_norm

def load_dataset(syn_file_cat, num_th, z_dim, seed):
    finp = '../data/cat{}/data.{}k.dz{}.seed{}.npy'.format(syn_file_cat, num_th, z_dim, seed)
    data = np.load(finp)
    return data

if __name__=='__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--cat', type=str, default='F')
    parser.add_argument('--num_th', type=int, default=10)
    parser.add_argument('--dz', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    x_dim = 1
    y_dim = 1
    z_dim = args.dz
    syn_file_cat = args.cat
    num_th = args.num_th
    N = num_th * 1000  # Number of samples in the dataset
    seed = args.seed  # Random seed for the dataset

    print('Category = {}, N = {}k,  d_z = {}'.format(syn_file_cat, num_th, z_dim))

    data = load_dataset(syn_file_cat, num_th, z_dim, seed)
    data = normalize_data(data)
    x, y, z = split_XYZ(data, x_dim, y_dim)
    k_list = [3, 5, 10]
    cmi_true = Util.get_true_mi(syn_file_cat, z_dim)

    for k in k_list:


        cmi_est_base2 = ee.cmi(x, y, z, k=k)
        cmi_est = cmi_est_base2 * log(2)
        print('k = {} True CMI = {},  Est_CMI = {}'.format(k, cmi_true, cmi_est))

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        fout = open('../logs/Res_baseline_ksg.txt', 'a')
        fout.write('{} : Category {}, N = {}k, d_z = {}, k = {}  True CMI = {}, Est. CMI = {} ,\n'
            .format(timestamp, syn_file_cat, num_th, z_dim, k, cmi_true, cmi_est))
        fout.close()


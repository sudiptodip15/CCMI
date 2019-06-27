import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
from CCIT import *

def load_data_postNonLin(file_index, dz, num_th, dep):
    filename = '../data/postNonLin_cos/syn{}.dz{}.{}k.{}.npy'.format(file_index, dz, num_th, dep)
    data = np.load(filename)
    return data

def split_XYZ(data, dx, dy, dz):
    X = data[:, 0:dx]
    Y = data[:, dx:dx+dy]
    Z = data[:, dx+dy:]
    return X, Y, Z


def normalize_data(data):

     data_norm = (data - np.mean(data, axis = 0))/(np.std(data, axis = 0))
     return data_norm

if __name__=='__main__':

    # dz_list = [1, 5, 20, 50, 70, 100]
    # CCIT higher dimensional data runs slow. Hence dz passed as command line argument and can be run in parallel. 
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dz', type=int)
    args = parser.parse_args()

    dz_inp = args.dz
    dz_list = [dz_inp]
    num_files = 100
    N = 5000
    dx = dy = 1
    num_th = int(N / 1000)

    for dz in dz_list:
        p_value_file = np.zeros(num_files)
        y_true = np.zeros(num_files)
        for i in range(1, num_files + 1):
            dep = i % 2
            data = load_data_postNonLin(i, dz, num_th, dep)
            data = normalize_data(data)
            X, Y, Z = split_XYZ(data, dx, dy, dz)
            #p_value = CCIT(X, Y, Z, num_iter = 50, bootstrap = True)
            p_value = CCIT(X, Y, Z)
            print('syn{}.dz{}.{}k.{} : p_value = {}'.format(i, dz, num_th, dep, p_value))
            p_value_file[i - 1] = p_value
            y_true[i-1] = dep

        auroc = roc_auc_score(1-y_true, p_value_file)

        with open('../logs/Res_postNonLin_AuROC.txt', 'a+') as f_auroc:
            f_auroc.write('N = {}k, # Datasets = {} : Tester = CCIT , dz = {},  AuROC = {}\n'
                          .format(num_th, num_files, dz, auroc))






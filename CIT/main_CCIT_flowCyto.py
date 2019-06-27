import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from CCIT import *

def load_data_flowCyto(file_index, dep):
    filename = '../data/flowCyto/data{}.{}.npy'.format(file_index, dep)
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

    num_files = 100
    dx = dy = 1

    p_value_file = np.zeros(num_files)
    y_true = np.zeros(num_files)
    for i in range(1, num_files + 1):
        dep = i % 2
        data = load_data_flowCyto(i, dep)
        data = normalize_data(data)
        dz = data.shape[1] - dx - dy
        X, Y, Z = split_XYZ(data, dx, dy, dz)
        p_value = CCIT(X, Y, Z, num_iter = 50, bootstrap = True)
        print('data{}.{} : p_value = {}'.format(i, dep, p_value))
        p_value_file[i - 1] = p_value
        y_true[i-1] = dep

    preds = 1-p_value_file
    auroc = roc_auc_score(y_true, preds)

    with open('../logs/Res_flowCyto_AuROC.txt', 'a+') as f_auroc:
        f_auroc.write('# Datasets = {} : Tester = CCIT, AuROC = {}\n'
                      .format(num_files, auroc))





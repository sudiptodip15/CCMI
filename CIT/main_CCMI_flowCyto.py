import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from CCMI import CCMI
import pdb


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

    tester = 'Classifier'
    metric = 'donsker_varadhan'
    num_boot_iter = 20
    h_dim = 64
    max_ep = 10


    cmi_score = np.zeros(num_files)
    y_true = np.zeros(num_files)
    for i in range(1, num_files + 1):
        dep = i % 2
        data = load_data_flowCyto(i, dep)
        data = normalize_data(data)
        dz = data.shape[1] - dx - dy
        X, Y, Z = split_XYZ(data, dx, dy, dz)

        ccmi = CCMI(X, Y, Z, tester, metric, num_boot_iter, h_dim, max_ep)
        cmi_est = ccmi.get_cmi_est()
        print('data{}.{} : CMI_est = {}'.format(i, dep, cmi_est))
        cmi_score[i - 1] = cmi_est
        y_true[i-1] = dep

    preds = cmi_score
    auroc = roc_auc_score(y_true, cmi_score)

    with open('../logs/Res_flowCyto_AuROC.txt', 'a+') as f_auroc:
        f_auroc.write('# Datasets = {} : Tester = {}, Metric = {}, AuROC = {}\n'
                      .format(num_files, tester, metric, auroc))

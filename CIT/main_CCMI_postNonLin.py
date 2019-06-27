import numpy as np
from sklearn.metrics import roc_auc_score
from CCMI import CCMI

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
    data_new = (data - np.mean(data, axis = 0))/(np.std(data, axis = 0))
    return data_new

if __name__=='__main__':


    dz_list = [1, 5, 20, 50, 70, 100]
    num_files = 100
    N = 5000
    dx = dy = 1
    num_th = int(N / 1000)

    tester = 'Classifier'
    metric = 'donsker_varadhan'
    num_boot_iter = 10
    h_dim = 64
    max_ep = 20

    for dz in dz_list:
        cmi_score = np.zeros(num_files)
        y_true = np.zeros(num_files)
        for i in range(1, num_files + 1):
            dep = i % 2
            data = load_data_postNonLin(i, dz, num_th, dep)
            data = normalize_data(data)
            X, Y, Z = split_XYZ(data, dx, dy, dz)

            ccmi = CCMI(X, Y, Z, tester, metric, num_boot_iter, h_dim, max_ep)
            cmi_est = ccmi.get_cmi_est()
            print('syn{}.dz{}.{}k.{} : CMI_est = {}'.format(i, dz, num_th, dep, cmi_est))
            cmi_score[i - 1] = cmi_est
            y_true[i-1] = dep

        auroc = roc_auc_score(y_true, cmi_score)

        with open('../logs/Res_postNonLin_AuROC.txt', 'a+') as f_auroc:
            f_auroc.write('N = {}k, # Datasets = {} : Tester = {}, Metric = {}, dz = {},  AuROC = {}\n'
                          .format(num_th, num_files, tester, metric, dz, auroc))






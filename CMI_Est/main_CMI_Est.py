import argparse
import dateutil.tz
import datetime
import Util
import numpy as np
from CMI_Est_modular import CMI_Est_modular

def load_dataset(syn_file_cat, num_th, z_dim, seed):
    finp = '../data/cat{}/data.{}k.dz{}.seed{}.npy'.format(syn_file_cat, num_th, z_dim, seed)
    data = np.load(finp)
    return data

if __name__=='__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--mimic', type=str, default='cgan')   #{'knn', 'cgan', 'cvae', 'mi_diff'}
    parser.add_argument('--bn', type=str, default='False')
    parser.add_argument('--tester', type=str, default='Classifier')
    parser.add_argument('--metric', type=str, default='donsker_varadhan')
    parser.add_argument('--cat', type=str, default='Non-lin-CI_25')
    parser.add_argument('--num_th', type=int, default=-1)
    parser.add_argument('--dz', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()


    bn = args.bn
    mimic = args.mimic
    tester = args.tester
    metric = args.metric

    # Synthetic data descriptions
    x_dim = 1
    y_dim = 1
    z_dim = args.dz
    syn_file_cat = args.cat
    num_th = args.num_th
    N = num_th * 1000  # Number of samples in the dataset
    seed = args.seed  # Random seed for the dataset

    outer_boot_iter = 10
    inner_dup_iter = 5

    data = load_dataset(syn_file_cat, num_th, z_dim, seed)

    cmi_true = Util.get_true_mi(syn_file_cat, z_dim)
    cmi_est_mu, cmi_est_std = CMI_Est_modular(data, x_dim, y_dim, z_dim, mimic,
                            tester, metric, cmi_true, outer_boot_iter, inner_dup_iter)

    print('True CMI = {},  Est_CMI = {}'.format(cmi_true, cmi_est_mu))

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    fout = open('../logs/Res_cmi_est.{}.{}.txt'.format(mimic, tester), 'a')
    fout.write('{} : Category {}, N = {}k, d_z = {}, Mimic = {}, Tester = {}, Metric = {}, : True CMI = {}, Est. CMI = {} +- {}, \n'
               .format(timestamp, syn_file_cat, num_th, z_dim, mimic, tester, metric, cmi_true, cmi_est_mu, cmi_est_std))
    fout.close()


import numpy as np

def get_true_mi(syn_file_cat, z_dim):

    cmi_est = np.load('../data/cat{}/ksg_gt.dz{}.npy'.format(syn_file_cat, z_dim))
    return float(cmi_est)

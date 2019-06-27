import numpy as np
from math import log
import os

def write_samples_to_file(data, file_index, dz, num_th, dep):
    save_dir = './postNonLin_cos'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = './postNonLin_cos/syn{}.dz{}.{}k.{}.npy'.format(file_index, dz, num_th, dep)
    np.save(filename, data)

def postNonLin(dz = 1, N = 10000, dep = 0):
     '''
     d_x = 1, d_y = 1, d_z can scale
     If I(X;Y|Z) = 0,  Z ~ N(1.0, 1.0), X = cos(a*Z + n1), Y = cos(b*Z + n2), ||a|| = 1, ||b|| = 1. a, b ~ U[0,1]
     Else Z ~ N(1.0, 1.0), X = cos(a*Z + n1), Y = cos(2*c*X + b*Z + n2), ||a|| = 1, ||b|| = 1. a, b, c ~ U[0,1]
     n1, n2 ~ N(0, 0.25)
     '''

     dx = 1
     dy = 1
     Z = np.random.normal(loc=1.0, scale=1.0, size=(N, dz))
     a_x = np.random.rand(dz, dx)
     a_x /= np.linalg.norm(a_x)

     b_y = np.random.rand(dz, dy)
     b_y /= np.linalg.norm(b_y)

     a_xy = np.random.rand()
     nstd = 0.50

     if dep == 0:
         X = np.cos(np.dot(Z, a_x) + np.random.normal(loc = 0.0, scale = nstd, size = (N, dx)))
         Y = np.cos(np.dot(Z, b_y) + np.random.normal(loc = 0.0, scale = nstd, size = (N, dy)))
     else:
         X = np.cos(np.dot(Z, a_x) + np.random.normal(loc=0.0, scale=nstd, size=(N, dx)))
         Y = np.cos(2*a_xy*X + np.dot(Z, b_y) + np.random.normal(loc=0.0, scale=nstd, size=(N, dy)))

     return np.hstack((X, Y, Z))


if __name__=='__main__':

    dz_list = [1, 5, 20, 50, 70, 100]
    num_files = 100
    N = 5000
    num_th = int(N/1000)
    for dz in dz_list:
        for i in range(1, num_files+1):
            dep = i % 2
            data = postNonLin(dz, N, dep)
            write_samples_to_file(data, i, dz, num_th, dep)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
eps = 1e-8

def plot_util(mi_with_iter, mi_true, mon_freq = 5000):
        plt.plot((np.arange(len(mi_with_iter)) + 1)*mon_freq, np.array(mi_with_iter))
        plt.axhline(y=mi_true, color='r')
        plt.xlabel('Iterations')
        plt.ylabel('I(X; Y)')
        plt.show()

def shuffle(batch_data, dx):

    batch_x = batch_data[:, 0:dx]
    batch_y = batch_data[:, dx:]
    batch_y = np.random.permutation(batch_y)
    return np.hstack((batch_x, batch_y))


def log_mean_exp_tf(fx_q, ax=0):

    max_ele = tf.reduce_max(fx_q, axis = ax, keepdims=True)
    return tf.squeeze(max_ele + tf.log(eps + tf.reduce_mean(tf.exp(fx_q-max_ele))))


def log_mean_exp_numpy(fx_q, ax = 0):

    max_ele = np.max(fx_q, axis=ax, keepdims = True)
    return (max_ele + np.log(eps + np.mean(np.exp(fx_q-max_ele), axis = ax, keepdims=True))).squeeze()

def smooth_ma(values, window_size=100):
    return [np.mean(values[i:i + window_size]) for i in range(0, len(values) - window_size)]


if __name__=='__main__':

    # Testing stabilization of log_mean_exp : Numpy
    X = np.array([[10, 45, -500], [4, 2, 4], [0.2, 0.3, -0.5]])
    val1 = np.log(eps + np.mean(np.exp(X), axis = 1, keepdims=True))
    val2 = log_mean_exp_numpy(X)
    print(val1)
    print(val2)

    # Testing stabilization of log_mean_exp : tensorflow

    X_node = tf.placeholder(dtype = tf.float32, shape = [None, X.shape[1]], name = 'fxq')
    val3_node = log_mean_exp_tf(X_node)

    with tf.Session() as sess:

        val3 = sess.run(val3_node, feed_dict={X_node : X})
        print(val3)

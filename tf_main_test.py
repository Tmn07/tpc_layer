# coding=utf-8

import tensorflow as tf
import numpy as np

import skimage.io
from scipy.misc import toimage
from scipy.misc import imsave

from functools import reduce


kron = tf.contrib.kfac.utils.kronecker_product

# def binary(data):
#     I = tf.zeros(data.shape)
#     h, w = data.shape
#     for i in range(h):
#         for j in range(w):
#             if data[i][j] > 0.4:
#                 I[i][j] = 1
#     return I

# def twopoint_correlation(data):
#     s = 0
#     h = data.shape[0]
#     w = data.shape[1]
#     for i1 in range(h):
#         for j1 in range(w):
#             for i2 in range(h):
#                 for j2 in range(w):
#                     s += tf.multiply(data[i1][j1], data[i2][j2])
#     # s = tf.expand_dims(s, 0)
#     return s

def twopoint_correlation(data):
    shape = data.get_shape().as_list()
    # print (data.shape)
    size = shape[0] * shape[1] * shape[0] * shape[1]
    res = kron(data[:,:,0], data[:,:,0])
    s = tf.reduce_sum(res)
    # print(s)
    # s.reshape([1])
    return tf.reshape(s,[1])

def minmax_scaling(data):
    ma = tf.reduce_max(data)
    mi = tf.reduce_min(data)
    return (data - mi) / (ma-mi)


def twopoint_correlation_layer(data, kernel_size=(3, 3)):

    data = minmax_scaling(data)

    # print(data.shape)
    h = data.shape[0]
    w = data.shape[1]
    # s_m = []
    for i in range(h - kernel_size[0] + 1):
        s_row = []
        for j in range(w - kernel_size[1] + 1):
            s = twopoint_correlation(data[i:i + kernel_size[0], j:j + kernel_size[1]])
            print(s)
            s_row = tf.concat([s_row, s], 0)

        if i == 0:
            s_m = tf.expand_dims(s_row, 0)
        else:
            s_row = tf.expand_dims(s_row, 0)
            s_m = tf.concat([s_m, s_row], 0)
            # print(s_row.eval())
            # print(s_m.eval())
    return s_m


def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    # size = reduce(lambda x, y: x * y, shape) ** 2
    size = shape[0]*shape[1]
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


if __name__ == '__main__':
    # from keras.datasets import mnist
    #
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.astype('float32')
    # img_rows, img_cols = 28, 28
    # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    # x_train /= 255

    EPOCHS = 300

    height = 30
    width = 30
    image_shape = (height, width, 1)

    noise_init = tf.truncated_normal(image_shape, mean=.5, stddev=.1)

    # img = skimage.io.imread("res/result-0.png", as_grey=True) / 255.0
    # noise_init = tf.convert_to_tensor(img, dtype=tf.float32)
    # noise_init = tf.expand_dims(noise_init, 2)

    noise = tf.Variable(noise_init, dtype=tf.float32)

    img = skimage.io.imread("tex1.png", as_grey=True) / 255.0
    # img = np.array([[0.4,0.6,0.1,0.2],[0.1,0.5,0.5,0],[1,1,0.4,0.3],[0.2,0.2,0.5,0.8]])
    timg = tf.convert_to_tensor(img, dtype=tf.float32)
    timg = tf.expand_dims(timg, 2)
    # print(timg.shape)


    gen_sm = twopoint_correlation_layer(noise)
    sm = twopoint_correlation_layer(timg[:height, :width])

    total_loss = get_l2_norm_loss(gen_sm - sm)

    optimizer = tf.train.AdamOptimizer(0.02).minimize(total_loss, var_list=noise)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        
        # print(sess.run(noise))

        for i in range(1, EPOCHS):
            _, loss, xxx = sess.run([optimizer, total_loss, noise])
            # print(xxx.reshape(height, width))
            print("Epoch %d | Loss: %.03f\n" % (i, loss))
            # np.reshape()
            imsave('res/result-%d.png' % i, xxx.reshape(height, width))

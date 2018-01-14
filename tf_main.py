# coding=utf-8

import tensorflow as tf
import numpy as np

import skimage.io
import skimage.transform

from scipy.misc import toimage
from scipy.misc import imsave

from functools import reduce

import custom_vgg19 as vgg19



kron = tf.contrib.kfac.utils.kronecker_product



def load_image(path):
    # Load image [height, width, depth]
    img = skimage.io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    shape = list(img.shape)

    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (shape[0], shape[1]))
    
    shape = [1] + shape
    resized_img = resized_img.reshape(shape).astype(np.float32)

    return resized_img, shape

def render_img(session, x, out_path):
    shape = x.get_shape().as_list()
    img = np.clip(session.run(x), 0, 1)

    toimage(np.reshape(img, shape[1:])).save(out_path)



def convert_to_gram(filter_maps):
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])
    Ml = dimension[1] * dimension[2]
    return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True) / Ml

def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    # size = reduce(lambda x, y: x * y, shape) ** 2
    size = shape[0]*shape[1]
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size

def get_texture_loss(x, s):
    with tf.name_scope('get_style_loss'):
        texture_layer_losses = [get_texture_loss_for_layer(x, s, l) for l in TEXTURE_LAYERS]
        texture_weights = tf.constant([1. / len(texture_layer_losses)] * len(texture_layer_losses), tf.float32)
        weighted_layer_losses = tf.multiply(texture_weights, tf.convert_to_tensor(texture_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)

def get_texture_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):
        x_layer_maps = getattr(x, l)
        t_layer_maps = getattr(s, l)
        x_layer_gram = convert_to_gram(x_layer_maps)
        t_layer_gram = convert_to_gram(t_layer_maps)

        # shape = x_layer_maps.get_shape().as_list()
        # size = reduce(lambda a, b: a * b, shape) ** 2
        gram_loss = get_l2_norm_loss(x_layer_gram - t_layer_gram)
        return gram_loss



def twopoint_correlation(data):
    shape = data.get_shape().as_list()
    # print (data.shape)
    size = shape[0] * shape[1] * shape[0] * shape[1]
    res = kron(data, data)
    # res = kron(data[:,:,0], data[:,:,0])
    s = tf.reduce_sum(res)
    return s
    # print(s)
    # s.reshape([1])
    # return tf.reshape(s,[1])


def minmax_scaling(data):
    ma = tf.reduce_max(data)
    mi = tf.reduce_min(data)
    if tf.equal(ma,mi) is not None:
        return data
    return (data - mi) / (ma-mi)


def twopoint_correlation_layer(data, kernel_size=(3, 3)):

    data = minmax_scaling(data)

    h = data.shape[0]
    w = data.shape[1]
    th = h - kernel_size[0] + 1
    tw = w - kernel_size[0] + 1
    s_m = []
    for i in range(h - kernel_size[0] + 1):
        for j in range(w - kernel_size[1] + 1):
            s = twopoint_correlation(data[i:i + kernel_size[0], j:j + kernel_size[1]])
            s_m.append(s)

    s_m = tf.reshape(tf.stack(s_m), [th, tw])
    return s_m


def l2_loss(diffs):
    shape = diffs.get_shape().as_list()
    # size = reduce(lambda x, y: x * y, shape) ** 2
    size = shape[0]*shape[1]
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


if __name__ == '__main__':

    EPOCHS = 30
    TEXTURE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
    height = 12
    width = 12
    image_shape = (height, width)
    Alpha = 1
    Beta = 1

    # , 1)


    
    # noise_init = tf.truncated_normal(image_shape, mean=.5, stddev=.1)

    img = skimage.io.imread("res/0result.png", as_grey=True) / 255.0
    noise_init = tf.convert_to_tensor(img, dtype=tf.float32)


    noise = tf.Variable(noise_init, dtype=tf.float32)

    img = skimage.io.imread("tex1-b.png", as_grey=True) / 255.0
    # img = np.array([[0.4,0.6,0.1,0.2],[0.1,0.5,0.5,0],[1,1,0.4,0.3],[0.2,0.2,0.5,0.8]])
    timg = tf.convert_to_tensor(img, dtype=tf.float32)
    # timg = tf.expand_dims(timg, 2)
    # print(timg.shape)

    gen_sm = twopoint_correlation_layer(noise)
    sm = twopoint_correlation_layer(timg[:height, :width])

    tpc_loss = l2_loss(gen_sm - sm)

    # compute style loss

    texture_img, texture_shape = load_image("tex3.png")
    # h,w,c
    texture_model = vgg19.Vgg19()
    texture_model.build(texture_img, texture_shape[1:])
    
    
    
    rgbnoise = tf.expand_dims(noise, 2)
    rgbnoise = tf.image.grayscale_to_rgb(rgbnoise)
    rgbnoise = tf.expand_dims(rgbnoise, 0)
    # print(rgbnoise.shape)
    x_model = vgg19.Vgg19()
    x_model.build(rgbnoise, rgbnoise.shape.as_list()[1:])

    style_loss = get_texture_loss(x_model, texture_model)


    total_loss = Alpha * tpc_loss + Beta * style_loss

    optimizer = tf.train.AdamOptimizer(0.02).minimize(total_loss, var_list=noise)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        
        # print(sess.run(noise))

        for i in range(1, EPOCHS):
            _, ls1, ls2, xxx = sess.run([optimizer, tpc_loss, style_loss, noise])
            # print(xxx.reshape(height, width))
            print("Epoch %d | tpc_loss: %.03f | style_loss: %.03f\n" % (i, ls1, ls2))
            if i % 10 == 0:
                imsave('res/result-%d.png' % i, xxx.reshape(height, width))

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



# todo: 重复计算优化
def twopoint_correlation(patch):
    tmp = np.kron(patch, patch)
    # print(tmp.size)
    return tmp.sum()
     # / tmp.size

# todo: strides!=1 s size?
# todo: multi channel
def twopoint_correlation_layer(image, size=3, strides=1):
    h, w = image.shape
    s_matrix = np.zeros((h-size+1, w-size+1))
    patchsum_matrix = np.zeros((h-size+1, w-size+1))
    for i1 in range(h-size+1):
        for j1 in range(w-size+1):
            # print (image[i1:i1+size, j1:j1+size])
            patch = image[i1:i1+size, j1:j1+size]
            s_matrix[i1, j1] = twopoint_correlation(patch)
            patchsum_matrix[i1, j1] = np.sum(patch) # /9
    # print (s_matrix)
    return s_matrix, patchsum_matrix


def sumgrad(i, j, grads_init):
    h, w = grads_init.shape
    sg = 0
    for x in range(i-2,i+1):
        for y in range(j-2,j+1):
            if x>-1 and x<h and y>-1 and y<w:
                # print((i,j),(x,y))
                sg += grads_init[x, y]
    return sg

# def compute_tpc_loss():
def compute_tpc_grad(grads_init, vars_):
    h, w = vars_.shape
    grads = np.zeros((h, w))

    for i1 in range(h):
        for j1 in range(w):
            grads[i1, j1] = sumgrad(i1, j1, grads_init)

    return grads

def npl2(diffs):
    # /
    return np.sum(np.power((diffs),2)) / diffs.size
    
if __name__ == '__main__':

    EPOCHS = 300
    
    TEXTURE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
    
    height = 327
    width = 327

    image_shape = (height, width)

    Alpha = 1
    Beta = 1

    # , 1)


    
    noise_init = tf.truncated_normal(image_shape, mean=.5, stddev=.1)

    # img = skimage.io.imread("res/result.png", as_grey=True) / 255.0
    # noise_init = tf.convert_to_tensor(img, dtype=tf.float32)
    # noise_init = tf.expand_dims(noise_init, 2)

    noise = tf.Variable(noise_init, dtype=tf.float32)
    img = skimage.io.imread("tex1-b.png", as_grey=True) / 255.0


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


    # total_loss = Alpha * tpc_loss + Beta * style_loss
    # beta1 = tf.Variable(0.9)
    # beta2 = tf.Variable(0.999) 
    
    # optimizer = tf.train.MomentumOptimizer(0.02,momentum=0.1)
    # optimizer = tf.train.GradientDescentOptimizer(0.02)

    # .minimize(style_loss, var_list=noise)
    
    optimizer = tf.train.AdamOptimizer(0.02,beta1=0.9,beta2=0.999) # ???? 
    grads_and_vars = optimizer.compute_gradients(style_loss, noise)


    grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g,v) in grads_and_vars]


    opt_step = optimizer.apply_gradients(grads_holder)    

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        for i in range(1, EPOCHS):
            # print(grads_and_vars)
            grads_and_vars_ = grads_and_vars[0]
            grads = grads_and_vars_[0]
            vars_ = grads_and_vars_[1]
            

            gen_img = vars_.eval()
            
            gen_sm, ps_m = twopoint_correlation_layer(gen_img)
            ref_sm, _ = twopoint_correlation_layer(img[:height, :width])
            loss_m = gen_sm-ref_sm
            tpc_loss = npl2(loss_m)
            tpcgrad_init = (gen_sm-ref_sm)*ps_m * 4 / ps_m.size
            tpc_grads = compute_tpc_grad(tpcgrad_init, gen_img)

            tpc_grads = tf.convert_to_tensor(tpc_grads, tf.float32)
            new_grads_and_vars = [(tf.add(gv[0], tpc_grads), gv[1]) for gv in grads_and_vars]


            grads_dict={} 
            for j in range(len(new_grads_and_vars)): 
                k = grads_holder[j][0]
                grads_dict[k] = new_grads_and_vars[j][0].eval()

            # opt_step = optimizer.apply_gradients(new_grads_and_vars)

            # debug 
            # print(gen_img)
            # print(gen_sm)
            # print("------------")
            
            # print(img[:height, :width])
            # print("------------")
            # print(ref_sm)
            # print(tpc_loss)
            # print("------------")
            # print(tpc_grads)
            # print(tpcgrad_init)
            # print(new_grads_and_vars)

            _, ls2, xxx = sess.run([opt_step, style_loss, noise],feed_dict=grads_dict)
            ls1 = tpc_loss
            print("Epoch %d | tpc_loss: %.03f | style_loss: %.03f" % (i, ls1, ls2))

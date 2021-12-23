# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np


def pynet_g(input, instance_norm=True, instance_norm_level_1=False, upscale="transpose", downscale="maxpool", self_att=False):
    with tf.compat.v1.variable_scope("pynet_g"):
        
        # -----------------------------------------
        # Downsampling layers
        conv_l1_d1 = _conv_multi_block(input, 3, num_maps=32, instance_norm=False)              # 128 -> 128
        pool1 = _downscale(conv_l1_d1, 64, 3, 2, downscale)                                     # 128 -> 64
        if self_att:
            pool1 = _stack(pool1, _self_attention(pool1, 64, sn=True))

        conv_l2_d1 = _conv_multi_block(pool1, 3, num_maps=64, instance_norm=instance_norm)      # 64 -> 64
        pool2 = _downscale(conv_l2_d1, 128, 3, 2, downscale)                                    # 64 -> 32

        conv_l3_d1 = _conv_multi_block(pool2, 3, num_maps=128, instance_norm=instance_norm)     # 32 -> 32
        pool3 = _downscale(conv_l3_d1, 256, 3, 2, downscale)                                      # 32 -> 16

        conv_l4_d1 = _conv_multi_block(pool3, 3, num_maps=256, instance_norm=instance_norm)     # 16 -> 16
        pool4 = _downscale(conv_l4_d1, 256, 3, 2, downscale)                                    # 16 -> 8

        # -----------------------------------------
        # Processing: Level 5,  Input size: 8 x 8
        conv_l5_d1 = _conv_multi_block(pool4, 3, num_maps=512, instance_norm=instance_norm)
        conv_l5_d2 = _conv_multi_block(conv_l5_d1, 3, num_maps=512, instance_norm=instance_norm) + conv_l5_d1
        conv_l5_d3 = _conv_multi_block(conv_l5_d2, 3, num_maps=512, instance_norm=instance_norm) + conv_l5_d2
        conv_l5_d4 = _conv_multi_block(conv_l5_d3, 3, num_maps=512, instance_norm=instance_norm)

        conv_t4a = _upscale(conv_l5_d4, 256, 3, 2, upscale)      # 8 -> 16
        conv_t4b = _upscale(conv_l5_d4, 256, 3, 2, upscale)      # 8 -> 16

        # -> Output: Level 5
        conv_l5_out = _conv_layer(conv_l5_d4, 3, 3, 1, relu=False, instance_norm=False)
        output_l5 = tf.nn.tanh(conv_l5_out) * 0.58 + 0.5

        # -----------------------------------------
        # Processing: Level 4,  Input size: 28 x 28
        conv_l4_d2 = _stack(conv_l4_d1, conv_t4a)
        conv_l4_d3 = _conv_multi_block(conv_l4_d2, 3, num_maps=256, instance_norm=instance_norm)
        conv_l4_d4 = _conv_multi_block(conv_l4_d3, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d3
        conv_l4_d5 = _conv_multi_block(conv_l4_d4, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d4
        conv_l4_d6 = _stack(_conv_multi_block(conv_l4_d5, 3, num_maps=256, instance_norm=instance_norm), conv_t4b)

        conv_l4_d7 = _conv_multi_block(conv_l4_d6, 3, num_maps=256, instance_norm=instance_norm)

        conv_t3a = _upscale(conv_l4_d7, 128, 3, 2, upscale)      # 16 -> 32
        conv_t3b = _upscale(conv_l4_d7, 128, 3, 2, upscale)      # 16 -> 32

        # -> Output: Level 4
        conv_l4_out = _conv_layer(conv_l4_d7, 3, 3, 1, relu=False, instance_norm=False)
        output_l4 = tf.nn.tanh(conv_l4_out) * 0.58 + 0.5

        # -----------------------------------------
        # Processing: Level 3,  Input size: 56 x 56
        conv_l3_d2 = _stack(conv_l3_d1, conv_t3a)
        conv_l3_d3 = _conv_multi_block(conv_l3_d2, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d2
        conv_l3_d4 = _conv_multi_block(conv_l3_d3, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d3
        conv_l3_d5 = _conv_multi_block(conv_l3_d4, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d4
        conv_l3_d6 = _stack(_conv_multi_block(conv_l3_d5, 5, num_maps=128, instance_norm=instance_norm), conv_l3_d1)
        conv_l3_d7 = _stack(conv_l3_d6, conv_t3b)

        conv_l3_d8 = _conv_multi_block(conv_l3_d7, 3, num_maps=128, instance_norm=instance_norm)

        conv_t2a = _upscale(conv_l3_d8, 64, 3, 2, upscale)       # 32 -> 64
        conv_t2b = _upscale(conv_l3_d8, 64, 3, 2, upscale)       # 32 -> 64

        # -> Output: Level 3
        conv_l3_out = _conv_layer(conv_l3_d8, 3, 3, 1, relu=False, instance_norm=False)
        output_l3 = tf.nn.tanh(conv_l3_out) * 0.58 + 0.5

        # -------------------------------------------
        # Processing: Level 2,  Input size: 112 x 112
        conv_l2_d2 = _stack(conv_l2_d1, conv_t2a)
        conv_l2_d3 = _stack(_conv_multi_block(conv_l2_d2, 5, num_maps=64, instance_norm=instance_norm), conv_l2_d1)

        conv_l2_d4 = _conv_multi_block(conv_l2_d3, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d3
        conv_l2_d5 = _conv_multi_block(conv_l2_d4, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d4
        conv_l2_d6 = _conv_multi_block(conv_l2_d5, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d5
        conv_l2_d7 = _stack(_conv_multi_block(conv_l2_d6, 7, num_maps=64, instance_norm=instance_norm), conv_l2_d1)

        conv_l2_d8 = _stack(_conv_multi_block(conv_l2_d7, 5, num_maps=64, instance_norm=instance_norm), conv_t2b)
        conv_l2_d9 = _conv_multi_block(conv_l2_d8, 3, num_maps=64, instance_norm=instance_norm)

        conv_t1a = _upscale(conv_l2_d9, 32, 3, 2, upscale)       # 64 -> 128
        conv_t1b = _upscale(conv_l2_d9, 32, 3, 2, upscale)       # 64 -> 128

        # -> Output: Level 2
        conv_l2_out = _conv_layer(conv_l2_d9, 3, 3, 1, relu=False, instance_norm=False)
        output_l2 = tf.nn.tanh(conv_l2_out) * 0.58 + 0.5

        # -------------------------------------------
        # Processing: Level 1,  Input size: 224 x 224
        conv_l1_d2 = _stack(conv_l1_d1, conv_t1a)
        conv_l1_d3 = _stack(_conv_multi_block(conv_l1_d2, 5, num_maps=32, instance_norm=False), conv_l1_d1)

        conv_l1_d4 = _conv_multi_block(conv_l1_d3, 7, num_maps=32, instance_norm=False)

        conv_l1_d5 = _conv_multi_block(conv_l1_d4, 9, num_maps=32, instance_norm=instance_norm_level_1)
        conv_l1_d6 = _conv_multi_block(conv_l1_d5, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d5
        conv_l1_d7 = _conv_multi_block(conv_l1_d6, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d6
        conv_l1_d8 = _conv_multi_block(conv_l1_d7, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d7

        conv_l1_d9 = _stack(_conv_multi_block(conv_l1_d8, 7, num_maps=32, instance_norm=False), conv_l1_d1)

        conv_l1_d10 = _stack(_conv_multi_block(conv_l1_d9, 5, num_maps=32, instance_norm=False), conv_t1b)
        conv_l1_d11 = _stack(conv_l1_d10, conv_l1_d1)

        conv_l1_d12 = _conv_multi_block(conv_l1_d11, 3, num_maps=32, instance_norm=False)

        # -> Output: Level 1
        conv_l1_out = _conv_layer(conv_l1_d12, 3, 3, 1, relu=False, instance_norm=False)
        output_l1 = tf.nn.tanh(conv_l1_out) * 0.58 + 0.5

        # ----------------------------------------------------------
        # Processing: Level 0 (x2 upscaling),  Input size: 224 x 224

        conv_l0 = _upscale(conv_l1_d12, 8, 3, 2, upscale)        # 128 -> 256
        conv_l0_out = _conv_layer(conv_l0, 3, 3, 1, relu=False, instance_norm=False)

        output_l0 = tf.nn.tanh(conv_l0_out) * 0.58 + 0.5

    return output_l0, output_l1, output_l2, output_l3, output_l4, output_l5



def _upscale(net, num_filters, filter_size, factor, method):
    if method == "transpose":
        return _conv_tranpose_layer(net, num_filters, filter_size, factor)
    elif method == "shuffle":
        return _conv_pixel_shuffle_up(net, num_filters, filter_size, factor)
    elif method == "dcl":
        return _pixel_dcl(net, num_filters, filter_size)
    elif method == "resnet":
        return _resblock_up(net, num_filters, sn=True)
    elif method == "nn":
        return _nearest_neighbor(net, factor)
    else:
        print("Unrecognized upscaling method")

def _downscale(net, num_filters, filter_size, factor, method):
    if method == "maxpool":
        return _max_pool(net, factor)
    elif method == "shuffle":
        return tf.nn.space_to_depth(net, factor)
    elif method == "stride":
        return _conv_layer(net, num_filters, filter_size, factor)
    elif method == "resnet":
        return _resblock_down(net, num_filters, sn=True)
    else:
        print("Unrecognized downscaling method")

def texture_d(image_, activation=True):
    with tf.compat.v1.variable_scope("texture_d"):

        conv1 = _conv_layer(image_, 48, 11, 4, instance_norm = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        
        flat = tf.compat.v1.layers.flatten(conv5)

        fc1 = _fully_connected_layer(flat, 1024)
        adv_out = _fully_connected_layer(fc1, 1, relu=False)
        
        if activation:
            adv_out = tf.nn.sigmoid(adv_out)
    
    return adv_out

def unet_d(input, activation=True):
    with tf.compat.v1.variable_scope("unet_d"):
        ch = 64
        sn = True

        x = _resblock_down(input, ch, use_bias = False, sn=sn) #128
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn) #64

        x = _self_attention(x, ch, sn)
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn) #32
        ch = ch*2

        x = _resblock_down(x, ch, use_bias = False, sn=sn) #16
        x = _resblock_down(x, ch, use_bias = False, sn=sn) #8

        x = tf.compat.v1.nn.leaky_relu(x)
        x = _global_sum_pool(x)

        flat = tf.compat.v1.layers.flatten(x)
        out = _fully_connected_layer(flat, 2, relu=False)

        if activation:
            out = tf.nn.softmax(out)
    return out

def _resblock_down(input, num_filters, use_bias=True, sn=False):
    x = _instance_norm(input)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_layer(x, num_filters, 3, 2, relu=False, use_bias=use_bias, sn=sn)

    x = _instance_norm(x)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_layer(x, num_filters, 3, 1, relu=False, use_bias=use_bias, sn=sn)

    input = _conv_layer(input, num_filters, 3, 2, relu=False, use_bias=use_bias, sn=sn)

    return x + input

def _resblock_up(input, num_filters, use_bias=True, sn=False):
    x = _instance_norm(input)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_tranpose_layer(x, num_filters, 3, 2, relu = False, use_bias = use_bias, sn=sn)

    x = _instance_norm(x)
    x = tf.compat.v1.nn.leaky_relu(x)
    x = _conv_tranpose_layer(x, num_filters, 3, 1, relu = False, use_bias = use_bias, sn=sn)

    input = _conv_tranpose_layer(input, num_filters, 3, 2, relu = False, use_bias = use_bias, sn=sn)

    return x + input

def _self_attention(x, num_filters, sn=False):
    batch_size, height, width, num_channels = x.get_shape().as_list()
    f = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    g = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    h = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)

    s = tf.matmul(_hw_flatten(g), _hw_flatten(f), transpose_b=True)
    beta = tf.nn.softmax(s)

    o = tf.matmul(beta, _hw_flatten(h))
    o = _conv_layer(o, num_filters=num_channels, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    gamma = tf.Variable(tf.zeros([1]))

    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    y = gamma * o + x

    return y

def _self_attention_v2(x, num_filters, sn=False):
    f = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    f = _max_pool(f, 2)

    g = _conv_layer(x, num_filters=num_filters//8, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)

    h = _conv_layer(x, num_filters=num_filters, filter_size=1, strides=1, relu=False, use_bias=False, sn=sn)
    h = _max_pool(g, 2)

    s = tf.matmul(_hw_flatten(g), _hw_flatten(f), transpose_b=True)
    beta = tf.nn.softmax(s)

    o = tf.matmul(beta, _hw_flatten(h))
    gamma = tf.Variable(tf.zeros([1]))

    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    o = _conv_layer(o, num_filters, 1, 1, relu=False, sn=sn)
    x = gamma * o + x

    return x

def _hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def _conv_multi_block(input, max_size, num_maps, instance_norm):

    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)
    conv_3b = _conv_layer(conv_3a, num_maps, 3, 1, relu=True, instance_norm=instance_norm)

    output_tensor = conv_3b

    if max_size >= 5:

        conv_5a = _conv_layer(input, num_maps, 5, 1, relu=True, instance_norm=instance_norm)
        conv_5b = _conv_layer(conv_5a, num_maps, 5, 1, relu=True, instance_norm=instance_norm)

        output_tensor = _stack(output_tensor, conv_5b)

    if max_size >= 7:

        conv_7a = _conv_layer(input, num_maps, 7, 1, relu=True, instance_norm=instance_norm)
        conv_7b = _conv_layer(conv_7a, num_maps, 7, 1, relu=True, instance_norm=instance_norm)

        output_tensor = _stack(output_tensor, conv_7b)

    if max_size >= 9:

        conv_9a = _conv_layer(input, num_maps, 9, 1, relu=True, instance_norm=instance_norm)
        conv_9b = _conv_layer(conv_9a, num_maps, 9, 1, relu=True, instance_norm=instance_norm)

        output_tensor = _stack(output_tensor, conv_9b)

    return output_tensor

def _stack(x, y):
    return tf.concat([x, y], 3)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME', leaky = True, use_bias=True, sn=False):

    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]

    if sn:
        weights_init = _spectral_norm(weights_init)

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)

    if use_bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
        net = tf.nn.bias_add(net, bias)

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        if leaky:
            net = tf.compat.v1.nn.leaky_relu(net)
        else:
            net = tf.compat.v1.nn.relu(net)

    return net

def _fully_connected_layer(net, num_weights, relu=True):
    batch, channels = [i for i in net.get_shape()]
    weights_shape = [channels, num_weights]

    weights = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.01, shape=[num_weights]))

    out = tf.matmul(net, weights) + bias

    if relu:
        out = tf.compat.v1.nn.leaky_relu(out)

    return out

def _instance_norm(net):

    batch, rows, cols, channels = [i for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init


def _conv_tranpose_layer(net, num_filters, filter_size, strides, relu=True, leaky = True, use_bias=True, sn=False):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]

    if sn:
        weights_init = _spectral_norm(weights_init)

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    if use_bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
        net = tf.nn.bias_add(net, bias)
    if relu:
        if leaky:
            net = tf.compat.v1.nn.leaky_relu(net)
        else:
            net = tf.compat.v1.nn.relu(net)
    
    return net

def _nearest_neighbor(net, factor=2):
    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_size = [rows * factor, cols * factor]
    return tf.compat.v1.image.resize_nearest_neighbor(net, new_size)

def _conv_pixel_shuffle_up(net, num_filters, filter_size, factor):
    weights_init = _conv_init_vars(net, num_filters*factor*2, filter_size)

    strides_shape = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')

    net = tf.nn.depth_to_space(net, factor)

    return tf.compat.v1.nn.leaky_relu(net)


def _pixel_dcl(net, num_filters, filter_size, d_format='NHWC'):
    axis = (d_format.index('H'), d_format.index('W'))

    conv_0 = _conv_layer(net, num_filters, filter_size, 1, relu=False)
    conv_1 = _conv_layer(conv_0, num_filters, filter_size, 1, relu=False)

    dil_conv_0 = _dilate(conv_0, axis, (0,0))
    dil_conv_1 = _dilate(conv_1, axis, (1,1))

    conv_a = tf.add(dil_conv_0, dil_conv_1)

    weights = _conv_init_vars(conv_a, num_filters, filter_size)
    weights = tf.multiply(weights, _get_mask([filter_size, filter_size, num_filters, num_filters]))
    conv_b =  tf.nn.conv2d(conv_a, weights, strides=[1,1,1,1], padding='SAME')

    out = tf.add(conv_a, conv_b)

    return tf.compat.v1.nn.leaky_relu(out)

def _dilate(net, axes, shifts):
    for index, axis in enumerate(axes):
        elements = tf.unstack(net, axis=axis)
        zeros = tf.zeros_like(elements[0])
        for element_index in range(len(elements), 0, -1):
            elements.insert(element_index-shifts[index], zeros)
        net = tf.stack(elements, axis=axis)
    return net

def _get_mask(shape):
    new_shape = (np.prod(shape[:-2]), shape[-2], shape[-1])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i, :, :] = 0
    mask = np.reshape(mask, shape, 'F')
    return tf.constant(mask, dtype=tf.float32)

def _max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def _global_sum_pool(x):
    return tf.reduce_sum(x, axis=[1, 2])

def _nearest_neighbor(net, factor=2):
    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_size = [rows * factor, cols * factor]
    return tf.compat.v1.image.resize_nearest_neighbor(net, new_size)

def _spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.Variable(tf.random.normal([1, w_shape[-1]]), dtype=tf.float32)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
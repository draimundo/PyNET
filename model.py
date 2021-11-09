# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np


def PyNET(input, instance_norm=True, instance_norm_level_1=False):
    with tf.compat.v1.variable_scope("generator"):

        # -----------------------------------------
        # Downsampling layers

        conv_l1_d1 = _conv_multi_block(input, 3, num_maps=32, instance_norm=False)              # 128 -> 128
        pool1 = max_pool(conv_l1_d1, 2)                                                         # 128 -> 64

        conv_l2_d1 = _conv_multi_block(pool1, 3, num_maps=64, instance_norm=instance_norm)      # 64 -> 64
        pool2 = max_pool(conv_l2_d1, 2)                                                         # 64 -> 32

        conv_l3_d1 = _conv_multi_block(pool2, 3, num_maps=128, instance_norm=instance_norm)     # 32 -> 32
        pool3 = max_pool(conv_l3_d1, 2)                                                         # 32 -> 16

        conv_l4_d1 = _conv_multi_block(pool3, 3, num_maps=256, instance_norm=instance_norm)     # 16 -> 16
        pool4 = max_pool(conv_l4_d1, 2)                                                         # 16 -> 8

        # -----------------------------------------
        # Processing: Level 5,  Input size: 8 x 8

        conv_l5_d1 = _conv_multi_block(pool4, 3, num_maps=512, instance_norm=instance_norm)
        conv_l5_d2 = _conv_multi_block(conv_l5_d1, 3, num_maps=512, instance_norm=instance_norm) + conv_l5_d1
        conv_l5_d3 = _conv_multi_block(conv_l5_d2, 3, num_maps=512, instance_norm=instance_norm) + conv_l5_d2
        conv_l5_d4 = _conv_multi_block(conv_l5_d3, 3, num_maps=512, instance_norm=instance_norm)

        conv_t4a = _conv_tranpose_layer(conv_l5_d4, 256, 3, 2)      # 8 -> 16
        conv_t4b = _conv_tranpose_layer(conv_l5_d4, 256, 3, 2)      # 8 -> 16

        # -> Output: Level 5

        conv_l5_out = _conv_layer(conv_l5_d4, 3, 3, 1, relu=False, instance_norm=False)
        output_l5 = tf.nn.tanh(conv_l5_out) * 0.58 + 0.5

        # -----------------------------------------
        # Processing: Level 4,  Input size: 28 x 28

        conv_l4_d2 = stack(conv_l4_d1, conv_t4a)
        conv_l4_d3 = _conv_multi_block(conv_l4_d2, 3, num_maps=256, instance_norm=instance_norm)
        conv_l4_d4 = _conv_multi_block(conv_l4_d3, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d3
        conv_l4_d5 = _conv_multi_block(conv_l4_d4, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d4
        conv_l4_d6 = stack(_conv_multi_block(conv_l4_d5, 3, num_maps=256, instance_norm=instance_norm), conv_t4b)

        conv_l4_d7 = _conv_multi_block(conv_l4_d6, 3, num_maps=256, instance_norm=instance_norm)

        conv_t3a = _conv_tranpose_layer(conv_l4_d7, 128, 3, 2)      # 28 -> 56
        conv_t3b = _conv_tranpose_layer(conv_l4_d7, 128, 3, 2)      # 28 -> 56

        # -> Output: Level 4

        conv_l4_out = _conv_layer(conv_l4_d7, 3, 3, 1, relu=False, instance_norm=False)
        output_l4 = tf.nn.tanh(conv_l4_out) * 0.58 + 0.5

        # -----------------------------------------
        # Processing: Level 3,  Input size: 56 x 56

        conv_l3_d2 = stack(conv_l3_d1, conv_t3a)
        conv_l3_d3 = _conv_multi_block(conv_l3_d2, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d2
        conv_l3_d4 = _conv_multi_block(conv_l3_d3, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d3
        conv_l3_d5 = _conv_multi_block(conv_l3_d4, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d4
        conv_l3_d6 = stack(_conv_multi_block(conv_l3_d5, 5, num_maps=128, instance_norm=instance_norm), conv_l3_d1)
        conv_l3_d7 = stack(conv_l3_d6, conv_t3b)

        conv_l3_d8 = _conv_multi_block(conv_l3_d7, 3, num_maps=128, instance_norm=instance_norm)

        conv_t2a = _conv_tranpose_layer(conv_l3_d8, 64, 3, 2)       # 56 -> 112
        conv_t2b = _conv_tranpose_layer(conv_l3_d8, 64, 3, 2)       # 56 -> 112

        # -> Output: Level 3

        conv_l3_out = _conv_layer(conv_l3_d8, 3, 3, 1, relu=False, instance_norm=False)
        output_l3 = tf.nn.tanh(conv_l3_out) * 0.58 + 0.5

        # -------------------------------------------
        # Processing: Level 2,  Input size: 112 x 112

        conv_l2_d2 = stack(conv_l2_d1, conv_t2a)
        conv_l2_d3 = stack(_conv_multi_block(conv_l2_d2, 5, num_maps=64, instance_norm=instance_norm), conv_l2_d1)

        conv_l2_d4 = _conv_multi_block(conv_l2_d3, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d3
        conv_l2_d5 = _conv_multi_block(conv_l2_d4, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d4
        conv_l2_d6 = _conv_multi_block(conv_l2_d5, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d5
        conv_l2_d7 = stack(_conv_multi_block(conv_l2_d6, 7, num_maps=64, instance_norm=instance_norm), conv_l2_d1)

        conv_l2_d8 = stack(_conv_multi_block(conv_l2_d7, 5, num_maps=64, instance_norm=instance_norm), conv_t2b)
        conv_l2_d9 = _conv_multi_block(conv_l2_d8, 3, num_maps=64, instance_norm=instance_norm)

        conv_t1a = _conv_tranpose_layer(conv_l2_d9, 32, 3, 2)       # 112 -> 224
        conv_t1b = _conv_tranpose_layer(conv_l2_d9, 32, 3, 2)       # 112 -> 224

        # -> Output: Level 2

        conv_l2_out = _conv_layer(conv_l2_d9, 3, 3, 1, relu=False, instance_norm=False)
        output_l2 = tf.nn.tanh(conv_l2_out) * 0.58 + 0.5

        # -------------------------------------------
        # Processing: Level 1,  Input size: 224 x 224

        conv_l1_d2 = stack(conv_l1_d1, conv_t1a)
        conv_l1_d3 = stack(_conv_multi_block(conv_l1_d2, 5, num_maps=32, instance_norm=False), conv_l1_d1)

        conv_l1_d4 = _conv_multi_block(conv_l1_d3, 7, num_maps=32, instance_norm=False)

        conv_l1_d5 = _conv_multi_block(conv_l1_d4, 9, num_maps=32, instance_norm=instance_norm_level_1)
        conv_l1_d6 = _conv_multi_block(conv_l1_d5, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d5
        conv_l1_d7 = _conv_multi_block(conv_l1_d6, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d6
        conv_l1_d8 = _conv_multi_block(conv_l1_d7, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d7

        conv_l1_d9 = stack(_conv_multi_block(conv_l1_d8, 7, num_maps=32, instance_norm=False), conv_l1_d1)

        conv_l1_d10 = stack(_conv_multi_block(conv_l1_d9, 5, num_maps=32, instance_norm=False), conv_t1b)
        conv_l1_d11 = stack(conv_l1_d10, conv_l1_d1)

        conv_l1_d12 = _conv_multi_block(conv_l1_d11, 3, num_maps=32, instance_norm=False)

        # -> Output: Level 1

        conv_l1_out = _conv_layer(conv_l1_d12, 3, 3, 1, relu=False, instance_norm=False)
        output_l1 = tf.nn.tanh(conv_l1_out) * 0.58 + 0.5

        # ----------------------------------------------------------
        # Processing: Level 0 (x2 upscaling),  Input size: 224 x 224

        conv_l0 = _conv_tranpose_layer(conv_l1_d12, 8, 3, 2)        # 224 -> 448
        conv_l0_out = _conv_layer(conv_l0, 3, 3, 1, relu=False, instance_norm=False)

        output_l0 = tf.nn.tanh(conv_l0_out) * 0.58 + 0.5

    return output_l0, output_l1, output_l2, output_l3, output_l4, output_l5

def adversarial(image_):
    with tf.compat.v1.variable_scope("discriminator"):
        conv1 = _conv_layer(image_, 48, 11, 4, instance_norm = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        
        flat_size = 128 * 16 * 16
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.compat.v1.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = tf.compat.v1.nn.leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.compat.v1.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out


def _upsample_switch


def _conv_multi_block(input, max_size, num_maps, instance_norm):

    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)
    conv_3b = _conv_layer(conv_3a, num_maps, 3, 1, relu=True, instance_norm=instance_norm)

    output_tensor = conv_3b

    if max_size >= 5:

        conv_5a = _conv_layer(input, num_maps, 5, 1, relu=True, instance_norm=instance_norm)
        conv_5b = _conv_layer(conv_5a, num_maps, 5, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_5b)

    if max_size >= 7:

        conv_7a = _conv_layer(input, num_maps, 7, 1, relu=True, instance_norm=instance_norm)
        conv_7b = _conv_layer(conv_7a, num_maps, 7, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_7b)

    if max_size >= 9:

        conv_9a = _conv_layer(input, num_maps, 9, 1, relu=True, instance_norm=instance_norm)
        conv_9b = _conv_layer(conv_9a, num_maps, 9, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_9b)

    return output_tensor

def fourierDiscrim(input):
    with tf.compat.v1.variable_scope("fourierDiscrim"):
        
        flat = tf.compat.v1.layers.flatten(input)

        fc1 = _fully_connected_layer(flat, 1024)
        fc2 = _fully_connected_layer(fc1, 1024)
        fc3 = _fully_connected_layer(fc2, 1024)
        fc4 = _fully_connected_layer(fc3, 1024)

        out = tf.nn.softmax(_fully_connected_layer(fc4, 2, relu=False))
    return out

def stack(x, y):
    return tf.concat([x, y], 3)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):

    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding) + bias

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        net = tf.compat.v1.nn.leaky_relu(net)

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


def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    return tf.compat.v1.nn.leaky_relu(net)

def _conv_pixel_shuffle(net, num_filters, filter_size, factor):
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

def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

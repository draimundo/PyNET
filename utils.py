# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from functools import reduce
import tensorflow as tf
import scipy.stats as st
import numpy as np
import sys
import os

NUM_DEFAULT_TRAIN_ITERS = [100000, 35000, 20000, 20000, 5000, 5000]


def process_command_args(arguments):

    # Specifying the default parameters

    level = 5
    batch_size = 50

    train_size = 5000
    learning_rate = 5e-5

    eval_step = 200
    restore_iter = 0
    num_train_iters = 0
    save_mid_imgs = False

    model_dir = 'models/'
    dataset_dir = 'raw_images/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'

    fac_mse = 100
    fac_content = 0
    fac_ssim = 0
    fac_color = 0

    over_dir = 'mediatek_raw_over/'
    under_dir = 'mediatek_raw_under/'
    triple_exposure = True

    if level == 3 or level == 2:
        fac_content = 1
    if level == 1:
        fac_mse, fac_content = 50, 1
    if level == 0:
        fac_content, fac_ssim = 1, 20

    for args in arguments:

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])

        if args.startswith("save_mid_imgs"):
            save_mid_imgs = eval(args.split("=")[1])

        
        if args.startswith("fac_content"):
            fac_content = float(args.split("=")[1])
        if args.startswith("fac_mse"):
            fac_mse = float(args.split("=")[1])
        if args.startswith("fac_ssim"):
            fac_ssim = float(args.split("=")[1])
        if args.startswith("fac_color"):
            fac_color = float(args.split("=")[1])

        if args.startswith("triple_exposure"):
            triple_exposure = bool(args.split("=")[1])

        if args.startswith("over_dir"):
            over_dir = args.split("=")[1]

        if args.startswith("under_dir"):
            under_dir = args.split("=")[1]


    if restore_iter is None and level < 5:
        restore_iter = get_last_iter(level + 1)
        num_train_iters += restore_iter
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for PyNET's level " + str(level + 1) + ".")
            print("Aborting the training.")
            sys.exit()

    if num_train_iters is None:
        num_train_iters = NUM_DEFAULT_TRAIN_ITERS[level]

    print("The following parameters will be applied for CNN training:")

    print("Training level: " + str(level))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(learning_rate))
    print("Training iterations: " + str(num_train_iters))
    print("Evaluation step: " + str(eval_step))
    print("Restore Iteration: " + str(restore_iter))
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to the dataset: " + dataset_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Triple exposure: " + str(triple_exposure))
    if triple_exposure:
        print("Path to the over dir: " + over_dir)
        print("Path to the under dir: " + under_dir)
    print("Loss function=" + " content:" + str(fac_content) + " +MSE:" + str(fac_mse) + " +SSIM:" + str(fac_ssim) + " +color:" + str(fac_color))

    return level, batch_size, train_size, learning_rate, restore_iter, num_train_iters, triple_exposure, over_dir, under_dir,\
            dataset_dir, model_dir, vgg_dir, eval_step, save_mid_imgs, fac_content, fac_mse, fac_ssim, fac_color


def process_test_model_args(arguments):

    level = 0
    restore_iter = None

    dataset_dir = 'raw_images/'
    use_gpu = "true"

    orig_model = "false"

    for args in arguments:

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

        if args.startswith("orig"):
            orig_model = args.split("=")[1]

    if restore_iter is None and orig_model == "false":
        restore_iter = get_last_iter(level)
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for PyNET's level " + str(level) + ".")
            sys.exit()

    return level, restore_iter, dataset_dir, use_gpu, orig_model


def get_last_iter(level):

    saved_models = [int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir("models/")
                    if model_file.startswith("pynet_level_" + str(level))]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return -1


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')
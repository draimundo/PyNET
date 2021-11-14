# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from functools import reduce
import tensorflow as tf
import scipy.stats as st
import numpy as np
import sys
import os

NUM_DEFAULT_TRAIN_ITERS = [200000, 200000, 200000, 200000, 2000, 1000]
DEFAULT_BATCH_SIZE = [10, 12, 18, 48, 50, 50]

def process_command_args(arguments):

    # Specifying the default parameters

    level = 5
    batch_size = 50

    train_size = 5000
    learning_rate = 5e-5

    eval_step = 1000
    restore_iter = 0
    num_train_iters = None
    save_mid_imgs = False

    model_dir = 'models/'
    dataset_dir = 'raw_images/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'

    default_facs = True
    fac_mse = 0
    fac_content = 0
    fac_ssim = 0
    fac_color = 0

    over_dir = 'mediatek_raw_over/'
    under_dir = 'mediatek_raw_under/'
    triple_exposure = False
    up_exposure = False
    down_exposure = False

    upscale = 'transpose'



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
            default_facs = False
        if args.startswith("fac_mse"):
            fac_mse = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_ssim"):
            fac_ssim = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_color"):
            fac_color = float(args.split("=")[1])
            default_facs = False

        if args.startswith("triple_exposure"):
            triple_exposure = eval(args.split("=")[1])

        if args.startswith("up_exposure"):
            up_exposure = eval(args.split("=")[1])

        if args.startswith("down_exposure"):
            down_exposure = eval(args.split("=")[1])

        if args.startswith("over_dir"):
            over_dir = args.split("=")[1]

        if args.startswith("under_dir"):
            under_dir = args.split("=")[1]

        if args.startswith("upscale"):
            upscale = args.split("=")[1]

    if num_train_iters is None:
        num_train_iters = NUM_DEFAULT_TRAIN_ITERS[level]

    if restore_iter == 0 and level < 5:
        restore_iter = get_last_iter(level + 1, model_dir)
        print(restore_iter)
        
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for PyNET's level " + str(level + 1) + ".")
            print("Aborting the training.")
            sys.exit()
    num_train_iters += restore_iter

    if default_facs:
        fac_mse = 100
        if level == 3 or level == 2:
            fac_content = 1
        if level == 1:
            fac_mse, fac_content = 50, 1
        if level == 0:
            fac_mse, fac_content, fac_ssim = 20, 1, 20

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
    print("Upscaling method: " + upscale)
    print("Triple exposure: " + str(triple_exposure))
    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    if triple_exposure:
        print("Path to the over dir: " + over_dir)
        print("Path to the under dir: " + under_dir)
    print("Loss function=" + " content:" + str(fac_content) + " +MSE:" + str(fac_mse) + " +SSIM:" + str(fac_ssim) + " +color:" + str(fac_color))

    return level, batch_size, train_size, learning_rate, restore_iter, num_train_iters, triple_exposure, over_dir, under_dir,\
            dataset_dir, model_dir, vgg_dir, eval_step, save_mid_imgs, fac_content, fac_mse, fac_ssim, fac_color, upscale, up_exposure, down_exposure


def process_test_model_args(arguments):
    out_dir = 'single_exp/'
    model_dir = 'models/single_exp/'
    interval = 0
    use_gpu = True
    triple_exposure = False
    up_exposure = False
    down_exposure = False
    level = 5
    upscale = "transpose"

    for args in arguments:

        
        if args.startswith("out_dir"):
            out_dir = args.split("=")[1]

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("start_iter"):
            start_iter = int(args.split("=")[1])

        if args.startswith("stop_iter"):
            stop_iter = int(args.split("=")[1])

        if args.startswith("interval"):
            interval = int(args.split("=")[1])

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

        if args.startswith("triple_exposure"):
            triple_exposure = eval(args.split("=")[1])

        if args.startswith("up_exposure"):
            up_exposure = eval(args.split("=")[1])

        if args.startswith("down_exposure"):
            down_exposure = eval(args.split("=")[1])

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("upscale"):
            upscale = args.split("=")[1]

    if interval == 0:
        restore_iters = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith("pynet_level_" + str(level))])))
    else:
        restore_iters = range(start_iter,stop_iter,interval)
    restore_iters = reversed(restore_iters)

    print("The following parameters will be applied for CNN testing:")
    print("Training level: " + str(level))
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Upscaling method: " + upscale)
    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    print("Triple exposure: " + str(triple_exposure))

    return out_dir, model_dir, restore_iters, use_gpu, triple_exposure, level, upscale, up_exposure, down_exposure

def process_evaluate_model_args(arguments):
    dataset_dir = 'raw_images/'
    model_dir = 'models/single_exp/'
    dslr_dir = 'fujifilm/'
    phone_dir = 'mediatek_raw/'
    over_dir = 'mediatek_raw_over/'
    under_dir = 'mediatek_raw_under/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    batch_size = 10
    use_gpu = True
    level = 5
    interval = 0
    use_gpu = True
    triple_exposure = False
    up_exposure = False
    down_exposure = False
    upscale = "transpose"

    for args in arguments:
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("dslr_dir"):
            dslr_dir = args.split("=")[1]

        if args.startswith("phone_dir"):
            phone_dir = args.split("=")[1]

        if args.startswith("over_dir"):
            over_dir = args.split("=")[1]

        if args.startswith("under_dir"):
            under_dir = args.split("=")[1]

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("start_iter"):
            start_iter = int(args.split("=")[1])

        if args.startswith("stop_iter"):
            stop_iter = int(args.split("=")[1])

        if args.startswith("interval"):
            interval = int(args.split("=")[1])

        if args.startswith("interval"):
            interval = int(args.split("=")[1])

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

        if args.startswith("triple_exposure"):
            triple_exposure = eval(args.split("=")[1])

        if args.startswith("up_exposure"):
            up_exposure = eval(args.split("=")[1])

        if args.startswith("down_exposure"):
            down_exposure = eval(args.split("=")[1])

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("upscale"):
            upscale = args.split("=")[1]

    if interval == 0:
        restore_iters = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith("pynet_level_" + str(level))])))
    else:
        restore_iters = range(start_iter,stop_iter,interval)
    restore_iters = reversed(restore_iters)

    print("The following parameters will be applied for CNN evaluation:")
    print("Training level: " + str(level))
    print("Batch size: " + str(batch_size))
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to the dataset: " + dataset_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Upscaling method: " + upscale)
    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    print("Triple exposure: " + str(triple_exposure))
    if triple_exposure:
        print("Path to the over dir: " + over_dir)
        print("Path to the under dir: " + under_dir)

    return dataset_dir, dslr_dir, phone_dir, over_dir, under_dir, vgg_dir, batch_size, model_dir, restore_iters, use_gpu, triple_exposure, level, upscale, up_exposure, down_exposure

def get_last_iter(level, model_dir):

    saved_models = [int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir(model_dir)
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
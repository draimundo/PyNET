# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from functools import reduce
import tensorflow as tf
import scipy.stats as st
import numpy as np
import sys
import os

NUM_DEFAULT_TRAIN_ITERS = [100000, 35000, 20000, 20000, 5000, 5000]
DEFAULT_BATCH_SIZE = [10, 12, 18, 48, 50, 50]

def process_command_args(arguments):

    # Specifying the default parameters

    level = 5
    batch_size = None

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
    fac_l1 = 0
    fac_ssim = 0
    fac_ms_ssim = 0
    fac_color = 0
    fac_vgg = 0
    fac_texture = 0
    fac_lpips = 0
    fac_huber = 0
    fac_fourier = 0
    fac_unet = 0
    fac_uv = 0

    dslr_dir = 'fujifilm/'
    over_dir = 'mediatek_raw_over/'
    under_dir = 'mediatek_raw_under/'
    triple_exposure = False
    up_exposure = False
    down_exposure = False

    upscale = 'transpose'
    downscale = 'maxpool'
    self_att = False
    flat = 0
    mix_input = False
    padding='SAME'

    norm='instance'
    norm_level_1='none'
    norm_scale='instance'
    sn=True
 
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

        if args.startswith("fac_mse"):
            fac_mse = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_l1"):
            fac_l1 = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_ssim"):
            fac_ssim = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_ms_ssim"):
            fac_ms_ssim = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_color"):
            fac_color = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_vgg"):
            fac_vgg = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_texture"):
            fac_texture = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_lpips"):
            fac_lpips = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_huber"):
            fac_huber = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_fourier"):
            fac_fourier = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_unet"):
            fac_unet = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_uv"):
            fac_uv = float(args.split("=")[1])
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

        if args.startswith("dslr_dir"):
            dslr_dir = args.split("=")[1]

        if args.startswith("upscale"):
            upscale = args.split("=")[1]
        if args.startswith("downscale"):
            downscale = args.split("=")[1]
        if args.startswith("self_att"):
            self_att = eval(args.split("=")[1])
        if args.startswith("flat"):
            flat = int(args.split("=")[1])
        if args.startswith("mix_input"):
            mix_input = eval(args.split("=")[1])
        if args.startswith("padding"):
            padding = args.split("=")[1]

        if args.startswith("norm_level_1"):
            norm_level_1 = args.split("=")[1]
        if args.startswith("norm_scale"):
            norm_scale = args.split("=")[1]
        if args.startswith("norm"):
            norm = args.split("=")[1]
        if args.startswith("sn"):
            sn = eval(args.split("=")[1])
        

    if num_train_iters is None:
        num_train_iters = NUM_DEFAULT_TRAIN_ITERS[level]

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[level]

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
            fac_vgg = 1
        if level == 1:
            fac_mse, fac_vgg = 50, 1
        if level == 0:
            fac_mse, fac_vgg, fac_ssim = 20, 1, 20

    print("The following parameters will be applied for CNN training:")
    print("Training level: " + str(level))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(learning_rate))
    print("Training iterations: " + str(num_train_iters))
    print("Evaluation step: " + str(eval_step))
    print("Restore Iteration: " + str(restore_iter))
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to the dataset: " + dataset_dir)
    print("Path to dslr images: " + dslr_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Upscaling method: " + upscale)
    print("Downscaling method: " + downscale)
    print("Self-attention :" + str(self_att))
    print("Flat: " + str(flat))
    print("Flat+channels :" + str(mix_input))
    print("Padding :" + str(padding))
    print("Triple exposure: " + str(triple_exposure))
    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    print("Normalization: " + str(norm))
    print("First Level Normalization: " + str(norm_level_1))
    print("Rescale Normalization: " + str(norm_scale))
    print("Spectral Normalization: " + str(sn))
    if triple_exposure:
        print("Path to the over dir: " + over_dir)
        print("Path to the under dir: " + under_dir)
    print("Loss function=" +
        " mse:" + str(fac_mse) +
        " l1:" + str(fac_l1) +
        " ssim:" + str(fac_ssim) +
        " ms-ssim:" + str(fac_ms_ssim) +
        " color:" + str(fac_color) +
        " vgg:" + str(fac_vgg) +
        " texture:" + str(fac_texture) +
        " lpips:" + str(fac_lpips) +
        " huber:" + str(fac_huber) +
        " fourier:" + str(fac_fourier) +
        " unet:" + str(fac_unet) +
        " uv:" + str(fac_uv))

    return level, batch_size, train_size, learning_rate, restore_iter, num_train_iters,\
        triple_exposure, up_exposure, down_exposure, over_dir, under_dir, dslr_dir, norm, norm_level_1, norm_scale, sn,\
        dataset_dir, model_dir, vgg_dir, eval_step, save_mid_imgs, upscale, downscale, self_att, flat, mix_input, padding,\
        fac_mse, fac_l1, fac_ssim, fac_ms_ssim, fac_color, fac_vgg, fac_texture, fac_lpips, fac_huber, fac_fourier, fac_unet, fac_uv

def process_test_model_args(arguments):
    out_dir = 'single_exp/'
    model_dir = 'models/single_exp/'
    restore_iter = 0
    use_gpu = True
    triple_exposure = False
    up_exposure = False
    down_exposure = False
    level = 5
    upscale = 'transpose'
    downscale = 'maxpool'
    self_att = True

    for args in arguments:

        
        if args.startswith("out_dir"):
            out_dir = args.split("=")[1]

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

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
        if args.startswith("downscale"):
            downscale = args.split("=")[1]
        if args.startswith("self_att"):
            self_att = eval(args.split("=")[1])


    if restore_iter == 0:
        restore_iter = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith("pynet_level_" + str(level))])))
    else:
        restore_iter = [restore_iter]

    print("The following parameters will be applied for CNN testing:")
    print("Training level: " + str(level))
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Upscaling method: " + upscale)
    print("Downscaling method: " + downscale)
    print("Self-attention :" + str(self_att))
    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    print("Triple exposure: " + str(triple_exposure))

    return out_dir, model_dir, restore_iter, use_gpu, triple_exposure, level, upscale, downscale, self_att, up_exposure, down_exposure

def process_evaluate_model_args(arguments):
    dataset_dir = 'raw_images/'
    model_dir = 'models/single_exp/'
    dslr_dir = 'fujifilm/'
    phone_dir = 'mediatek_raw/'
    over_dir = 'mediatek_raw_over/'
    under_dir = 'mediatek_raw_under/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    level = 5
    batch_size = 10
    use_gpu = True
    restore_iter = 0
    triple_exposure = False
    up_exposure = False
    down_exposure = False
    upscale = 'transpose'
    downscale = 'maxpool'
    self_att = False
    flat = 0
    mix_input = False
    padding='SAME'
    norm='instance'
    norm_level_1='none'
    norm_scale='instance'
    sn=True

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

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

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
        if args.startswith("downscale"):
            downscale = args.split("=")[1]
        if args.startswith("self_att"):
            self_att = eval(args.split("=")[1])
        if args.startswith("flat"):
            flat = int(args.split("=")[1])
        if args.startswith("mix_input"):
            mix_input = eval(args.split("=")[1])
        if args.startswith("padding"):
            padding = args.split("=")[1]

        if args.startswith("norm_level_1"):
            norm_level_1 = args.split("=")[1]
        if args.startswith("norm_scale"):
            norm_scale = args.split("=")[1]
        if args.startswith("norm"):
            norm = args.split("=")[1]
        if args.startswith("sn"):
            sn = eval(args.split("=")[1])

    if restore_iter == 0:
        restore_iter = sorted(list(set([int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith("pynet_level_" + str(level))])))
    else:
        restore_iter = [restore_iter]

    print("The following parameters will be applied for CNN evaluation:")
    print("Training level: " + str(level))
    print("Batch size: " + str(batch_size))
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to the dataset: " + dataset_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Upscaling method: " + upscale)
    print("Downscaling method: " + downscale)
    print("Self-attention :" + str(self_att))
    print("Flat: " + str(flat))
    print("Flat+channels :" + str(mix_input))
    print("Padding :" + str(padding))
    print("Up exposure: " + str(up_exposure))
    print("Down exposure: " + str(down_exposure))
    print("Triple exposure: " + str(triple_exposure))
    print("Normalization: " + str(norm))
    print("Layer Normalization: " + str(norm_level_1))
    print("Rescale Normalization: " + str(norm_scale))
    print("Spectral Normalization: " + str(sn))
    if triple_exposure:
        print("Path to the over dir: " + over_dir)
        print("Path to the under dir: " + under_dir)

    return dataset_dir, dslr_dir, phone_dir, over_dir, under_dir, vgg_dir, batch_size, model_dir, restore_iter, use_gpu, triple_exposure, level, upscale, downscale, self_att, up_exposure, down_exposure, flat, mix_input, padding, norm, norm_level_1, norm_scale, sn

def get_last_iter(level, model_dir):

    saved_models = [int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith("pynet_level_" + str(level))]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return -1

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
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
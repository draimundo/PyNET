# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np
import sys
import os

from tqdm import tqdm
from RAdam import RAdamOptimizer
from load_dataset import load_train_patch, load_val_data
from model import pynet_g, texture_d, unet_d
import utils
import vgg
import lpips_tf

from skimage.filters import window

# Processing command arguments
dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir, restore_iter,\
        triple_exposure, up_exposure, down_exposure, over_dir, under_dir,\
        patch_w, patch_h, batch_size, train_size, learning_rate, eval_step, num_train_iters, level, \
        upscale, downscale, self_att, flat, mix_input, padding, norm, norm_level_1, norm_scale, sn,\
        fac_mse, fac_l1, fac_ssim, fac_ms_ssim, fac_color, fac_vgg, fac_texture, fac_fourier, fac_lpips, fac_huber, fac_unet, fac_uv \
    = utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches
if flat:
    FAC_PATCH = 1
    PATCH_DEPTH = 1
else:
    FAC_PATCH = 2
    PATCH_DEPTH = 4
if triple_exposure:
    PATCH_DEPTH *= 3
elif up_exposure or down_exposure:
    PATCH_DEPTH *= 2

PATCH_WIDTH = patch_w//FAC_PATCH
PATCH_HEIGHT = patch_h//FAC_PATCH
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

DSLR_SCALE = float(1) / (2 ** (max(level,0)))
TARGET_WIDTH = int(PATCH_WIDTH * FAC_PATCH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * FAC_PATCH * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)
tf.random.set_seed(0)

# Defining the model architecture
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    
    # Placeholders for training data
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image
    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 = \
        pynet_g(phone_, norm=norm, norm_scale=norm_scale, sn=sn, upscale=upscale, downscale=downscale, self_att=self_att, flat=flat, mix_input=mix_input, padding=padding)

    if level == 5:
        enhanced = output_l5
    if level == 4:
        enhanced = output_l4
    if level == 3:
        enhanced = output_l3
    if level == 2:
        enhanced = output_l2
    if level == 1:
        enhanced = output_l1
    if level == 0:
        enhanced = output_l0

    # Losses
    dslr_gray = tf.image.rgb_to_grayscale(dslr_)
    enhanced_gray = tf.image.rgb_to_grayscale(enhanced)

    # MSE loss
    loss_mse = tf.reduce_mean(tf.math.squared_difference(enhanced, dslr_))
    loss_generator = loss_mse * fac_mse
    loss_list = [loss_mse]
    loss_text = ["loss_mse"]

    # L1 loss
    if fac_l1 > 0:
        loss_l1 = tf.reduce_mean(tf.abs(tf.math.subtract(enhanced, dslr_)))
        loss_list.append(loss_l1)
        loss_text.append("loss_l1")
        loss_generator += loss_l1 * fac_l1

    # PSNR metric
    metric_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list.append(metric_psnr)
    loss_text.append("metric_psnr")

    # SSIM loss
    if level < 5: #SSIM needs at least 11*11 images
        loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(enhanced_gray, dslr_gray, 1.0))
        loss_list.append(loss_ssim)
        loss_text.append("loss_ssim")
        if fac_ssim > 0:
            loss_generator += loss_ssim * fac_ssim

    # MS-SSIM loss
    if level < 1: #MS-SSIM needs at least 11*2^4=176*176 images
        loss_ms_ssim = 1 - tf.reduce_mean(tf.image.ssim_multiscale(enhanced_gray, dslr_gray, 1.0))
        loss_list.append(loss_ms_ssim)
        loss_text.append("loss_ms_ssim")
        if fac_ms_ssim > 0:
            loss_generator += loss_ms_ssim * fac_ms_ssim
    
    # Color loss
    if fac_color > 0:
        enhanced_blur = utils.blur(enhanced)
        dslr_blur = utils.blur(dslr_)
        loss_color = tf.reduce_mean(tf.math.squared_difference(dslr_blur, enhanced_blur))
        loss_generator += loss_color * fac_color
        loss_list.append(loss_color)
        loss_text.append("loss_color")

    # Huber loss
    delta = 1
    abs_error = tf.abs(tf.math.subtract(enhanced, dslr_))
    quadratic = tf.math.minimum(abs_error, delta)
    linear = tf.math.subtract(abs_error, quadratic)
    loss_huber = tf.reduce_mean(0.5*tf.math.square(quadratic)+linear)
    if fac_huber > 0:
        loss_generator += loss_huber * fac_huber
        loss_list.append(loss_huber)
        loss_text.append("loss_huber")

    # ## AB loss - not differentiable
    # dslr_lab = rgb_to_lab(dslr_)
    # enhanced_lab = rgb_to_lab(enhanced)
    # enhanced_ab_blur = utils.blur(enhanced_lab)[..., -2:]
    # dslr_ab_blur = utils.blur(dslr_lab)[..., -2:]
    # loss_ab = tf.reduce_mean(tf.math.squared_difference(dslr_lab, enhanced_lab))
    # if fac_ab > 0:
    #     loss_generator += loss_ab * fac_ab
    #     loss_list.append(loss_ab)
    #     loss_text.append("loss_ab")

    ## UV loss
    dslr_yuv = tf.image.rgb_to_yuv(dslr_)
    enhanced_lab = tf.image.rgb_to_yuv(enhanced)
    enhanced_uv_blur = utils.blur(enhanced_lab)[..., -2:]
    dslr_uv_blur = utils.blur(dslr_yuv)[..., -2:]
    loss_uv = tf.reduce_mean(tf.abs(tf.math.subtract(dslr_uv_blur, enhanced_uv_blur)))
    if fac_uv > 0:
        loss_generator += loss_uv * fac_uv
        loss_list.append(loss_uv)
        loss_text.append("loss_uv")

    # Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_vgg = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_vgg)
    loss_text.append("loss_vgg")
    if fac_vgg > 0:
        loss_generator += loss_vgg * fac_vgg

    ## Adversarial loss - discrim
    if fac_texture > 0:
        adv_real = dslr_gray
        adv_fake = enhanced_gray

        pred_real = texture_d(adv_real, activation=False)
        pred_fake = texture_d(adv_fake, activation=False)

        loss_texture_g = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_fake - pred_real), 1e-10, 1.0)))
        loss_texture_d = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_real - pred_fake), 1e-10, 1.0)))

        loss_generator += loss_texture_g * fac_texture
        loss_list.append(loss_texture_g)
        loss_text.append("loss_texture")

    ## Fourier loss
    if fac_fourier > 0:
        h2d = np.float32(window('hann', (TARGET_WIDTH, TARGET_HEIGHT)))
        hann2d = tf.stack([h2d,h2d,h2d],axis=2) #stack for 3 color channels

        enhanced_filter = tf.cast(tf.multiply(enhanced, hann2d),tf.float32)
        dslr_filter = tf.cast(tf.multiply(dslr_, hann2d),tf.float32)

        # from NHWC to NCHW and back, rfft2d performed on 2 innermost dimensions
        enhanced_fft = tf.signal.rfft2d(tf.transpose(enhanced_filter, [0, 3, 1, 2]))
        enhanced_fft = tf.transpose(enhanced_fft,[0,2,3,1])
        enhanced_abs = tf.abs(enhanced_fft)
        enhanced_angle = tf.math.angle(enhanced_fft)

        dslr_fft = tf.signal.rfft2d(tf.transpose(dslr_filter, [0, 3, 1, 2]))
        dslr_fft = tf.transpose(dslr_fft,[0,2,3,1])
        dslr_abs = tf.abs(dslr_fft)
        dslr_angle = tf.math.angle(dslr_fft)

        loss_fourier = tf.reduce_mean(tf.abs(tf.math.subtract(enhanced_fft, dslr_fft)))
        
        loss_list.append(loss_fourier)
        loss_text.append("loss_fourier")
        loss_generator += loss_fourier * fac_fourier

    if fac_unet > 0:
        adv_real = dslr_
        adv_fake = enhanced

        pred_real = unet_d(adv_real, activation=False)
        pred_fake = unet_d(adv_fake, activation=False)

        loss_unet_g = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_fake - pred_real), 1e-10, 1.0)))
        loss_unet_d = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pred_real - pred_fake), 1e-10, 1.0)))

        loss_generator += loss_unet_g * fac_unet
        loss_list.append(loss_unet_g)
        loss_text.append("loss_unet")

    ## LPIPS
    if level < 1:
        loss_lpips = tf.reduce_mean(lpips_tf.lpips(enhanced, dslr_, net='alex'))
        loss_list.append(loss_lpips)
        loss_text.append("loss_lpips")
        if fac_lpips > 0:
            loss_generator += loss_lpips * fac_lpips

    # Final loss function
    loss_list.insert(0, loss_generator)
    loss_text.insert(0, "loss_generator")

    # Optimize network parameters
    vars_pynet_g = [v for v in tf.compat.v1.global_variables() if v.name.startswith("pynet_g")]
    train_step_pynet_g = RAdamOptimizer(learning_rate).minimize(loss_generator, var_list=vars_pynet_g)

    if fac_texture > 0:
        loss_texture_g_ = 0.0
        n_texture_d_ = 0.0
        lr_texture_d = learning_rate
        vars_texture_d = [v for v in tf.compat.v1.global_variables() if v.name.startswith("texture_d")]
        train_step_texture_d = RAdamOptimizer(lr_texture_d/10000.0).minimize(loss_texture_d, var_list=vars_texture_d)

    if fac_unet > 0:
        loss_unet_g_ = 0.0
        n_unet_d_ = 0.0
        lr_unet_d = learning_rate
        vars_unet_d = [v for v in tf.compat.v1.global_variables() if v.name.startswith("unet_d")]
        train_step_unet_d = RAdamOptimizer(lr_unet_d/10000.0).minimize(loss_unet_d, var_list=vars_unet_d)
    
    # Initialize and restore the variables
    print("Initializing variables")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=vars_pynet_g, max_to_keep=100)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if restore_iter%2 == 1:
        print("Restoring Variables of same level")
        saver.restore(sess, model_dir + "pynet_level_" + str(level) + "_iteration_" + str(restore_iter-1) + ".ckpt")
    elif level < 5:
        print("Restoring Variables of higher level")
        saver.restore(sess, model_dir + "pynet_level_" + str(level + 1) + "_iteration_" + str(restore_iter) + ".ckpt")

    # Loading training and validation data
    print("Loading validation data...")
    val_data, val_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure, flat)
    print("Validation data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure, flat)
    print("Training data was loaded\n")

    VAL_SIZE = val_data.shape[0]
    num_val_batches = int(val_data.shape[0] / batch_size)

    visual_crops_ids = np.random.randint(0, VAL_SIZE, batch_size)
    visual_val_crops = val_data[visual_crops_ids, :]
    visual_target_crops = val_answ[visual_crops_ids, :]

    print("Training network")

    iter_start = restore_iter+1 if restore_iter > 0 else 0
    logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "w+")
    logs.close()

    loss_pynet_g_ = 0.0

    for i in tqdm(range(iter_start, num_train_iters + 1)):
        # Train texture discriminator
        if fac_texture > 0:
            idx_texture_d = np.random.randint(0, train_size, batch_size)
            phone_texture_d = train_data[idx_texture_d]
            dslr_texture_d = train_answ[idx_texture_d]

            feed_texture_d = {phone_: phone_texture_d, dslr_: dslr_texture_d}
            [loss_g, loss_d] = sess.run([loss_texture_g, loss_texture_d], feed_dict=feed_texture_d)

            if loss_g < 3*loss_d:
                [loss_temp, temp] = sess.run([loss_texture_d, train_step_texture_d], feed_dict=feed_texture_d)
                n_texture_d_ += 1

        # Train unet discriminator
        if fac_unet > 0:
            idx_unet_d = np.random.randint(0, train_size, batch_size)
            phone_unet_d = train_data[idx_unet_d]
            dslr_unet_d = train_answ[idx_unet_d]

            feed_unet_d = {phone_: phone_unet_d, dslr_: dslr_unet_d}
            [loss_g, loss_d] = sess.run([loss_unet_g, loss_unet_d], feed_dict=feed_unet_d)

            if loss_g < 3*loss_d:
                [loss_temp, temp] = sess.run([loss_unet_d, train_step_unet_d], feed_dict=feed_unet_d)
                n_unet_d_ += 1

        # Train PyNET model
        idx_g = np.random.randint(0, train_size, batch_size)
        phone_g = train_data[idx_g]
        dslr_g = train_answ[idx_g]

        # Random flips and rotations
        if flat == 0:
            for k in range(batch_size):
                random_rotate = np.random.randint(1, 100) % 4
                phone_g[k] = np.rot90(phone_g[k], random_rotate)
                dslr_g[k] = np.rot90(dslr_g[k], random_rotate)
                random_flip = np.random.randint(1, 100) % 2
                if random_flip == 1:
                    phone_g[k] = np.flipud(phone_g[k])
                    dslr_g[k] = np.flipud(dslr_g[k])

        feed_g = {phone_: phone_g, dslr_: dslr_g}
        [loss_temp, temp] = sess.run([loss_generator, train_step_pynet_g], feed_dict=feed_g)
        loss_pynet_g_ += loss_temp / eval_step

        # Evaluate PyNET model
        if i % eval_step == 0:
            val_losses_g = np.zeros((1, len(loss_text)))

            if fac_texture > 0:
                val_loss_texture_d = 0.0
            if fac_unet > 0:
                val_loss_unet_d = 0.0

            for j in range(num_val_batches):
                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                valdict = {phone_: phone_images, dslr_: dslr_images}
                toRun = [loss_list]
                
                loss_temp = sess.run(toRun, feed_dict=valdict)
                val_losses_g += np.asarray(loss_temp) / num_val_batches

                if fac_texture > 0:
                    loss_temp = sess.run(loss_texture_d, feed_dict=valdict)
                    val_loss_texture_d += loss_temp / num_val_batches
                if fac_unet > 0:
                    loss_temp = sess.run(loss_unet_d, feed_dict=valdict)
                    val_loss_unet_d += loss_temp / num_val_batches

            logs_gen = "step %d | training: %.4g,  "  % (i, loss_pynet_g_)
            for idx, loss in enumerate(loss_text):
                logs_gen += "%s: %.4g; " % (loss, val_losses_g[0][idx])
            if fac_texture > 0:
                logs_gen += " | texture_d loss: %.4g; n_texture_d: %.4g" % (val_loss_texture_d, n_texture_d_)
            if fac_unet > 0:
                logs_gen += " | unet_d loss: %.4g; n_unet_d: %.4g" % (val_loss_unet_d, n_unet_d_)

            print(logs_gen)

            # Save the results to log file

            logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            loss_pynet_g_ = 0.0
            if fac_texture > 0:
                n_texture_d_ = 0.0
            if fac_unet > 0:
                n_unet_d_ = 0.0

            # Saving the model that corresponds to the current iteration
            saver.save(sess, model_dir + "pynet_level_" + str(level) + "_iteration_" + str(i) + ".ckpt", write_meta_graph=False)

        # Loading new training data
        if i % 1000 == 0  and i > 0:
            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure, flat)

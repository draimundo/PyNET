# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import imageio
import tensorflow as tf
import numpy as np
import sys

from tqdm import tqdm
from load_dataset import load_training_batch, load_val_data
from model import PyNET
import utils
import vgg




# Processing command arguments

level, batch_size, train_size, learning_rate, restore_iter, num_train_iters,\
dataset_dir, vgg_dir, eval_step, save_mid_imgs, fac_content, fac_mse, fac_ssim, fac_color\
        = utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches

PATCH_WIDTH, PATCH_HEIGHT = 128, 128

DSLR_SCALE = float(1) / (2 ** (max(level,0) - 1))
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)

# Defining the model architecture

with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    
    # Placeholders for training data

    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 4])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image

    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 = \
        PyNET(phone_, instance_norm=True, instance_norm_level_1=False)

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
    # MSE loss
    loss_mse = tf.reduce_mean(tf.math.squared_difference(enhanced, dslr_))
    loss_generator = loss_mse * fac_mse
    loss_list = [loss_mse]
    loss_text = ["loss_mse"]

    # PSNR loss
    loss_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list = [loss_psnr]
    loss_text = ["loss_psnr"]

    # SSIM loss
    if fac_ssim > 0:
        loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_, 1.0))
        loss_generator += loss_ssim * fac_ssim
        loss_list.append(loss_ssim)
        loss_text.append("loss_ssim")

    # MS-SSIM loss
    # loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))

    # Color loss
    if fac_color > 0:
        enhanced_blur = utils.blur(enhanced)
        dslr_blur = utils.blur(dslr_)
        loss_color = tf.reduce_mean(tf.math.squared_difference(dslr_blur, enhanced_blur))
        loss_generator += loss_color * fac_color
        loss_list.append(loss_color)
        loss_text.append("loss_color")

    # Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_content = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_content)
    loss_text.append("loss_content")
    if fac_content > 0:
        loss_generator += loss_content * fac_content

    # Final loss function
    loss_list.insert(0, loss_generator)
    loss_text.insert(0, "loss_generator")

    # Optimize network parameters

    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    # Initialize and restore the variables

    print("Initializing variables")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)

    if level < 5:
        print("Restoring Variables")
        saver.restore(sess, "models/pynet_level_" + str(level + 1) + "_iteration_" + str(restore_iter) + ".ckpt")

    # Loading training and val data
    print("Loading val data...")
    val_data, val_answ = load_val_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Val data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_training_batch(dataset_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Training data was loaded\n")

    VAL_SIZE = val_data.shape[0]
    num_val_batches = int(val_data.shape[0] / batch_size)

    visual_crops_ids = np.random.randint(0, VAL_SIZE, batch_size)
    visual_val_crops = val_data[visual_crops_ids, :]
    visual_target_crops = val_answ[visual_crops_ids, :]

    print("Training network")

    iter_start = restore_iter+1 if restore_iter > 0 else 0
    logs = open('models/' + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "w+")
    logs.close()

    training_loss = 0.0

    for i in tqdm(range(iter_start, num_train_iters + 1)):

        # Train PyNET model

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        # Random flips and rotations

        for k in range(batch_size):

            random_rotate = np.random.randint(1, 100) % 4
            phone_images[k] = np.rot90(phone_images[k], random_rotate)
            dslr_images[k] = np.rot90(dslr_images[k], random_rotate)
            random_flip = np.random.randint(1, 100) % 2

            if random_flip == 1:
                phone_images[k] = np.flipud(phone_images[k])
                dslr_images[k] = np.flipud(dslr_images[k])

        # Training step

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen], feed_dict={phone_: phone_images, dslr_: dslr_images})
        training_loss += loss_temp / eval_step

        if i % eval_step == 0:

            # Evaluate PyNET model
            val_losses_gen = np.zeros((1, len(loss_text)))

            for j in range(num_val_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                losses = sess.run([loss_list], \
                                      feed_dict={phone_: phone_images, dslr_: dslr_images})

                val_losses_gen += np.asarray(losses) / num_val_batches

            logs_gen = "step %d | training: %.4g,  "  % (i, training_loss)
            for idx, loss in enumerate(loss_text):
                logs_gen += "%s: %.4g; " % (loss, val_losses_gen[0][idx])

            print(logs_gen)

            # Save the results to log file

            logs = open('models/' + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # Save visual results for several val image crops
            if save_mid_imgs:
                enhanced_crops = sess.run(enhanced, feed_dict={phone_: visual_val_crops, dslr_: dslr_images})
                idx = 0
                for crop in enhanced_crops:
                    if idx < 10:
                        before_after = np.hstack((crop,
                                        np.reshape(visual_target_crops[idx], [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])))
                        imageio.imwrite("results/pynet_img_" + str(idx) + "_level_" + str(level) + "_iter_" + str(i) + ".jpg",
                                        before_after)
                    idx += 1

            training_loss = 0.0

            # Saving the model that corresponds to the current iteration
            saver.save(sess, "models/pynet_level_" + str(level) + "_iteration_" + str(i) + ".ckpt", write_meta_graph=False)

        # Loading new training data
        if i % 1000 == 0  and i > 0:
            del train_data
            del train_answ
            train_data, train_answ = load_training_batch(dataset_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
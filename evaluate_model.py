import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys

from load_dataset import load_test_data
from model import PyNET

import utils
import vgg

dataset_dir, dslr_dir, phone_dir, over_dir, under_dir, vgg_dir, batch_size, model_dir, restore_iters, use_gpu, triple_exposure, level, upscale, up_exposure, down_exposure = utils.process_evaluate_model_args(sys.argv)

DSLR_SCALE = float(1) / (2 ** (max(level,0) - 1))
PATCH_WIDTH, PATCH_HEIGHT = 128, 128
PATCH_DEPTH = 4
if triple_exposure:
    PATCH_DEPTH *= 3
elif up_exposure or down_exposure:
    PATCH_DEPTH *= 2
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

print("Loading testing data...")
test_data, test_answ = load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure, down_exposure)
print("Testing data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0] / batch_size)

time_start = datetime.now()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None
with tf.compat.v1.Session(config=config) as sess:
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 = \
        PyNET(phone_, instance_norm=True, instance_norm_level_1=False, upscale=upscale)

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

    saver = tf.compat.v1.train.Saver()

    dslr_gray = tf.image.rgb_to_grayscale(dslr_)
    enhanced_gray = tf.image.rgb_to_grayscale(dslr_)

    ## PSNR loss
    loss_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list = [loss_psnr]
    loss_text = ["loss_psnr"]

    ## L1 loss
    loss_l1 = tf.reduce_mean(tf.abs(tf.math.subtract(enhanced, dslr_)))
    loss_list.append(loss_l1)
    loss_text.append("loss_l1")

    ## Color loss
    # enhanced_blur = utils.blur(enhanced)
    # dslr_blur = utils.blur(dslr_)
    # loss_color = tf.reduce_mean(tf.math.squared_difference(dslr_blur, enhanced_blur))
    # loss_list.append(loss_color)
    # loss_text.append("loss_color")

    ## SSIM loss
    if level < 5:
        loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(enhanced_gray, dslr_gray, 1.0))
        loss_list.append(loss_ssim)
        loss_text.append("loss_ssim")

    # MS-SSIM loss
    if level < 5:
        loss_ms_ssim = 1 - tf.reduce_mean(tf.image.ssim_multiscale(enhanced_gray, dslr_gray, 1.0))
        loss_list.append(loss_ms_ssim)
        loss_text.append("loss_ms_ssim")

    ## Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_content = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_content)
    loss_text.append("loss_content")

    logs = open(model_dir + "test_" + "level" + str(level) + ".txt", "w+")
    logs.close()

    for restore_iter in tqdm(restore_iters):
        test_losses_gen = np.zeros((1, len(loss_text)))
        saver.restore(sess, model_dir + "pynet_level_" + str(level) + "_iteration_" + str(restore_iter) + ".ckpt")

        for j in range(num_test_batches):

            be = j * batch_size
            en = (j+1) * batch_size

            phone_images = test_data[be:en]
            dslr_images = test_answ[be:en]

            losses = sess.run(loss_list, feed_dict={phone_: phone_images, dslr_: dslr_images})
            test_losses_gen += np.asarray(losses) / num_test_batches

        logs_gen = "Losses - iter: " + str(restore_iter) + "-> "
        for idx, loss in enumerate(loss_text):
            logs_gen += "%s: %.4g; " % (loss, test_losses_gen[0][idx])
        logs_gen += '\n'
        print(logs_gen)

        logs = open(model_dir + "test_" + "level" + str(level) + ".txt", "a+")
        logs.write(logs_gen)
        logs.write('\n')
        logs.close()

print('total test time:', datetime.now() - time_start)
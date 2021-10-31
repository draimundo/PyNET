import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import tqdm

from load_dataset import load_test_data
from model import PyNET

import utils
import vgg

triple_exposure = False

level = 3
DSLR_SCALE = float(1) / (2 ** (max(level,0) - 1))
PATCH_WIDTH, PATCH_HEIGHT = 128, 128
PATCH_DEPTH = 4
if triple_exposure:
    PATCH_DEPTH *= 3
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

dataset_dir = 'raw_images/'
model_dir = 'models/single_exp/'
dslr_dir = 'fujifilm/'
phone_dir = 'mediatek_raw/'
over_dir = 'mediatek_raw_over/'
under_dir = 'mediatek_raw_under/'
vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
restore_iters = range(5600,25600,200)
batch_size = 10
use_gpu = True

print("Loading testing data...")
test_data, test_answ = load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir)
print("Testing data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0] / batch_size)

time_start = datetime.now()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None
with tf.compat.v1.Session(config=config) as sess:
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

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

    saver = tf.compat.v1.train.Saver()

    ## PSNR loss
    loss_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list = [loss_psnr]
    loss_text = ["loss_psnr"]

    ## Color loss
    # enhanced_blur = utils.blur(enhanced)
    # dslr_blur = utils.blur(dslr_)
    # loss_color = tf.reduce_mean(tf.math.squared_difference(dslr_blur, enhanced_blur))
    # loss_list.append(loss_color)
    # loss_text.append("loss_color")

    ## SSIM loss
    # loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_, 1.0))
    # loss_list.append(loss_ssim)
    # loss_text.append("loss_ssim")

    ## Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_content = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_content)
    loss_text.append("loss_content")



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
print('total test time:', datetime.now() - time_start)
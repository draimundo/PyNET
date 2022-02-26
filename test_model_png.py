# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import numpy as np
import tensorflow as tf
import imageio
from tqdm import tqdm
import os
import imageio
import utils
import sys

from model import pynet_g

from load_dataset import extract_bayer_channels

dataset_dir, dslr_dir, phone_dir, over_dir, under_dir, vgg_dir, batch_size, model_dir, restore_iters, use_gpu, triple_exposure, level, upscale, downscale, self_att, up_exposure, down_exposure, flat, mix_input, padding, norm, norm_level_1, norm_scale, sn = utils.process_evaluate_model_args(sys.argv)


TARGET_DEPTH = 3
if flat:
    FAC_PATCH = 2
    PATCH_DEPTH = 1
else:
    FAC_PATCH = 1
    PATCH_DEPTH = 4
if triple_exposure:
    PATCH_DEPTH *= 3
elif up_exposure or down_exposure:
    PATCH_DEPTH *= 2

IMAGE_HEIGHT, IMAGE_WIDTH = 1500, 2000
DSLR_SCALE = float(1) / (2 ** (max(level,0) - 1))
MAX_SCALE = float(1) / (2 ** (5 - 1))

IMAGE_HCROP= int(np.floor(IMAGE_HEIGHT * MAX_SCALE)/MAX_SCALE)
IMAGE_WCROP = int(np.floor(IMAGE_WIDTH * MAX_SCALE)/MAX_SCALE)

TARGET_HEIGHT = int(np.floor(IMAGE_HCROP * DSLR_SCALE))
TARGET_WIDTH = int(np.floor(IMAGE_WCROP * DSLR_SCALE))

PATCH_HEIGHT = int(np.floor(IMAGE_HCROP*DSLR_SCALE)/DSLR_SCALE*FAC_PATCH)
PATCH_WIDTH = int(np.floor(IMAGE_WCROP*DSLR_SCALE)/DSLR_SCALE*FAC_PATCH)

TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == False else None

if not os.path.isdir("results/full-resolution/"+ model_dir):
    os.makedirs("results/full-resolution/"+ model_dir, exist_ok=True)

with tf.compat.v1.Session(config=config) as sess:

    # Placeholders for test data
    phone_ = tf.compat.v1.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])

    # generate enhanced image
    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 =\
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

    # Loading pre-trained model
    saver = tf.compat.v1.train.Saver()

    # Processing full-resolution RAW images
    test_dir_full = 'validation_full_resolution_visual_data/' + phone_dir
    test_dir_over = 'validation_full_resolution_visual_data/' + over_dir
    test_dir_under = 'validation_full_resolution_visual_data/' + under_dir

    test_photos = [f for f in os.listdir(test_dir_full) if os.path.isfile(test_dir_full + f)]
    test_photos.sort()

    print("Loading images")
    images = np.zeros((len(test_photos), PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH))
    for i, photo in tqdm(enumerate(test_photos)):
        print("Processing image " + photo)

        In = np.asarray(imageio.imread((test_dir_full + photo)))
        if not flat:
            In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(imageio.imread((test_dir_over + photo)))
            Iu = np.asarray(imageio.imread((test_dir_full + photo)))
            if not flat:
                Io = extract_bayer_channels(Io)
                Iu = extract_bayer_channels(Iu)
            I = np.dstack((In, Io, Iu))
            images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
        elif up_exposure:
            Io = np.asarray(imageio.imread((test_dir_over + photo)))
            if not flat:
                Io = extract_bayer_channels(Io)

            I = np.dstack((In, Io))
            images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
        elif down_exposure:
            Iu = np.asarray(imageio.imread((test_dir_full + photo)))
            if not flat:
                Iu = extract_bayer_channels(Iu)

            I = np.dstack((In, Iu))
            images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
        else:
            I = In
            if flat:
                images[i,..., 0] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
            else:
                images[i,..., 0:PATCH_DEPTH] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]

    print("Images loaded")
    # Run inference

    for restore_iter in tqdm(restore_iters):
        saver.restore(sess, model_dir + "pynet_level_" + str(level) + "_iteration_" + str(restore_iter) + ".ckpt")
        
        for i, photo in enumerate(test_photos):
            enhanced_tensor = sess.run(enhanced, feed_dict={phone_: [images[i,...]]})
            enhanced_image = np.reshape(enhanced_tensor, [TARGET_HEIGHT, TARGET_WIDTH, 3])

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            imageio.imwrite("results/full-resolution/"+ model_dir + photo_name + "_level_" + str(level) +
                        "_iteration_" + str(restore_iter) + ".png", enhanced_image)

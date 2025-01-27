# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import numpy as np
import tensorflow as tf
import imageio
from tqdm import tqdm
import os
import rawpy
import utils
import sys

from model import PyNET

from load_dataset import extract_bayer_channels

out_dir, model_dir, restore_iters, use_gpu, triple_exposure, level, upscale, up_exposure, down_exposure = utils.process_test_model_args(sys.argv)

IMAGE_HEIGHT, IMAGE_WIDTH = 1500, 2000
DSLR_SCALE = float(1) / (2 ** (max(level,0) - 1))
MAX_SCALE = float(1) / (2 ** (5 - 1))
IMAGE_HEIGHT, IMAGE_WIDTH = 1500, 2000

IMAGE_HCROP= int(np.floor(IMAGE_HEIGHT * MAX_SCALE)/MAX_SCALE)
IMAGE_WCROP = int(np.floor(IMAGE_WIDTH * MAX_SCALE)/MAX_SCALE)

TARGET_HEIGHT = int(np.floor(IMAGE_HCROP * DSLR_SCALE))
TARGET_WIDTH = int(np.floor(IMAGE_WCROP * DSLR_SCALE))

PATCH_HEIGHT = int(np.floor(IMAGE_HCROP*DSLR_SCALE)/DSLR_SCALE)
PATCH_WIDTH = int(np.floor(IMAGE_WCROP*DSLR_SCALE)/DSLR_SCALE)

TARGET_DEPTH = 3
PATCH_DEPTH = 4
if triple_exposure:
    PATCH_DEPTH *= 3
elif up_exposure or down_exposure:
    PATCH_DEPTH *= 2

TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == False else None

if not os.path.isdir("results/full-resolution/"+ out_dir):
    os.makedirs("results/full-resolution/"+ out_dir, exist_ok=True)

with tf.compat.v1.Session(config=config) as sess:

    # Placeholders for test data
    phone_ = tf.compat.v1.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])

    # generate enhanced image
    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 =\
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

    # Loading pre-trained model
    saver = tf.compat.v1.train.Saver()

    # Processing full-resolution RAW images
    test_dir_full = 'validation_full_resolution_visual_data/mediatek_raw_normal/'
    test_dir_over = 'validation_full_resolution_visual_data/mediatek_raw_over/'
    test_dir_under = 'validation_full_resolution_visual_data/mediatek_raw_under/'

    test_photos = [f for f in os.listdir(test_dir_full) if os.path.isfile(test_dir_full + f)]
    test_photos.sort()

    print("Loading images")
    images = np.zeros((len(test_photos), PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH))
    for i, photo in tqdm(enumerate(test_photos)):
        print("Processing image " + photo)

        In = np.asarray(rawpy.imread((test_dir_full + photo)).raw_image.astype(np.float32))
        In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(rawpy.imread((test_dir_over + photo)).raw_image.astype(np.float32))
            Io = extract_bayer_channels(Io)

            Iu = np.asarray(rawpy.imread((test_dir_full + photo)).raw_image.astype(np.float32))
            Iu = extract_bayer_channels(Iu)

            I = np.dstack((In, Io, Iu))
        elif up_exposure:
            Io = np.asarray(rawpy.imread((test_dir_over + photo)).raw_image.astype(np.float32))
            Io = extract_bayer_channels(Io)

            I = np.dstack((In, Io))
        elif down_exposure:
            Iu = np.asarray(rawpy.imread((test_dir_full + photo)).raw_image.astype(np.float32))
            Iu = extract_bayer_channels(Iu)

            I = np.dstack((In, Iu))
        else:
            I = In

        images[i,...] = I[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
    print("Images loaded")
    # Run inference

    for restore_iter in tqdm(restore_iters):
        saver.restore(sess, model_dir + "pynet_level_" + str(level) + "_iteration_" + str(restore_iter) + ".ckpt")
        
        for i, photo in enumerate(test_photos):
            enhanced_tensor = sess.run(enhanced, feed_dict={phone_: [images[i,...]]})
            enhanced_image = np.reshape(enhanced_tensor, [TARGET_HEIGHT, TARGET_WIDTH, 3])

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            imageio.imwrite("results/full-resolution/"+ out_dir + photo_name + "_level_" + str(level) +
                        "_iteration_" + str(restore_iter) + ".png", enhanced_image)

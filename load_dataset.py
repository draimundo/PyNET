# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from __future__ import print_function
import imageio
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


def load_val_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir):

    val_directory_dslr = dataset_dir + 'val/fujifilm/'
    val_directory_phone = dataset_dir + 'val/mediatek_raw/'

    if triple_exposure:
        val_directory_over = dataset_dir + 'val/' + over_dir
        val_directory_under = dataset_dir + 'val/' + under_dir

    PATCH_DEPTH = 4
    if triple_exposure:
        PATCH_DEPTH *= 3

    # NUM_VAL_IMAGES = 1204
    NUM_VAL_IMAGES = len([name for name in os.listdir(val_directory_phone)
                           if os.path.isfile(os.path.join(val_directory_phone, name))])

    val_data = np.zeros((NUM_VAL_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    val_answ = np.zeros((NUM_VAL_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    for i in tqdm(range(0, NUM_VAL_IMAGES)):

        In = np.asarray(imageio.imread(val_directory_phone + str(i) + '.png'))
        In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(imageio.imread(val_directory_over + str(i) + '.png'))
            Io = extract_bayer_channels(Io)
            Iu = np.asarray(imageio.imread(val_directory_under + str(i) + '.png'))
            Iu = extract_bayer_channels(Iu)
            val_data[i, :] = np.dstack((In, Io, Iu))
        else:
            val_data[i, :] = In

        I = Image.open(val_directory_dslr + str(i) + '.png')
        I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        val_answ[i, :] = I

    return val_data, val_answ


def load_training_batch(dataset_dir, TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir):

    train_directory_dslr = dataset_dir + 'train/fujifilm/'
    train_directory_phone = dataset_dir + 'train/mediatek_raw/'

    if triple_exposure:
        train_directory_over = dataset_dir + 'train/' + over_dir
        train_directory_under = dataset_dir + 'train/' + under_dir

    PATCH_DEPTH = 4
    if triple_exposure:
        PATCH_DEPTH *= 3

    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)
    #TRAIN_IMAGES = np.arange(0, TRAIN_SIZE)
    train_data = np.zeros((TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    train_answ = np.zeros((TRAIN_SIZE, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    i = 0
    for img in tqdm(TRAIN_IMAGES):

        In = np.asarray(imageio.imread(train_directory_phone + str(img) + '.png'))
        In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(imageio.imread(train_directory_over + str(img) + '.png'))
            Io = extract_bayer_channels(Io)
            Iu = np.asarray(imageio.imread(train_directory_under + str(img) + '.png'))
            Iu = extract_bayer_channels(Iu)
            train_data[i, :] = np.dstack((In, Io, Iu))
        else:
            train_data[i, :] = In

        I = Image.open(train_directory_dslr + str(img) + '.png')
        I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        train_answ[i, :] = I

        i += 1

    return train_data, train_answ
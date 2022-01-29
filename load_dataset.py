# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from __future__ import print_function
import imageio
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import rawpy
import matplotlib.pyplot as plt

def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


def load_val_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure = False, down_exposure = False, flat=False, dslr_dir = 'fujifilm/'):

    val_directory_dslr = dataset_dir + 'val/' + dslr_dir
    val_directory_phone = dataset_dir + 'val/mediatek_raw/'

    val_directory_over = dataset_dir + 'val/' + over_dir
    val_directory_under = dataset_dir + 'val/' + under_dir

    PATCH_DEPTH = 4
    FAC_SCALE = 1
    if flat:
        PATCH_DEPTH = 1
        FAC_SCALE = 2
    if triple_exposure:
        PATCH_DEPTH *= 3
    elif up_exposure or down_exposure:
        PATCH_DEPTH *= 2

    # NUM_VAL_IMAGES = 1204
    NUM_VAL_IMAGES = len([name for name in os.listdir(val_directory_phone)
                           if os.path.isfile(os.path.join(val_directory_phone, name))])

    val_data = np.zeros((NUM_VAL_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    val_answ = np.zeros((NUM_VAL_IMAGES, int(PATCH_WIDTH * DSLR_SCALE/FAC_SCALE), int(PATCH_HEIGHT * DSLR_SCALE/FAC_SCALE), 3))

    for i in tqdm(range(0, NUM_VAL_IMAGES)):

        In = np.asarray(imageio.imread(val_directory_phone + str(i) + '.png'))
        if not flat:
            In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(imageio.imread(val_directory_over + str(i) + '.png'))
            Iu = np.asarray(imageio.imread(val_directory_under + str(i) + '.png'))
            if not flat:
                Io = extract_bayer_channels(Io)
                Iu = extract_bayer_channels(Iu)
                val_data[i, :] = np.dstack((In, Io, Iu))
            else:
                val_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Io, Iu))
        elif down_exposure:
            Iu = np.asarray(imageio.imread(val_directory_under + str(i) + '.png'))
            if not flat:
                Iu = extract_bayer_channels(Iu)
                val_data[i, :] = np.dstack((In, Iu))
            else:
                val_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Iu))
        elif up_exposure:
            Io = np.asarray(imageio.imread(val_directory_over + str(i) + '.png'))
            if not flat:
                Io = extract_bayer_channels(Io)
                val_data[i, :] = np.dstack((In, Io))
            else:
                val_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Io))
        else:
            if not flat:
                val_data[i, :] = In
            else:
                val_data[i, ..., 0] = In

        I = Image.open(val_directory_dslr + str(i) + '.png')
        I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE/FAC_SCALE), int(PATCH_HEIGHT * DSLR_SCALE/FAC_SCALE), 3])) / 255
        val_answ[i, :] = I

    return val_data, val_answ

def load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure = False, down_exposure = False, flat = False):

    test_directory_dslr = dataset_dir + 'test/fujifilm/'
    test_directory_phone = dataset_dir + 'test/mediatek_raw/'

    test_directory_over = dataset_dir + 'test/' + over_dir
    test_directory_under = dataset_dir + 'test/' + under_dir

    PATCH_DEPTH = 4
    FAC_SCALE = 1
    if flat:
        PATCH_DEPTH = 1
        FAC_SCALE = 2
    if triple_exposure:
        PATCH_DEPTH *= 3
    elif up_exposure or down_exposure:
        PATCH_DEPTH *= 2
        

    # NUM_VAL_IMAGES = 1204
    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    test_answ = np.zeros((NUM_TEST_IMAGES, int(PATCH_WIDTH * DSLR_SCALE / FAC_SCALE), int(PATCH_HEIGHT * DSLR_SCALE / FAC_SCALE), 3))

    for i in tqdm(range(0, NUM_TEST_IMAGES)):

        In = np.asarray(imageio.imread(test_directory_phone + str(i) + '.png'))
        if not flat:
            In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(imageio.imread(test_directory_over + str(i) + '.png'))
            Iu = np.asarray(imageio.imread(test_directory_under + str(i) + '.png'))
            if not flat:
                Io = extract_bayer_channels(Io)
                Iu = extract_bayer_channels(Iu)
                test_data[i, :] = np.dstack((In, Io, Iu))
            else:
                test_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Io, Iu))
        elif down_exposure:
            Iu = np.asarray(imageio.imread(test_directory_under + str(i) + '.png'))
            if not flat:
                Iu = extract_bayer_channels(Iu)
                test_data[i, :] = np.dstack((In, Iu))
            else:
                test_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Iu))
        elif up_exposure:
            Io = np.asarray(imageio.imread(test_directory_over + str(i) + '.png'))
            if not flat:
                Io = extract_bayer_channels(Io)
                test_data[i, :] = np.dstack((In, Io))
            else:
                test_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Io))
        else:
            if not flat:
                test_data[i, :] = In
            else:
                test_data[i, ..., 0] = In

        I = Image.open(test_directory_dslr + str(i) + '.png')
        I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE/FAC_SCALE), int(PATCH_HEIGHT * DSLR_SCALE/FAC_SCALE), 3])) / 255
        test_answ[i, :] = I

    return test_data, test_answ

def load_training_batch(dataset_dir, TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, triple_exposure, over_dir, under_dir, up_exposure = False, down_exposure = False, flat = False, dslr_dir = 'fujifilm/'):

    train_directory_dslr = dataset_dir + 'train/' + dslr_dir
    train_directory_phone = dataset_dir + 'train/mediatek_raw/'

    train_directory_over = dataset_dir + 'train/' + over_dir
    train_directory_under = dataset_dir + 'train/' + under_dir

    PATCH_DEPTH = 4
    FAC_SCALE = 1
    if flat:
        PATCH_DEPTH = 1
        FAC_SCALE = 2
    if triple_exposure:
        PATCH_DEPTH *= 3
    elif up_exposure or down_exposure:
        PATCH_DEPTH *= 2

    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)
    #TRAIN_IMAGES = np.arange(0, TRAIN_SIZE)
    train_data = np.zeros((TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    train_answ = np.zeros((TRAIN_SIZE, int(PATCH_WIDTH * DSLR_SCALE / FAC_SCALE), int(PATCH_HEIGHT * DSLR_SCALE / FAC_SCALE), 3))

    i = 0
    for img in tqdm(TRAIN_IMAGES):

        In = np.asarray(imageio.imread(train_directory_phone + str(img) + '.png'))
        if not flat:
            In = extract_bayer_channels(In)

        if triple_exposure:
            Io = np.asarray(imageio.imread(train_directory_over + str(img) + '.png'))
            Iu = np.asarray(imageio.imread(train_directory_under + str(img) + '.png'))
            if not flat:
                Io = extract_bayer_channels(Io)
                Iu = extract_bayer_channels(Iu)
                train_data[i, :] = np.dstack((In, Io, Iu))
            else:
                train_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Io, Iu))
        elif down_exposure:
            Iu = np.asarray(imageio.imread(train_directory_under + str(img) + '.png'))
            if not flat:
                Iu = extract_bayer_channels(Iu)
                train_data[i, :] = np.dstack((In, Iu))
            else:
                train_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Iu))
        elif up_exposure:
            Io = np.asarray(imageio.imread(train_directory_over + str(img) + '.png'))
            if not flat:
                Io = extract_bayer_channels(Io)
                train_data[i, :] = np.dstack((In, Io))
            else:
                train_data[i, ..., 0:PATCH_DEPTH] = np.dstack((In, Io))
        else:
            if not flat:
                train_data[i, :] = In
            else:
                train_data[i, ..., 0] = In

        I = Image.open(train_directory_dslr + str(img) + '.png')
        I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE / FAC_SCALE), int(PATCH_HEIGHT * DSLR_SCALE / FAC_SCALE), 3])) / 255
        train_answ[i, :] = I

        i += 1

    return train_data, train_answ



def load_test_data_raw():
    dataset_dir = 'raw_images/'

    test_directory_dslr = dataset_dir + 'test/fujifilm/'
    test_directory_phone = dataset_dir + 'test/mediatek_dng/converted/'
        
    PATCH_WIDTH = 256
    PATCH_HEIGHT = 256
    DSLR_SCALE = 1

    # NUM_VAL_IMAGES = 1204
    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])
    test_data = np.zeros((NUM_TEST_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))
    test_answ = np.zeros((NUM_TEST_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    for i in tqdm(range(0, NUM_TEST_IMAGES)):

        I = Image.open(test_directory_phone + str(i) + '.png')
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        test_data[i, :] = I

        I = Image.open(test_directory_dslr + str(i) + '.png')
        #I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        test_answ[i, :] = I

    return test_data, test_answ
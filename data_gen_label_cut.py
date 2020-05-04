#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-9-21 上午10:45
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : data_gen_label_cut.py
"""
import math

import numpy as np
from skimage.transform import resize
from skimage.io import imread

# from Cutup import cutup

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, Sequence
import pandas as pd
from imagenet_utils import preprocess_input


class BaseSequence(Sequence):

    def __init__(self, image_filenames, labels, batch_size,input_size):

        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.input_size = input_size
    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))
    def preprocess_img(self, img_path):
        img = imread(img_path)
        if self.use:
            img = self.eraser(img)
            datagen = ImageDataGenerator(
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip = True,
            )
            img = datagen.random_transform(img)
            #     print("train set")
            # if not self.use:
            #     print("val set")
            # plt.imshow((img).astype(np.uint8))
            # pylab.show()

        img = img[:, :, ::-1]
        return img
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = preprocess_input(batch_x)
        return np.array([
            resize(imread(file_name), (self.input_size, self.input_size))
               for file_name in batch_x]), np.array(batch_y)


def smooth_labels(y, smooth_factor = 0.1):
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def data_flow(batch_size, num_classes, input_size):
    train_dir = '../datat/train.csv'
    val_dir = '../datat/val.csv'
    train_df = pd.read_csv(train_dir)
    val_df = pd.read_csv(val_dir)

    train_label = train_df['type']
    train_label = list(train_label)
    print('total train_label: %d ' % len(train_label))
    train_path = train_df['FileName']
    train_path = list(train_path)
    print('total train_path: %d ' % len(train_path))
    val_label = val_df['type']
    val_label = list(val_label)
    print('total val_label: %d ' % len(val_label))
    val_path = val_df['FileName']
    val_path = list(val_path)
    print('total val_path: %d ' % len(val_path))

    train_label = np_utils.to_categorical(train_label, num_classes)
    val_label = np_utils.to_categorical(val_label, num_classes)

    # 标签平滑
    train_label = smooth_labels(train_label)

    train_sequence = BaseSequence(train_path, train_label, batch_size, input_size)
    validation_sequence = BaseSequence(val_path, val_label, batch_size, input_size)

    return train_sequence, validation_sequence


if __name__ == '__main__':
    train_sequence, validation_sequence = data_flow(batch_size = 4, num_classes = 9, input_size = 224)

    for i in range(3):
        print(i)
        batch_data, bacth_label = train_sequence.__getitem__(i)
        print(batch_data, bacth_label)
        batch_data_, bacth_label_ = validation_sequence.__getitem__(i)
        print(batch_data_, bacth_label_)

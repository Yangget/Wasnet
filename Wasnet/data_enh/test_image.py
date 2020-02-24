# -*- coding: utf-8 -*-
"""
 @Time    : 19-12-18 上午9:57
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : test_image.py
"""
import glob as gb

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from data_enh.Mixup import mixup
from data_enh.Cutmix import cutmix
from data_enh.Ranera import get_random_eraser
def getData():
    batch_x = []
    size = 400
    img_path = gb.glob("./images/*.jpg")
    id = 0

    for ip in img_path:
        img = Image.open(ip)
        img = img.convert('RGB')
        img = img.resize((size, size), Image.ANTIALIAS)
        img = np.array(img)
        img = img[:, :, ::-1]
        batch_x.append(img)
    batch_y = np.arange(0, 4, 1)
    batch_x = np.array(batch_x)
    return batch_x, batch_y


def convert_to_one_hot(y, C):
    return np.around(np.eye(C)[y.reshape(-1)].T,decimals=5)


if __name__ == '__main__':



    batch_x, batch_y = getData( )
    batch_y = convert_to_one_hot(batch_y, 4)

    batch_x_t = []

    # batch_x_0_ = batch_x
    # eraser = get_random_eraser(s_h=0.3, pixel_level=True)
    # for img in batch_x_0_:
    #     batch_x_0 = eraser(img)
    #     batch_x_t.append(batch_x_0)

    batch_x_1, batch_y_1 = mixup(batch_x, batch_y)
    batch_x_t = list(batch_x_1)

    # batch_x_2, batch_y_2 = cutmix(batch_x, batch_y)
    # batch_x_t = list(batch_x_2)

    # np.savetxt('./result/batch_y_alpha(0.5).csv',batch_y,delimiter=',')


    fig = plt.figure(figsize=(30, 16))
    for i in range(0, 4):
        ax = fig.add_subplot(1, 4, i + 1)
        plt.imshow(batch_x_t[i].astype(np.uint8))
        plt.axis('off')
    plt.savefig('./result/Mixup.jpg')
    plt.show( )

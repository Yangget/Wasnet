# -*- coding: utf-8 -*-
"""
 @Time    : 20-1-4 下午4:28
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : test_centerimg.py
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def resize_scale(img,img_size,fill_value=120):

    scale = img_size / max(img.size[:2])
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
    img = np.array(img)

    h, w = img.shape[:2]
    shape = (img_size, img_size) + img.shape[2:]
    background = np.full(shape, fill_value, np.uint8)
    center_x = (img_size - w) // 2
    center_y = (img_size - h) // 2
    background[center_y:center_y + h, center_x:center_x + w] = img

    return background

if __name__ == '__main__':
    img_np = []
    img_size = 224

    img_path = './center_img/img_1288.jpg'
    img = Image.open(img_path)
    img_np.append(img)

    resize_img = img.resize((img_size,img_size))
    img_np.append(resize_img)

    scale_img = resize_scale(img, img_size)
    img_np.append(scale_img)

    fig = plt.figure(figsize=(30, 16))
    for i in range(0, 3):
        ax = fig.add_subplot(1, 3, i + 1)
        plt.imshow(img_np[i])
        plt.axis('off')
    plt.savefig('./center_img/img_1288_center.jpg')
    plt.show( )

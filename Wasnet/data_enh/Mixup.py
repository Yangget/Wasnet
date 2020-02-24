# -*- coding: utf-8 -*-
"""
 @Time    : 19-12-18 上午9:54
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : Mixup.py
"""
import numpy as np


def mixup(batch_x, batch_y):
    """
    :param alpha:
    :param batch_x:
    :param batch_y:
    :return:
    """
    alpha = 0.5
    size = batch_x.shape[0]
    l = np.random.beta(alpha, alpha, size)

    X_l = l.reshape(size, 1, 1, 1)
    y_l = l.reshape(size, 1)

    X1 = batch_x
    Y1 = batch_y
    X2 = batch_x[::-1]
    Y2 = batch_y[::-1]

    X = X1 * X_l + X2 * (1 - X_l)
    Y = Y1 * y_l + Y2 * (1 - y_l)

    return X, Y

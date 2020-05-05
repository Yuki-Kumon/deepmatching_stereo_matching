# -*- coding: utf-8 -*-

"""
optimize loop for sub pix
Author :
    Yuki Kumon
Last Update :
    2020-01-13
"""


import numpy as np


def optimize_loop(img_dis, coefficient, alpha, exclusion, size):

    error = 0.0
    img_dis = image_threshold(img_dis)
    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            sum_d = img_dis[i, j - 1] + img_dis[i, j + 1] + img_dis[i - 1, j] + img_dis[i + 1, j]
            # a = coefficient[i, j] * (check[i, j] + 1.0)
            a = coefficient[i, j]
            d_new = (-a * img_dis[i, j] + alpha * sum_d) / (-a + 4.0 * alpha)
            img_dis[i, j] = d_new
    # 逆から
    for i in range(exclusion, size[0] - exclusion - 1):
        for j in range(exclusion, size[1] - exclusion - 1):
            i = size[0] - i - 1
            j = size[1] - j - 1
            sum_d = img_dis[i, j - 1] + img_dis[i, j + 1] + img_dis[i - 1, j] + img_dis[i + 1, j]
            a = coefficient[i, j]
            d_new = (-a * img_dis[i, j] + alpha * sum_d) / (-a + 4.0 * alpha)
            error += abs(img_dis[i, j] - d_new)
            # print(str(img_dis[i, j]) + ',' + str(sum_d) + ',' + str(d_new) + ',' + str(coefficient[i, j]) + ',' + str(error))
            img_dis[i, j] = d_new
    return img_dis, error


def image_threshold(arr, threshold=[0, 10]):
    arr = np.where(arr > threshold[1], threshold[1], arr)
    arr = np.where(arr < threshold[0], threshold[0], arr)

    return arr

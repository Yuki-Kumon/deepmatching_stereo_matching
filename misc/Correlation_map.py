# -*- coding: utf-8 -*-

"""
correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-14
"""


import sys

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from misc.Feature_value import Feature_value
    print('misc.Feature_value loaded')
except ModuleNotFoundError as e:
    print(e)
try:
    from Feature_value import Feature_value
    print('Feature_value loaded')
except ModuleNotFoundError as e:
    print(e)


class Correlation_map():
    '''
    deepmatchingみたいにピラミッド状の特徴マップを作成する
    '''

    def __init__(self, img, template, window_size=3, co_map_height=5, feature_name='cv2.TM_CCOEFF_NORMED'):
        if img.shape != template.shape:
            print('use same size images!(サイズが違うと悲しい気持ちになるので(そのうち対応したいですね))')
            sys.exit()
        self.img = img
        self.template = template
        self.window_size = window_size
        self.co_map_height = co_map_height

        self.exclusive_pix = int((window_size - 1) / 2)
        self.co_map_exclusive = int((co_map_height - 1) / 2)
        self.image_size = [x for x in img.shape]

        self.Feature = Feature_value(feature_name=feature_name)

    def _create_atomic_patch(self):
        '''
        重なりありのatomic patchを作成する
        '''
        atomic_patch = np.empty((
                                self.image_size[0] - 2 * self.exclusive_pix,
                                self.image_size[1] - 2 * self.exclusive_pix,
                                self.window_size,
                                self.window_size
                                ))
        for i in range(self.exclusive_pix, self.image_size[0] - self.exclusive_pix):
            for j in range(self.exclusive_pix, self.image_size[1] - self.exclusive_pix):
                atomic_patch[i - self.exclusive_pix, j - self.exclusive_pix] = \
                    self.img[i - self.exclusive_pix:i + self.exclusive_pix + 1, j - self.exclusive_pix:j + self.exclusive_pix + 1]

        # データ型を符号なし整数に変えておく
        self.atomic_patch = atomic_patch.astype(np.uint8)

    """
    def _create_initial_co_map(self):
        '''
        初めの相関マップを計算する。
        x方向に長い短冊型にする(y方向に検索する意味はないので)
        '''
        co_map = np.empty((
            self.atomic_patch.shape[0],
            self.atomic_patch.shape[1],
            self.co_map_height,
            self.atomic_patch.shape[1]
        ))

        for i in range(self.exclusive_pix + self.co_map_exclusive, self.image_size[0] - self.exclusive_pix - self.co_map_exclusive):
            for j in range(self.exclusive_pix, self.image_size[1] - self.exclusive_pix):
                co_here = self.Feature(
                    self.atomic_patch[i - self.exclusive_pix - self.co_map_exclusive, j - self.exclusive_pix],
                    self.template[i - self.exclusive_pix - self.co_map_exclusive:i + self.exclusive_pix + self.co_map_exclusive + 1, :]
                )
                co_map[i - self.exclusive_pix, j - self.exclusive_pix] = co_here
        self.co_map = co_map
    """

    def _create_simple_initial_co_map(self):
        '''
        初めの相関マップを計算する。
        普通のdeepmatching用
        '''
        co_map = np.empty((
            self.atomic_patch.shape[0],
            self.atomic_patch.shape[1],
            self.atomic_patch.shape[0],
            self.atomic_patch.shape[1]
        ))
        for i in range(self.exclusive_pix, self.image_size[0] - self.exclusive_pix):
            for j in range(self.exclusive_pix, self.image_size[1] - self.exclusive_pix):
                co_here = self.Feature(
                    self.atomic_patch[i - self.exclusive_pix - self.co_map_exclusive, j - self.exclusive_pix],
                    self.template
                )
                co_map[i - self.exclusive_pix, j - self.exclusive_pix] = co_here
        self.co_map = co_map

    def _aggregation(self, p_window_size=3, stride=2):
        '''
        aggregation to make upper class co_map
        pooling window:3
        stride:1
        '''
        co_map_list = []
        co_map_list.append(self.co_map)

        # 原著の手順に従ってmulti-level correlation pyramidを計算する
        N = 4
        while N < np.max(self.co_map.shape[:2]):
            hoge


class Maxpool(nn.Module):

    def __init__(self, window=3, stride=2, padding=1):
        super(Maxpool, self).__init__()

        self.pool = nn.MaxPool2d(window, stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


if __name__ == '__main__':
    """
    sanity check
    """

    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[:100, :100]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[:100, :100]

    """
    hoge = np.random.rand(1, 2, 3, 4)

    print(np.max(hoge.shape[2:]))
    """

    # img = torch.from_numpy(np.random.rand(1, 1, 32, 32))
    Mod = Maxpool()
    # print(Mod(img).size())

    map = np.random.rand(98, 98, 98, 98)
    map_len_1 = int((map.shape[2]) / 2)
    map_len_2 = int((map.shape[3]) / 2)
    res = np.empty((map.shape[0], map.shape[1], map_len_1, map_len_2))
    # max pool
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            res[i, j] = Mod(torch.from_numpy(map[i, j][None])).numpy()[0]
    # shift and average
    for i in range(map_len_1):
        for j in raneg(map_len_2):
            upper_left
    print(res)


    """
    cls = Correlation_map(img1, img2)
    cls._create_atomic_patch()
    # cls._create_initial_co_map()
    cls._create_simple_initial_co_map()
    print(cls.co_map.shape)
    """

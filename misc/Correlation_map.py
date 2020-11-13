# -*- coding: utf-8 -*-

"""
correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-24
"""


import sys

import numpy as np
import cv2

import torch
import torch.nn as nn
# import torch.nn.functional as F

from joblib import Parallel, delayed

try:
    from misc.Feature_value import Feature_value
except ModuleNotFoundError as e:
    from Feature_value import Feature_value


class Correlation_map():
    '''
    deepmatchingみたいにピラミッド状の特徴マップを作成する
    '''

    def __init__(self, img, template, window_size=3, feature_name='cv2.TM_CCOEFF_NORMED', is_color=False):
        if img.shape != template.shape:
            print('use same size images!(サイズが違うと悲しい気持ちになるので(そのうち対応したいですね))')
            sys.exit()
        self.img = img
        self.template = template
        self.window_size = window_size
        self.lam = 1.4  # rectification

        self.is_color = is_color

        self.exclusive_pix = int((window_size - 1) / 2)
        self.image_size = [x for x in img.shape]

        self.Feature = Feature_value(feature_name=feature_name)

        self.Maxpool = Maxpool()
        self.Maxpool.eval()

    def _create_atomic_patch(self):
        '''
        重なりありのatomic patchを作成する
        '''
        if self.is_color:
            atomic_patch = np.empty((
                                    self.image_size[0] - 2 * self.exclusive_pix,
                                    self.image_size[1] - 2 * self.exclusive_pix,
                                    self.window_size,
                                    self.window_size,
                                    3,
                                    ))
            for i in range(self.exclusive_pix, self.image_size[0] - self.exclusive_pix):
                for j in range(self.exclusive_pix, self.image_size[1] - self.exclusive_pix):
                    atomic_patch[i - self.exclusive_pix, j - self.exclusive_pix] = \
                        self.img[i - self.exclusive_pix:i + self.exclusive_pix + 1, j - self.exclusive_pix:j + self.exclusive_pix + 1]
        else:
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
                    self.atomic_patch[i - self.exclusive_pix, j - self.exclusive_pix],
                    self.template
                )
                co_map[i - self.exclusive_pix, j - self.exclusive_pix] = co_here
        self.co_map = co_map

    def _aggregation(self, map):
        '''
        aggregation to make upper class co_map
        pooling window:3
        stride:2
        この設定は変えない(なぜならつらいから。。。)
        '''
        map_len_1 = int((map.shape[2]) / 2)
        map_len_2 = int((map.shape[3]) / 2)
        res = np.empty((map.shape[0], map.shape[1], map_len_1, map_len_2))

        # max pool
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                res[i, j] = self.Maxpool(torch.from_numpy(map[i, j][None])).numpy()[0]

        # shift and average
        output = np.empty((map_len_1, map_len_2, map_len_1, map_len_2))

        # process
        def process(i, j, res):
            # 平均を取る対象のインデックスを計算しておく
            upper_left = [i * 2, j * 2]
            upper_right = [i * 2, j * 2 + 1]
            lower_left = [i * 2 + 1, j * 2]
            lower_right = [i * 2 + 1, j * 2 + 1]

            # shiftはpaddingしているので不要
            upper_left_img = res[upper_left[0], upper_left[1]]
            upper_right_img = res[upper_right[0], upper_right[1]]
            lower_left_img = res[lower_left[0], lower_left[1]]
            lower_right_img = res[lower_right[0], lower_right[1]]
            # average
            return (upper_left_img + upper_right_img + lower_left_img + lower_right_img) / 4, (i, j)

        # parallel processing
        response = Parallel(n_jobs=-1)([delayed(process)(i, j, res) for i in range(map_len_1) for j in range(map_len_2)])
        # assign
        for res_here in response:
            output[res_here[1]] = res_here[0]

        return output

    def _multi_level_correlation_pyramid(self):
        '''
        aggregationを繰り返し、multi-level correlation pyramidを計算する
        初めの特徴マップはFeatureに依存(現時点では原著と異なりznccを使っている)
        '''

        # 原著の手順に従ってmulti-level correlation pyramidを計算する
        co_map_list = []
        co_map = self.co_map
        co_map = self._rectification(co_map)
        co_map_list.append(co_map)
        N = 1
        iteration = 1
        while N < np.min(self.co_map.shape[:2]):
            # aggregation
            aggregated_map = self._aggregation(co_map)
            aggregated_map = self._rectification(aggregated_map)
            co_map_list.append(aggregated_map)
            del co_map
            co_map = aggregated_map
            N *= 2
            iteration += 1
        self.co_map_list = co_map_list
        self.iteration = iteration
        self.N_map = N

    def _rectification(self, map):
        return map**self.lam

    def __call__(self):
        '''
        特徴マップの計算までを行う
        '''
        self._create_atomic_patch()
        # print('complete to create atomic patch')
        self._create_simple_initial_co_map()
        # print('complete to create initial correlation map')
        self._multi_level_correlation_pyramid()
        # print('complete to create multi-level correlation pyramid')
        # print('pyramid level: {}, N={}'.format(self.iteration, self.N_map))

        return self.co_map_list


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

    # atomicな特徴マップが一辺が2^nじゃないとバグるカス実装です。。。
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[:130, :130]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[:130, :130]

    cls = Correlation_map(img1, img2)

    hoge = cls()
    # print(cls.co_map.shape)

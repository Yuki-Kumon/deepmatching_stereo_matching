# -*- coding: utf-8 -*-

"""
image cut tool
Author :
    Yuki Kumon
Last Update :
    2019-12-22
"""


import numpy as np
from tqdm import trange

import sys
sys.path.append('.')

from misc.Correlation_map import Correlation_map
from misc.Matching import Matching
from misc.Calc_difference import Calc_difference


class ImageCutSolver():
    '''
    小画像に切ってそれぞれdeepmatchingに入れる
    '''

    def __init__(
        self, img1, img2,
        image_size=[32, 32], stride=[32, 32], window_size=5,
        feature_name='cv2.TM_CCOEFF_NORMED', degree_map_mode='elevation',
        padding=False
    ):
        self.img_shape = img1.shape
        assert self.img_shape == img2.shape, '2枚の画像は同じサイズ！'
        self.img1 = img1
        self.img2 = img2
        self.stride = stride
        self.window_size = window_size
        self.degree_map_mode = degree_map_mode
        self.exclusive_pix = int((window_size - 1) / 2)
        self.image_size = image_size
        self.trimed_size = [image_size[i] + 2 * self.exclusive_pix for i in range(2)]
        self.feature_name = feature_name

        if padding:
            self._padding()

        # loop length
        self.len = [int(np.floor((self.img_shape[i] - self.trimed_size[i]) / self.stride[i])) for i in range(2)]

        self.padding = padding

    def _padding(self):
        img1 = self.img1
        img2 = self.img2
        self.img1 = np.zeros([
            img1.shape[0] + 2 * self.exclusive_pix,
            img1.shape[1] + 2 * self.exclusive_pix
        ])
        self.img1[
            self.exclusive_pix:-self.exclusive_pix,
            self.exclusive_pix:-self.exclusive_pix
        ] = img1
        self.img1 = self.img1.astype(np.uint8)
        self.img2 = np.zeros([
            img2.shape[0] + 2 * self.exclusive_pix,
            img2.shape[1] + 2 * self.exclusive_pix
        ])
        self.img1[
            self.exclusive_pix:-self.exclusive_pix,
            self.exclusive_pix:-self.exclusive_pix
        ] = img1
        self.img2 = self.img2.astype(np.uint8)

    def _cut_and_pool(self):
        '''
        画像を切り出しリストで保存
        '''
        self.img1_sub = []
        self.img2_sub = []
        self.img_index = []

        for j in range(self.len[1]):
            for i in range(self.len[0]):
                self.img1_sub.append(self.img1[
                    self.stride[0] * i:self.stride[0] * i + self.trimed_size[0],
                    self.stride[1] * j:self.stride[1] * j + self.trimed_size[1]
                ])
                self.img2_sub.append(self.img2[
                    self.stride[0] * i:self.stride[0] * i + self.trimed_size[0],
                    self.stride[1] * j:self.stride[1] * j + self.trimed_size[1]
                ])
                self.img_index.append([i, j])

    def _solver(self, solve_image, solve_template):
        """
        小画像に対しdeepmatchingを実施する
        """
        co_cls = Correlation_map(solve_image, solve_template, window_size=self.window_size, feature_name=self.feature_name)
        co_cls()

        cls = Matching(co_cls)
        out = cls()

        del co_cls
        del cls

        # d_map = Calc_difference.cal_map(out, mode='elevation')
        return Calc_difference.cal_map(out, mode=self.degree_map_mode)

    def _execute_matching(self):
        """
        小画像ごとにマッチングを行い結果を結合
        重なりなしのみ対応
        """
        self.d_map = np.empty(
            [self.stride[i] * self.img_index[-1][i] + self.image_size[i] for i in range(2)],
            dtype=np.uint8)
        for idx in trange(len(self.img_index), desc='executing deepmatching'):
            i_here, j_here = self.img_index[idx]
            self.d_map[
                self.stride[0] * i_here:self.stride[0] * i_here + self.image_size[0],
                self.stride[1] * j_here:self.stride[1] * j_here + self.image_size[1]
            ] = self._solver(self.img1_sub[idx], self.img2_sub[idx])
            """
            print(self.stride[0] * i_here, self.stride[0] * i_here + self.image_size[0],
            self.stride[1] * j_here, self.stride[1] * j_here + self.image_size[1])
            """


if __name__ == '__main__':
    """
    sanity check
    """
    import cv2

    """
    start = 100
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[start:start + 260, start:start + 260]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[start:start + 260, start:start + 260]

    cls = ImageCutSolver(img1, img2)
    cls._cut_and_pool()
    # cls._solver(cls.img1_sub[0], cls.img2_sub[0])
    cls._execute_matching()
    print(cls.d_map.shape)

    cv2.imwrite('./here.png', img1)
    cv2.imwrite('./output.png', cls.d_map * 30 + 100)
    """

    start_x = 1650
    start_y = 3500
    img_loaded = cv2.imread('./data/after-before-crossdis.tif')
    img1_raw = img_loaded[:, :, 1]  # 地震前
    img2_raw = img_loaded[:, :, 2]  # 地震後
    img3_raw = img_loaded[:, :, 0]  # 変化マップ

    img1 = img1_raw[start_y:start_y + 500, start_x:start_x + 500]
    img2 = img2_raw[start_y:start_y + 500, start_x:start_x + 500]
    img3 = img3_raw[start_y:start_y + 500, start_x:start_x + 500]

    cls = ImageCutSolver(img1, img2, degree_map_mode='distance', window_size=15)
    cls._cut_and_pool()
    # cls._solver(cls.img1_sub[0], cls.img2_sub[0])
    cls._execute_matching()
    print(cls.d_map.shape)

    cv2.imwrite('./here.png', img1[:cls.d_map.shape[0], :cls.d_map.shape[1]])
    cv2.imwrite('./here2.png', img3[:cls.d_map.shape[0], :cls.d_map.shape[1]])
    cv2.imwrite('./output.png', cls.d_map * 30 + 100)

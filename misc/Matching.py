# -*- coding: utf-8 -*-

"""
matching on correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-18
"""


import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from misc.Correlation_map import Correlation_map
    print('misc.Correlation_map loaded')
except ModuleNotFoundError as e:
    print(e)
try:
    from Correlation_map import Correlation_map
    print('Correlation_map loaded')
except ModuleNotFoundError as e:
    print(e)


class Matching():
    '''
    multi-level correlation pyramidからマッチングを行う
    原著の14式に従って計算していく
    ピラミッド上位の類似度を足し算していけば良さそう？
    '''

    def __init__(self, Co_obj=None):
        try:
            Co_obj.co_map_list
        except AttributeError as e:
            print('Error!: {}'.format(e))
            print('please run \'obj=Correlation_map()\' and \'obj()\' first.')
            sys.exit()

        self.obj = Co_obj

    def _initial_move_map(self):
        '''
        ピラミッドの頂点での動きのマップを計算する
        '''
        map = np.zeros((3, self.obj.co_map_list[-1].shape[0], self.obj.co_map_list[-1].shape[1]))
        self.map = map

    def _B(self):
        '''
        原著の14式
        座標はchildrenに合わせる(4倍する)
        各パッチの左上の座標を用いて計算する
        '''
        # Nとiterarionとmapはこの後更新する。N > 0でwhileループで回せばいいと思う
        N = self.obj.N_map
        idx = self.obj.iteration
        map_here = self.map


class Zero_padding(nn.Module):
    '''
    原著の14式の計算のため、ゼロパディングしておく
    '''

    def __init__(self):
        self.m = nn.ZeroPad2d(1)

    def forward(self, x):
        return self.m(x)


if __name__ == '__main__':
    """
    sanity check
    """
    # atomicな特徴マップが一辺が2^nじゃないとバグるカス実装です。。。
    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)[500:500 + 130, 500:500 + 130]
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)[500:500 + 130, 500:500 + 130]

    co_cls = Correlation_map(img1, img2)
    co_cls()

    # 試しに書き出し
    """
    cv2.imwrite('out.png', co_cls.co_map_list[1][0, 0] * 50)
    cv2.imwrite('out0.png', co_cls.co_map_list[0][0, 0] * 50)
    cv2.imwrite('here.png', img1)
    """

    cls = Matching(co_cls)
    cls._initial_move_map()

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
        self.Padding = Zero_padding()
        self.Padding.eval()

    def _calc_near_match(self, co_map, p, p_dot):
        '''
        pとp_dotはco_mapの解像度に合わせた座標
        原著の13式に従って対応点を計算する
        13式のmを計算する
        '''
        # pに対応した特徴マップを取り出す
        map_on_p = co_map[p[0], p[1]]
        # 端対策にゼロパディングする
        map_on_p_padded = self.Padding(torch.from_numpy(map_on_p[None])).numpy()[0]


    def _initial_move_map(self):
        '''
        ピラミッドの頂点での動きのマップを計算する
        '''
        # 各ピクセルに(対応するx, 対応するy, 相関値s)が格納されている
        map = np.zeros((3, self.obj.co_map_list[-1].shape[0], self.obj.co_map_list[-1].shape[1]))
        for i in range(map.shape[1]):
            for j in range(map.shape[2]):
                pass
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

        # mapをゼロパディングする
        # map = self.Padding(torch.from_numpy(map_here[i, j][None])).numpy()[0]

        len_1 = map_here.shape[1]
        len_2 = map_here.shape[2]
        o_list = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        for i in range(len_1):
            for j in range(len_2):
                p = np.array([2 * i + len_1, 2 * j + len_2])
                p_dot = [int(2 * map_here[0, i, j]), int(2 * map_here[1, i, j])]
                for index in range(4):
                    o_here = o_list[index]
                    p_here = p + o_here
                    print(p_here)
                print(p_dot)


class Zero_padding(nn.Module):
    '''
    原著の14式の計算のため、ゼロパディングしておく
    '''

    def __init__(self):
        super(Zero_padding, self).__init__()
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
    cls._B()

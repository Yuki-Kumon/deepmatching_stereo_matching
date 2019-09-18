# -*- coding: utf-8 -*-

"""
matching on correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-18
"""


import sys

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

    def _update_co_map(self, map, children):
        '''
        14式に従ってパッチ間の類似度を計算する
        '''

    def _B(self):
        '''
        原著の14式。。。
        '''
        pass


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

    cls = Matching()

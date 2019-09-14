# -*- coding: utf-8 -*-

"""
correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-14
"""


import numpy as np

class Correlation_map():
    '''
    deepmatchingみたいにピラミッド状の特徴マップを作成する
    '''

    def __init__(self, img, template, window_size=3):
        self.img = img
        self.template = template
        self.window_size = window_size

        self.xclusive_pix = int((window_size - 1) / 2)
        self.image_size = [x for x in img.shape]

    def create_atomic_patch(self):
        '''
        重なりありのatomic patchを作成する
        '''
        # atomic_patch = np.empty(())

# -*- coding: utf-8 -*-

"""
Feature Value
Author :
    Yuki Kumon
Last Update :
    2019-09-14
"""


import sys

import cv2


class Feature_value():
    '''
    2つのパッチ間の特徴量を計算
    '''

    def __init__(self, feature_name='cv2.TM_CCOEFF_NORMED'):
        FEATURE_NAME_LIST = ['cv2.TM_CCOEFF_NORMED']

        if feature_name not in FEATURE_NAME_LIST:
            print('invalid feature_name \'{}\' is inputed!'.format(feature_name))
            sys.exit()

        self.method = eval(feature_name)

    def __call__(self, img, template):
        res = cv2.matchTemplate(img, template, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# -*- coding: utf-8 -*-

"""
image loader
Author :
    Yuki Kumon
Last Update :
    2019-11-19
"""


import cv2


class Loader():
    '''
    class to load images
    '''

    def __init__(
        self,
        path,
        start=[3500, 1760],
        size=[68, 260],
        integrated=True
    ):

        self.img_list = []
        if integrated:
            assert type(path) == 'str', 'path must be string when integrated mode'
            img_loaded = cv2.imread(path)
            img1_raw = img_loaded[:, :, 1]  # 地震前
            img2_raw = img_loaded[:, :, 2]  # 地震後
            img3_raw = img_loaded[:, :, 0]  # 変化マップ

            self.img_list.append(img1_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]])
            self.img_list.append(img2_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]])
            self.img_list.append(img3_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]])
        else:
            assert type(path) == 'list', 'path must be list when not integrated mode'
            img1_raw = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)
            img2_raw = cv2.imread(path[1], cv2.IMREAD_GRAYSCALE)

            self.img_list.append(img1_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]])
            self.img_list.append(img2_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]])

    def __call__(self):
        return self.img_list

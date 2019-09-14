# -*- coding: utf-8 -*-

"""
correlation map
Author :
    Yuki Kumon
Last Update :
    2019-09-14
"""


import numpy as np
import cv2


class Correlation_map():
    '''
    deepmatchingみたいにピラミッド状の特徴マップを作成する
    '''

    def __init__(self, img, template, window_size=3):
        self.img = img
        self.template = template
        self.window_size = window_size

        self.exclusive_pix = int((window_size - 1) / 2)
        self.image_size = [x for x in img.shape]

    def create_atomic_patch(self):
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

        self.atomic_patch = atomic_patch


if __name__ == '__main__':
    """
    sanity check
    run on deepmatching_stereo_matching dir
    """

    img1 = cv2.imread('./data/band3s.tif', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./data/band3bs.tif', cv2.IMREAD_GRAYSCALE)

    cls = Correlation_map(img1, img2)
    cls.create_atomic_patch()

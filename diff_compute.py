# -*- coding: utf-8 -*-

"""
compute difference between two images
Author :
    Yuki Kumon
Last Update :
    2020-01-18
"""

import cv2
from misc.image_cut_solver import ImageCutSolver

import numpy as np

# file1 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/igarss/raw/_for_final/s=6,g=15,nofilter/elevation2.png'
# file1 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/igarss/raw/_for_final/s=6,g=15,filtering_num=4,window=3,median/elevation2.png'
# file1 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/rssj69/差し替え用の画像/setup1_elevation2.png'
file1 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/rssj69/差し替え/images/s=15_g=6_elevation2.png'
# file2 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/igarss/raw/temp_out/elevation2.png'
# file2 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/igarss/raw/_for_final/s=6,g=15,filtering_num=4,window=3/elevation2.png'
# file2 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/rssj69/差し替え用の画像/nofilter_elevation2.png'
file2 = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/rssj69/差し替え/images/s=15_g=6_nofilter_elevation2.png'
output_path = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/output/rssj69/差し替え/images/rgb_concated.png'

im1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
# im3 = np.zeros_like(im1, dtype='uint8')
im3 = im1  # same image to visualize better

rgb_im = np.dstack([im1, im3, im2])

ImageCutSolver.image_save(output_path, rgb_im, threshold=[0, 255])

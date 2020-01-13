# -*- coding: utf-8 -*-

"""
論文の図に貼るため、テンプレートマッチングの概念図を作りたい。
それに用いるための図を計算する。
Author :
    Yuki Kumon
Last Update :
    2019-10-01
"""


# import numpy as np
import cv2

import sys
sys.path.append('.')
from misc.Feature_value import Feature_value


start_x = 1650
start_y = 3500
img_loaded = cv2.imread('./data/after-before-crossdis.tif')
img1_raw = img_loaded[:, :, 1]  # 地震前
img2_raw = img_loaded[:, :, 2]  # 地震後
img3_raw = img_loaded[:, :, 0]  # 変化マップ

img1 = img1_raw[start_y:start_y + 500, start_x:start_x + 500]
img2 = img2_raw[start_y:start_y + 500, start_x:start_x + 500]
img3 = img3_raw[start_y:start_y + 500, start_x:start_x + 500]

# img1とimg2で相関マップを計算しておく
img1_patch = img1[100:149, 100:149]
Fet = Feature_value(feature_name='cv2.TM_CCOEFF_NORMED')
map = Fet(img1_patch, img2)

# save
cv2.imwrite('./img1_patch.png', img1_patch)
cv2.imwrite('./co_map_igarss.png', map * 100)
cv2.imwrite('./img2.png', img2)

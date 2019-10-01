# -*- coding: utf-8 -*-

"""
stereo mathing with deepmatching
Author :
    Yuki Kumon
Last Update :
    2019-10-01
"""

from misc.Correlation_map import Correlation_map
from misc.Matching import Matching
from misc.Calc_difference import Calc_difference

from absl import app, flags, logging
from absl.flags import FLAGS

import numpy as np
import cv2


flags.DEFINE_string('original_image_path', './data/band3s.tif', 'image path of original image')
flags.DEFINE_string('template_image_path', './data/band3bs.tif', 'image path of template image')
flags.DEFINE_list('image_cut_size', '68, 260', 'image size cut from start point')
flags.DEFINE_list('image_cut_start', '100, 100', 'point to cut image from')


def main(_argv):
    # load images
    img1_raw = cv2.imread(FLAGS.original_image_path, cv2.IMREAD_GRAYSCALE)
    img2_raw = cv2.imread(FLAGS.template_image_path, cv2.IMREAD_GRAYSCALE)

    # image cut
    image_cut_size = [int(x) for x in FLAGS.image_cut_size]
    image_cut_start = [int(x) for x in FLAGS.image_cut_start]


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

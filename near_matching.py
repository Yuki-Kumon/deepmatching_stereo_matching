# -*- coding: utf-8 -*-

"""
近傍の点のみでマッチングを行うことでステレオマッチングする
Author :
    Yuki Kumon
Last Update :
    2019-10-01
"""

from misc.Correlation_map import Correlation_map
from misc.Matching import Matching
from misc.Calc_difference import Calc_difference
from misc.loader import Loader

from absl import app, flags, logging
from absl.flags import FLAGS

import numpy as np
import cv2


flags.DEFINE_string('original_image_path', './data/band3s.tif', 'image path of original image')
flags.DEFINE_string('template_image_path', './data/band3bs.tif', 'image path of template image')
flags.DEFINE_string('integrated_image_path', './data/after-before-crossdis.tif', 'image path to integrated image')
flags.DEFINE_bool('integrated_input', True, 'integrated images are inputed or not')
flags.DEFINE_string('save_name', './output/result.png', 'save name')
flags.DEFINE_string('origin_save_name', './output/here.png', 'save name of original one')
flags.DEFINE_string('correlation_save_name', './output/correlation.png', 'save name of correlation')
flags.DEFINE_string('GT_save_name', './output/gt.png', 'save name of grand truth')
flags.DEFINE_string('array_save_name', './output/response.npy', 'save name of deepmathing result')
flags.DEFINE_string('feature_name', 'cv2.TM_CCOEFF_NORMED', 'feature name used to calculate feature map')
flags.DEFINE_string('degree_map_mode', 'elevation', 'mode to calculate degree map')
flags.DEFINE_list('image_cut_size', '68, 260', 'image size cut from start point')  # window size が5だと4を引いて2の累乗なら大丈夫
flags.DEFINE_list('image_cut_start', '3500, 1760', 'point to cut image from')


def main(_argv):
    '''
    main function
    '''
    # load images
    if FLAGS.integrated_input:
        loader = Loader(
            FLAGS.integrated_image_path,
            start=[int(x) for x in FLAGS.image_cut_start],
            size=[int(x) for x in FLAGS.image_cut_size],
            integrated=True
        )
    else:
        loader = Loader(
            [FLAGS.original_image_path, FLAGS.template_image_path],
            start=[int(x) for x in FLAGS.image_cut_start],
            SIZE=[int(x) for x in FLAGS.image_cut_size],
            integrated=False
        )

    print(loader())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

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
flags.DEFINE_string('integrated_image_path', './data/after-before-crossdis.tif', 'image path to integrated image')
flags.DEFINE_bool('two_images_input', True, '2 images are inputed or not')
flags.DEFINE_string('save_name', './output/result.png', 'save name')
flags.DEFINE_string('origin_save_name', './output/here.png', 'save name of original one')
flags.DEFINE_string('correlation_save_name', './output/correlation.png', 'save name of correlation')
flags.DEFINE_string('GT_save_name', './output/gt.png', 'save name of grand truth')
flags.DEFINE_string('array_save_name', './output/response.npy', 'save name of deepmathing result')
flags.DEFINE_string('feature_name', 'cv2.TM_CCOEFF_NORMED', 'feature name used to calculate feature map')
flags.DEFINE_string('degree_map_mode', 'elevation', 'mode to calculate degree map')
flags.DEFINE_list('image_cut_size', '68, 260', 'image size cut from start point')  # window size が5だと4を引いて2の累乗なら大丈夫
flags.DEFINE_list('image_cut_start', '100, 100', 'point to cut image from')


def main(_argv):
    # image cut size
    size = [int(x) for x in FLAGS.image_cut_size]
    start = [int(x) for x in FLAGS.image_cut_start]
    # load images
    if not FLAGS.two_images_input:
        img_loaded = cv2.imread(FLAGS.integrated_image_path)
        img1_raw = img_loaded[:, :, 1]  # 地震前
        img2_raw = img_loaded[:, :, 2]  # 地震後
        img3_raw = img_loaded[:, :, 0]  # 変化マップ

        img1 = img1_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]
        img2 = img2_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]
        img3 = img3_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]
    else:
        img1_raw = cv2.imread(FLAGS.original_image_path, cv2.IMREAD_GRAYSCALE)
        img2_raw = cv2.imread(FLAGS.template_image_path, cv2.IMREAD_GRAYSCALE)

        img1 = img1_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]
        img2 = img2_raw[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]
    logging.info('complete to load images')

    # deepmathing
    logging.info('start deepmathing')
    co_cls = Correlation_map(img1, img2, window_size=5, feature_name=FLAGS.feature_name)
    co_cls()

    cls = Matching(co_cls)
    out = cls()

    # d_map = Calc_difference.cal_map(out, mode='elevation')
    d_map = Calc_difference.cal_map(out, mode=FLAGS.degree_map_mode)

    # save
    cv2.imwrite(FLAGS.save_name, d_map * 30 + 100)
    cv2.imwrite(FLAGS.correlation_save_name, out[2, :, :] * 70)
    cv2.imwrite(FLAGS.origin_save_name, img1)
    cv2.imwrite('./output/here2.png', img2)
    np.save(FLAGS.array_save_name, out)
    # cv2.imwrite('./output/co_map.png', co_cls.co_map_list[0][0, 0] * 70)
    if not FLAGS.two_images_input:
        cv2.imwrite(FLAGS.GT_save_name, img3)
    logging.info('complete to save results')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

# -*- coding: utf-8 -*-

"""
execute extended deepmatching with raw file input
Author :
    Yuki Kumon
Last Update :
    2020-01-17
"""


import numpy as np

from absl import app, flags, logging
from absl.flags import FLAGS

from misc.image_cut_solver import ImageCutSolver
from misc.raw_read import RawRead


flags.DEFINE_string('image1_path', './data/newdata/KumonColor/ortho2b.raw', '')
flags.DEFINE_string('image2_path', './data/newdata/KumonColor/ortho2a.raw', '')
flags.DEFINE_list('rates', '1, 1', '')

flags.DEFINE_list('cut_start', '2050, 1600', '')
flags.DEFINE_list('cut_size', '1000, 1000', '')

flags.DEFINE_integer('window_size', 15, '')
flags.DEFINE_list('crop_size', '32, 32', '')
flags.DEFINE_list('stride', '28, 28', '')
flags.DEFINE_bool('sub_pix', False, '')
flags.DEFINE_bool('filtering', False, '')
flags.DEFINE_integer('filtering_window_size', 3, '')
flags.DEFINE_integer('filtering_num', 3, '')


def main(_argv):
    # set integer list
    rates = [int(x) for x in FLAGS.rates]
    cut_start = [int(x) for x in FLAGS.cut_start]
    cut_size = [int(x) for x in FLAGS.cut_size]

    crop_size = [int(x) for x in FLAGS.crop_size]
    stride = [int(x) for x in FLAGS.stride]

    DEGREE_NAMES = ['elevation', 'elevation2']

    # read image
    image1 = RawRead.read(FLAGS.image1_path, rate=rates[0])
    image2 = RawRead.read(FLAGS.image2_path, rate=rates[1])

    # crop images
    image1 = image1[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1]]
    image2 = image2[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1]]

    # print raw image
    ImageCutSolver.image_save('./output/igarss/raw/here.png', image1, threshold=[0, 255])
    ImageCutSolver.image_save('./output/igarss/raw/here2.png', image2, threshold=[0, 255])

    # execute deepmatching
    solver = ImageCutSolver(
        image1, image2,
        degree_map_mode=DEGREE_NAMES,
        window_size=FLAGS.window_size,
        image_size=crop_size,
        stride=stride,
        sub_pix=FLAGS.sub_pix,
        filtering=FLAGS.filtering,
        filtering_window_size=FLAGS.filtering_window_size,
        filtering_num=FLAGS.filtering_num
    )

    res_list, correlation_map = solver()

    # save computation result as numpy array
    for i, name in enumerate(DEGREE_NAMES):
        np.save('./output/igarss/raw/' + name, res_list[i])
    np.save('./output/igarss/raw/' + 'correlation', correlation_map)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

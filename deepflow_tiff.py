import argparse
import os
import cv2
import numpy as np

# from misc.raw_read import RawRead
# from misc.file_util import FileUtil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--image', default='/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Ethiopia(after-before-X).tif')
    # parser.add_argument('--output', default='./output/extra/ethiopia_deepflow')
    parser.add_argument('--image', default='/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Kunlun(after-before-X).tif')
    parser.add_argument('--output', default='./output/extra/kunlun_deepflow')
    parser.add_argument('--cut_start', nargs='+', default=[2800, 2100], type=int)
    parser.add_argument('--cut_size', nargs='+', default=[1000, 1000], type=int)

    args = parser.parse_args()

    # FileUtil.mkdir(args.output)
    path = args.output
    if not os.path.exists(path):
        os.makedirs(path)

    cut_start = args.cut_start
    cut_size = args.cut_size

    images = cv2.imread(args.image)
    images = images[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1], 1:]

    # 2枚の画像でdeepflow
    flow = np.empty_like(images[:, :, 0])
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(images[:, :, 0], images[:, :, 1], flow)
    print(flow.shape)

    # 画像にして出力
    for idx in range(2):
        cv2.imwrite(os.path.join(args.output, 'flow_' + str(idx) + '.png'), (flow[:, :, idx] * -100).astype('uint8'))

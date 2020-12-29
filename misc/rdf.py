# -*- coding: utf-8 -*-

"""
同径分布関数(のようなもの)
Author :
    Yuki Kumon
Last Update :
    2020-12-28
"""


import numpy as np
from scipy import ndimage as ndi
from sklearn.neighbors import NearestNeighbors


class RDF:

    def __init__(self, max_filter_size=9, n_neighbors=10, th=5):
        self.max_filter_size = max_filter_size
        self.n_neighbors = n_neighbors
        self.th = th

    def _maxFilter(self, image):
        return ndi.maximum_filter(image, size=self.max_filter_size, mode='constant')

    def _computeParticle(self, image, max_image):
        # コロイド位置の検出
        pts_list = np.where(image1 == image1_max)
        pts_list = np.array(pts_list)
        return pts_list

    def _removeDuplicates(self, pts_list):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(pts_list.T)
        distances, indices = nbrs.kneighbors(pts_list.T)
        # 閾値以下の近接点との重心を計算。
        center_list = []
        for d, i in zip(distances, indices):
            i = i[np.where(d < self.th)]
            pts = pts_list.T[i]
            center = np.mean(pts, axis=0)
            center_list.append(center)
        center_list = np.array(center_list).astype(np.int32)

        # 重複を削除
        center_list = np.unique(center_list, axis=0)
        center_list = center_list.T

        return center_list


if __name__ == '__main__':
    """
    sanity check
    """
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import sys
    sys.path.append('.')

    from misc.raw_read import RawRead
    # テスト用画像の読み込み
    cut_start = [2050, 1600]
    cut_size = [1000, 1000]
    image1 = RawRead.read('./data/newdata/KumonColor/ortho2b.raw')
    image1 = image1[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1]]

    cls = RDF()

    # 最大化フィルター
    image1_max = cls._maxFilter(image1)
    # コロイド位置の検出
    """
    pts_list = np.where(image1 == image1_max)
    pts_list = np.array(pts_list)
    fig = plt.figure(dpi=150, facecolor='white')
    """
    pts_list = cls._computeParticle(image1, image1_max)
    print(len(pts_list[0]))
    pts_list = cls._removeDuplicates(pts_list)
    print(len(pts_list[0]))
    plt.gray()
    plt.imshow(image1)
    plt.scatter(pts_list[1], pts_list[0], s=3, c='red', alpha=0.7)
    plt.savefig('./test.png')
    # cv2.imwrite('./hoge.png', image1_max)

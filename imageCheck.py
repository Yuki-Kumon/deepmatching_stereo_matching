import cv2

file = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Ethiopia(after-before-X).tif'
cut_start = [2800, 2100]
# file = '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Kunlun(after-before-X).tif'
# cut_start = [1100, 2200]
cut_size = [1000, 1000]

image = cv2.imread(file)

for idx in range(3):
    cv2.imwrite('./test_' + str(idx) + '.png', image[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1], idx])
    # cv2.imwrite('./test_' + str(idx) + '.png', image[:, :, idx])

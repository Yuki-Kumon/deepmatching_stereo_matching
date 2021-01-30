# deepmatching
python ex_deepmatching_tiffinput_optim.py --crop_size 64,64 --stride 60,60 --window_size 15 --optim_mode 0 --cut_start 2800,2100 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Ethiopia(after-before-X).tif' --output_root './output/extra/ethiopia_s=15_g=6'
python ex_deepmatching_tiffinput_optim.py --crop_size 32,32 --stride 30,30 --window_size 15 --optim_mode 0 --cut_start 2800,2100 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Ethiopia(after-before-X).tif' --output_root './output/extra/ethiopia_s=15_g=5'
python ex_deepmatching_tiffinput_optim.py --crop_size 64,64 --stride 60,60 --window_size 9 --optim_mode 0 --cut_start 2800,2100 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Ethiopia(after-before-X).tif' --output_root './output/extra/ethiopia_s=9_g=6'
python ex_deepmatching_tiffinput_optim.py --crop_size 64,64 --stride 60,60 --window_size 15 --optim_mode 2 --cut_start 2800,2100 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Ethiopia(after-before-X).tif' --output_root './output/extra/ethiopia_s=15_g=6_sub'

# visualize
python convert_to_image.py --array_path './output/extra/ethiopia_s=15_g=6/elevation.npy','./output/extra/ethiopia_s=15_g=6/elevation2.npy' --output_path './output/extra/ethiopia_s=15_g=6'
python convert_to_image.py --array_path './output/extra/ethiopia_s=15_g=5/elevation.npy','./output/extra/ethiopia_s=15_g=5/elevation2.npy' --output_path './output/extra/ethiopia_s=15_g=5'
python convert_to_image.py --array_path './output/extra/ethiopia_s=9_g=6/elevation.npy','./output/extra/ethiopia_s=9_g=6/elevation2.npy' --output_path './output/extra/ethiopia_s=9_g=6'
python convert_to_image.py --array_path './output/extra/ethiopia_s=15_g=6_sub/elevation.npy','./output/extra/ethiopia_s=15_g=6_sub/elevation2.npy' --output_path './output/extra/ethiopia_s=15_g=6_sub'

# deepmatching
python ex_deepmatching_tiffinput_optim.py --crop_size 64,64 --stride 60,60 --window_size 15 --optim_mode 0 --cut_start 1100,2200 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Kunlun(after-before-X).tif' --output_root './output/extra/kunlun_s=15_g=6'
python ex_deepmatching_tiffinput_optim.py --crop_size 32,32 --stride 30,30 --window_size 15 --optim_mode 0 --cut_start 1100,2200 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Kunlun(after-before-X).tif' --output_root './output/extra/kunlun_s=15_g=5'
python ex_deepmatching_tiffinput_optim.py --crop_size 64,64 --stride 60,60 --window_size 9 --optim_mode 0 --cut_start 1100,2200 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Kunlun(after-before-X).tif' --output_root './output/extra/kunlun_s=9_g=6'
python ex_deepmatching_tiffinput_optim.py --crop_size 64,64 --stride 60,60 --window_size 15 --optim_mode 2 --cut_start 1100,2200 --image '/Users/yuki_kumon/Documents/python/deepmatching_stereo_matching/data/other_scene/Kunlun(after-before-X).tif' --output_root './output/extra/kunlun_s=15_g=6_sub'


# visualize
python convert_to_image.py --array_path './output/extra/kunlun_s=15_g=6/elevation.npy','./output/extra/kunlun_s=15_g=6/elevation2.npy' --output_path './output/extra/kunlun_s=15_g=6'
python convert_to_image.py --array_path './output/extra/kunlun_s=15_g=5/elevation.npy','./output/extra/kunlun_s=15_g=5/elevation2.npy' --output_path './output/extra/kunlun_s=15_g=5'
python convert_to_image.py --array_path './output/extra/kunlun_s=9_g=6/elevation.npy','./output/extra/kunlun_s=9_g=6/elevation2.npy' --output_path './output/extra/kunlun_s=9_g=6'
python convert_to_image.py --array_path './output/extra/kunlun_s=15_g=6_sub/elevation.npy','./output/extra/kunlun_s=15_g=6_sub/elevation2.npy' --output_path './output/extra/kunlun_s=15_g=6_sub'

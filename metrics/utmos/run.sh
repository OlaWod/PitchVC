mkdir TEST_
cd TEST_

ln -s ../../../outputs/s2s s2s
ln -s ../../../outputs/u2s u2s
ln -s ../../../outputs/esd_en esd_en
ln -s ../../../outputs/esd_zh esd_zh

cd ..


CUDA_VISIBLE_DEVICES=0 python utmos.py --wavdir TEST_/s2s --title s2s
CUDA_VISIBLE_DEVICES=0 python utmos.py --wavdir TEST_/u2s --title u2s
CUDA_VISIBLE_DEVICES=0 python utmos.py --wavdir TEST_/esd_en --title esd_en
CUDA_VISIBLE_DEVICES=0 python utmos.py --wavdir TEST_/esd_zh --title esd_zh

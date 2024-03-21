mkdir TEST_
cd TEST_

ln -s ../../../test/TEST_SRC_VCTK src_vctk
ln -s ../../../test/TEST_SRC_LIBRI src_libri
ln -s ../../../test/TEST_SRC_ESD_EN src_esd_en
ln -s ../../../test/TEST_SRC_ESD_ZH src_esd_zh

ln -s ../../../outputs/s2s s2s
ln -s ../../../outputs/u2s u2s
ln -s ../../../outputs/esd_en esd_en
ln -s ../../../outputs/esd_zh esd_zh

cd ..


CUDA_VISIBLE_DEVICES=0 python get_gt.py --wavdir TEST_/src_vctk --title src_vctk
CUDA_VISIBLE_DEVICES=0 python get_gt.py --wavdir TEST_/src_libri --title src_libri
CUDA_VISIBLE_DEVICES=0 python get_gt.py --wavdir TEST_/src_esd_en --title src_esd_en
CUDA_VISIBLE_DEVICES=0 python get_gt.py --wavdir TEST_/src_esd_zh --title src_esd_zh --lang chinese


CUDA_VISIBLE_DEVICES=0 python asr_sp.py --wavdir TEST_/s2s --title s2s --gt src_vctk
CUDA_VISIBLE_DEVICES=0 python asr.py --wavdir TEST_/s2s --title s2s --gt src_vctk
CUDA_VISIBLE_DEVICES=0 python asr.py --wavdir TEST_/u2s --title u2s --gt src_libri
CUDA_VISIBLE_DEVICES=0 python asr.py --wavdir TEST_/esd_en --title esd_en --gt src_esd_en
CUDA_VISIBLE_DEVICES=0 python asr.py --wavdir TEST_/esd_zh --title esd_zh --gt src_esd_zh --lang chinese

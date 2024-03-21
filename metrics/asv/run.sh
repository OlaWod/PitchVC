mkdir TEST_
cd TEST_

ln -s ../../../test/TEST_TGT tgt

ln -s ../../../outputs/s2s s2s
ln -s ../../../outputs/u2s u2s
ln -s ../../../outputs/esd_en esd_en
ln -s ../../../outputs/esd_zh esd_zh

cd ..

python get_txt.py --wav_dir TEST_/s2s --title s2s
python get_txt.py --wav_dir TEST_/u2s --title u2s
python get_txt.py --wav_dir TEST_/esd_en --title esd_en
python get_txt.py --wav_dir TEST_/esd_zh --title esd_zh


CUDA_VISIBLE_DEVICES=0 python ssim_sp.py --txtpath txts/s2s.txt --title s2s

CUDA_VISIBLE_DEVICES=0 python ssim.py --txtpath txts/s2s.txt --title s2s
CUDA_VISIBLE_DEVICES=0 python ssim.py --txtpath txts/u2s.txt --title u2s
CUDA_VISIBLE_DEVICES=0 python ssim.py --txtpath txts/esd_en.txt --title esd_en
CUDA_VISIBLE_DEVICES=0 python ssim.py --txtpath txts/esd_zh.txt --title esd_zh

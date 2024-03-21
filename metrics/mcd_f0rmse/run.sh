mkdir TEST_
cd TEST_

ln -s ../../../dataset/vctk-16k VCTK
ln -s ../../../outputs/s2s MODEL1

cd ..

python get_map.py --txt_root "/home/Datasets/lijingyi/data/vctk/txt"
python get_map2.py
python norm.py

python get_valid.py --wav_dir TEST_/MODEL1
python get_txt.py --wav_dir TEST_/MODEL1 --title model1

python mcd.py --txtpath txts/model1.txt --title mcd_model1
python f0rmse.py --txtpath txts/model1.txt --title f0rmse_model1

mv rtf.py ../..
cd ../..

CUDA_VISIBLE_DEVICES=-1 python rtf.py --hpfile config_v1_16k.json --ptfile exp/default/g_00700000 --txtpath test/txts/s2s.txt 

mv rtf.py metrics/rtf
cd metrics/rtf
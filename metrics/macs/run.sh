mv macs.py ../..
cd ../..

CUDA_VISIBLE_DEVICES=0 python macs.py --hpfile config_v1_16k.json --txtpath test/txts/u2s.txt 

mv macs.py metrics/macs
cd metrics/macs
# PitchVC: Pitch Conditioned Any-to-Many Voice Conversion

![GitHub](https://img.shields.io/github/license/OlaWod/PitchVC)

[🎧 Audio Samples](https://olawod.github.io/PitchVC-demo/).
$\quad\quad$
[🤗 Play Online](https://huggingface.co/spaces/OlaWod/PitchVC).

## Description
A simple VC framework.

<table style="width:100%">
  <tr>
    <td><img src="./resources/train.png" alt="training" height="200"></td>
    <td><img src="./resources/infer.png" alt="inference" height="200"></td>
  </tr>
  <tr>
    <th>(a) Training</th>
    <th>(b) Inference</th>
  </tr>
  <tr>
    <td><img src="./resources/train-o.png" alt="training-o" height="200"></td>
    <td><img src="./resources/infer-o.png" alt="inference-o" height="200"></td>
  </tr>
  <tr>
    <th>(c) Training (w/ optional properties)</th>
    <th>(d) Inference (w/ optional properties)</th>
  </tr>
</table>

[Detailed description](Description.md).

## Pre-requisites
1. Clone this repo: `git clone https://github.com/OlaWod/PitchVC.git`
2. CD into this repo: `cd PitchVC`
3. Install python requirements: `pip install -r requirements.txt`
4. Download files on demand (e.g. pretrained checkpoint) ([download link](https://1drv.ms/f/s!AnvukVnlQ3ZTmK9VGwAn4ES00GS96w?e=4A9xuz))

## Inference Example
Files on demand:
1. Pretrained checkpoint (e.g. `exp/default/g_00700000`)
2. Source wavs (e.g. `src1.wav`) and target wavs&embs (e.g. `p244_008.wav`&`p244_008.npy`) in `convert.txt`
3. `Utils/JDC/bst.t7`
4. (Optional) `speakerlab/pretrained/speech_eres2net_sv_en_voxceleb_16k/pretrained_eres2net.ckpt` and `speakerlab/pretrained/speech_eres2net_sv_zh-cn_16k-common/pretrained_eres2net_aug.ckpt`

```bash
# single process
CUDA_VISIBLE_DEVICES=0 python convert_sp.py --hpfile config_v1_16k.json --ptfile exp/default/g_00700000 --txtpath convert.txt --outdir outputs/test

# single process; finetune input f0 automatically
CUDA_VISIBLE_DEVICES=0 python convert_sp.py --hpfile config_v1_16k.json --ptfile exp/default/g_00700000 --txtpath convert.txt --outdir outputs/test --search

# multi process
CUDA_VISIBLE_DEVICES=0 python convert_mp.py --hpfile config_v1_16k.json --ptfile exp/default/g_00700000 --txtpath convert.txt --outdir outputs/test --n_processes 6

# multi process; finetune input f0 automatically
CUDA_VISIBLE_DEVICES=0 python convert_mp.py --hpfile config_v1_16k.json --ptfile exp/default/g_00700000 --txtpath convert.txt --outdir outputs/test --n_processes 6 --search
```

## Training Example
Files on demand:
1. [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset
2. `speaker_encoder/ckpt/pretrained_bak_5805000.pt`
3. `Utils/JDC/bst.t7`

Preprocess:
```bash
export PYTHONPATH=.

python preprocess/1_downsample.py --in_dir </path/to/VCTK/wavs> # dataset/vctk-16k/{spk}/{xx}.wav
python preprocess/2_get_flist.py    # filelists/{situation}.txt
python preprocess/3_get_spk2id.py   # filelists/spk2id.json
python preprocess/4_get_spk_emb.py  # dataset/spk/{spk}/{xx}.npy
python preprocess/5_get_spk_emb_best.py # filelists/spk_stats.json
python preprocess/6_get_f0.py       # dataset/f0/{spk}/{xx}.pt
python preprocess/7_get_f0_stats.py # filelists/f0_stats.json

cd dataset
ln -s vctk-16k audio
cd ..
```

Training:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config_v1_16k.json --checkpoint_path exp/test
```

## Test Example
```bash
python test/1_select_tgt.py # test/TEST_TGT/{xx}.wav
python test/2_select_src.py # test/TEST_SRC_{CORPUS}/{xx}.wav
python test/3_get_txts.py   # test/txts/{scenario}.txt

CUDA_VISIBLE_DEVICES=0 python convert_mp.py --hpfile config_v1_16k.json --ptfile exp/default/g_00700000 --txtpath test/txts/<scenario>.txt --outdir outputs/<scenario> --n_processes 6 --search

cd metrics/<metrics>
bash run.sh
```

## References
- https://github.com/yl4579/HiFTNet
- https://github.com/jaywalnut310/vits
- https://github.com/liusongxiang/ppg-vc
- https://github.com/alibaba-damo-academy/3D-Speaker
- https://github.com/open-mmlab/Amphion

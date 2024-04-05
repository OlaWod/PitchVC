# OpenVINO

### Installation

[https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)

### Set Environment

```bash
cd /path/to/PitchVC
export PYTHONPATH=.

source /opt/intel/openvino_2024/setupvars.sh
```

### Convert PyTorch model to OpenVINO IR

```bash
CUDA_VISIBLE_DEVICES=-1 python openvino_/export_g1.py
CUDA_VISIBLE_DEVICES=-1 python openvino_/export_g2.py
```

### Infer with OpenVINO Python Runtime

```bash
CUDA_VISIBLE_DEVICES=-1 python openvino_/convert.py --hpfile config_v1_16k.json --txtpath test/txts/s2s.txt --outdir outputs/test_vino
```

# ONNX

### Installation

```bash
pip install onnx onnxruntime
```

### Set PYTHONPATH

```bash
cd /path/to/PitchVC
export PYTHONPATH=.
```

### Convert PyTorch model to ONNX format

```bash
CUDA_VISIBLE_DEVICES=-1 python onnx_/export_g1.py
CUDA_VISIBLE_DEVICES=-1 python onnx_/export_g2.py
```

### Infer with ONNX Python Runtime

```bash
CUDA_VISIBLE_DEVICES=-1 python onnx_/convert.py --hpfile config_v1_16k.json --txtpath test/txts/s2s.txt --outdir outputs/test_onnxrt
```

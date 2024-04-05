# OpenVINO Runtime C++ Example

### Installation

1. Install [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html).

2. Install [LibTorch](https://pytorch.org/get-started/locally/) with cxx11 ABI.

(To pick a suitable version of LibTorch for your environment, look [here](https://download.pytorch.org/libtorch/cpu).)

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip

unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
# obtain the libtorch directory
```

### Set Environment

1.

```bash
cd /path/to/PitchVC/openvino_/cpp
source /opt/intel/openvino_2024/setupvars.sh

# modify the path to libtorch directory in build.sh
bash build.sh
# obtain the executable file bin/app
```

(Note: Other platforms such as Windows and MacOS are not considered, modification to `openvino_/cpp/CMakeLists.txt`, `openvino_/cpp/src/CMakeLists.txt`, etc. might be necessary.)

2.

Make sure `assets/dataset/` has directory `audio/` and `spk/`, and contains the needed files. (`ln -s /path/to/PitchVC/dataset assets/dataset`)

Make sure `assets/json/` has the following files: `f0_stats.json`, `spk_stats.json`, `spk2id.json`. (`ln -s /path/to/PitchVC/filelists assets/jsons`)

Make sure `assets/model/` has the following files: `g1.bin`, `g1.xml`, `g2.bin`, `g2.xml`. (`ln -s /path/to/PitchVC/exp assets/models`)

### Infer with OpenVINO C++ Runtime

```bash
# modify assets/infer.txt
bash infer.sh
```

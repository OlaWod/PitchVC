import json

import torch
from transformers import WavLMModel
import openvino as ov

from env import AttrDict
from onnx_.models_onnx import Generator1
from Utils.JDC.model import JDCNet


hp_path = "config_v1_16k.json"
pt_path = "exp/default/g_00700000"
f0_path = "Utils/JDC/bst.t7"
save_path = "dump/g1.xml"


# global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load config
with open(hp_path) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

# load models
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")

f0_model = JDCNet(num_class=1, seq_len=192).to(device)
params = torch.load(f0_path, map_location=device)
f0_model.load_state_dict(params['model'])
f0_model.eval()

model = Generator1(h, wavlm, f0_model).to(device)

state_dict_g = torch.load(pt_path, map_location=device)
model.load_state_dict(state_dict_g['generator'], strict=False)
model.eval()

# Input to the model
B = 1
T = 70
T_wav = T * 320

wav = torch.randn(B, T_wav).to(device)
spk_emb = torch.randn(B, 256).to(device)
spk_id = torch.randint(0, 107, (B, 1)).to(device)
mel = torch.randn(B, 80, T).to(device)
f0_mean_tgt = torch.FloatTensor([100]).to(device)
f0_mean_tgt = f0_mean_tgt.unsqueeze(1)
inputs = (wav, mel, spk_emb, spk_id, f0_mean_tgt)

with torch.no_grad():
    x, har_source = model(wav, mel, spk_emb, spk_id, f0_mean_tgt)

# Create OpenVINO Core object instance
core = ov.Core()

# Convert model to openvino.runtime.Model object
ov_model = ov.convert_model(model, example_input=inputs)

# Save openvino.runtime.Model object on disk
ov.save_model(ov_model, save_path)

compiled_model = core.compile_model(ov_model, "CPU")

# Run model inference
out = compiled_model([wav, mel, spk_emb, spk_id, f0_mean_tgt])
x_ = out[compiled_model.output(0)]
har_source_ = out[compiled_model.output(1)]
print(x, x_, x.shape, x_.shape)
print(har_source, har_source_, har_source.shape, har_source_.shape)

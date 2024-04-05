import json

import torch
import openvino as ov

from env import AttrDict
from onnx_.models_onnx import Generator2


hp_path = "config_v1_16k.json"
pt_path = "exp/default/g_00700000"
save_path = "exp/g2.xml"


# global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load config
with open(hp_path) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

# load models
model = Generator2(h).to(device)

state_dict_g = torch.load(pt_path, map_location=device)
model.load_state_dict(state_dict_g['generator'], strict=False)
model.remove_weight_norm()
model.eval()

# Input to the model
B = 1
T = 70

x = torch.randn(B, 512, T).to(device)
har_spec = torch.randn(B, 9, T * 80 + 1).to(device)
har_phase = torch.randn(B, 9, T * 80 + 1).to(device)
inputs = (x, har_spec, har_phase)

with torch.no_grad():
    spec, phase = model(x, har_spec, har_phase)

# Create OpenVINO Core object instance
core = ov.Core()

# Convert model to openvino.runtime.Model object
ov_model = ov.convert_model(model, example_input=inputs)

# Save openvino.runtime.Model object on disk
ov.save_model(ov_model, save_path)

compiled_model = core.compile_model(ov_model, "CPU")

# Run model inference
out = compiled_model([x, har_spec, har_phase])
spec_ = out[compiled_model.output(0)]
phase_ = out[compiled_model.output(1)]
print(spec, spec_, spec.shape, spec_.shape)
print(phase, phase_, phase.shape, phase_.shape)

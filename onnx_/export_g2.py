import json

import torch

from env import AttrDict
from onnx_.models_onnx import Generator2


hp_path = "config_v1_16k.json"
pt_path = "exp/default/g_00700000"
save_path = "exp/g2.onnx"


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

with torch.no_grad():
    spec, phase = model(x, har_spec, har_phase)

input_names = ["x", "har_spec", "har_phase"]
output_names = ["spec", "phase"]
dynamic_axes = {
    "x": {0: "batch", 2: "time"},
    "har_spec": {0: "batch", 2: "time"},
    "har_phase": {0: "batch", 2: "time"},
    "spec": {0: "batch", 2: "time"},
    "phase": {0: "batch", 2: "time"},
}
inputs = (x, har_spec, har_phase)

# Export the model
torch.onnx.export(
    model,
    inputs,
    save_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes
)

              
# Check
# Verify the model’s structure and confirm that the model has a valid schema.
# The validity of the ONNX graph is verified by checking the model’s version, 
# the graph’s structure, as well as the nodes and their inputs and outputs.
import onnx

onnx_model = onnx.load(save_path)
onnx.checker.check_model(onnx_model)


# Check
# Verify that ONNX Runtime and PyTorch are computing the same value for the network.
import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession(save_path, providers=["CPUExecutionProvider"])

# compute ONNX Runtime output prediction
ort_inputs = {
            ort_session.get_inputs()[0].name: x.cpu().numpy(),
            ort_session.get_inputs()[1].name: har_spec.cpu().numpy(),
            ort_session.get_inputs()[2].name: har_phase.cpu().numpy(),
            }
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(spec.cpu().numpy(), ort_outs[0], rtol=1e-02, atol=1e-04)
np.testing.assert_allclose(phase.cpu().numpy(), ort_outs[1], rtol=1e-02, atol=1e-04)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# Test dynamic axes
B = 2
T = 80

x = torch.randn(B, 512, T).to(device)
har_spec = torch.randn(B, 9, T * 80 + 1).to(device)
har_phase = torch.randn(B, 9, T * 80 + 1).to(device)

ort_inputs = {
            ort_session.get_inputs()[0].name: x.cpu().numpy(),
            ort_session.get_inputs()[1].name: har_spec.cpu().numpy(),
            ort_session.get_inputs()[2].name: har_phase.cpu().numpy(),
            }
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0].shape, ort_outs[1].shape, x.shape, har_spec.shape, har_phase.shape)
import json

import torch
from transformers import WavLMModel

from env import AttrDict
from onnx_.models_onnx import Generator1
from Utils.JDC.model import JDCNet


hp_path = "config_v1_16k.json"
pt_path = "exp/default/g_00700000"
f0_path = "Utils/JDC/bst.t7"
save_path = "exp/g1.onnx"


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

with torch.no_grad():
    x, har_source = model(wav, mel, spk_emb, spk_id, f0_mean_tgt)

input_names = ["wav", "mel", "spk_emb", "spk_id", "f0_mean_tgt"]
output_names = ["x", "har_source"]
# dynamic_axes = {
#     "wav": {0: "batch", 1: "time"}, 
#     "spk_emb": {0: "batch"}, 
#     "spk_id": {0: "batch"}, 
#     "mel": {0: "batch", 2: "time"},
#     "f0_mean_tgt": {0: "batch"},
#     "x": {0: "batch", 2: "time"},
#     "har_source": {0: "batch", 1: "time"},
# }
dynamic_axes = {
    "wav": {1: "time"}, 
    "mel": {2: "time"},
    "x": {2: "time"},
    "har_source": {1: "time"},
}
inputs = (wav, mel, spk_emb, spk_id, f0_mean_tgt)

# Export the model
torch.onnx.export(
    model,
    inputs,
    save_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
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
            ort_session.get_inputs()[0].name: wav.cpu().numpy(),
            ort_session.get_inputs()[1].name: mel.cpu().numpy(),
            ort_session.get_inputs()[2].name: spk_emb.cpu().numpy(),
            ort_session.get_inputs()[3].name: spk_id.cpu().numpy(),
            ort_session.get_inputs()[4].name: f0_mean_tgt.cpu().numpy(),
            }
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(x.cpu().numpy(), ort_outs[0], rtol=1e-02, atol=1e-04)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# Test dynamic axes
B = 1
T = 80
T_wav = T * 320

wav = torch.randn(B, T_wav).to(device)
spk_emb = torch.randn(B, 256).to(device)
spk_id = torch.randint(0, 107, (B, 1)).to(device)
mel = torch.randn(B, 80, T).to(device)
f0_mean_tgt = torch.FloatTensor([100]).to(device)
f0_mean_tgt = f0_mean_tgt.unsqueeze(1)

ort_inputs = {
            ort_session.get_inputs()[0].name: wav.cpu().numpy(),
            ort_session.get_inputs()[1].name: mel.cpu().numpy(),
            ort_session.get_inputs()[2].name: spk_emb.cpu().numpy(),
            ort_session.get_inputs()[3].name: spk_id.cpu().numpy(),
            ort_session.get_inputs()[4].name: f0_mean_tgt.cpu().numpy(),
            }
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0].shape, wav.shape)
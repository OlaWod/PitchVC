import json


with open("filelists/test.txt") as f:
    lines = f.readlines()
with open("filelists/val.txt") as f:
    lines += f.readlines()
with open("filelists/train.txt") as f:
    lines += f.readlines()

spks = set()
for line in lines:
    spk = line.strip().split("|")[0]
    spks.add(spk)

spks = list(spks)
spks.sort()

spk2id = {}
for i, spk in enumerate(spks):
    spk2id[spk] = i + 1

with open("filelists/spk2id.json", "w") as f:
    json.dump(spk2id, f, indent=2)
    
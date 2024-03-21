import os
import json
import collections

from glob import glob


with open(f"txts/txt_map.json", "r") as f:
    txt_map = json.load(f)

txt_map2 = collections.defaultdict(dict)

for k, v in txt_map.items():
    spk = k.split("_")[0]
    txt_map2[v][spk] = k

with open(f"txts/txt_map2.json", "w") as f:
    json.dump(txt_map2, f, indent=4)
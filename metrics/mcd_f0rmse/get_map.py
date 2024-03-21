import os
import json
import argparse

from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_root", type=str, default="/home/Datasets/lijingyi/data/vctk/txt", help="path to txt dir")
    args = parser.parse_args()

    txt_map = {}
    for txt in glob(f"{args.txt_root}/*/*.txt"):
        basename = txt.split("/")[-1].split(".")[0]
        with open(txt, "r") as f:
            line = f.readlines()[0].strip()
        txt_map[basename] = line

    os.makedirs("txts", exist_ok=True)
    with open(f"txts/txt_map.json", "w") as f:
        json.dump(txt_map, f, indent=4)
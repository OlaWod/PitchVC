import os

from glob import glob


tgts = glob("test/TEST_TGT/*.wav")
tgts.sort()
emb_dir = "dataset/spk"
os.makedirs("test/txts", exist_ok=True)

# s2s
srcs = glob("test/TEST_SRC_VCTK/*.wav")
srcs.sort()

with open("test/txts/s2s.txt", "w") as f:
    for src in srcs:
        for tgt in tgts:
            title = f"{src.split('/')[-1][:-4]}-{tgt.split('/')[-1][:-4]}"
            tgt_spk = tgt.split("/")[-1].split("_")[1]
            tgt_basename = tgt.split("/")[-1][2:-4]
            tgt_emb = f"{emb_dir}/{tgt_spk}/{tgt_basename}.npy"
            f.write(f"{title}|{src}|{tgt}|{tgt_spk}|{tgt_emb}\n")

# u2s
srcs = glob("test/TEST_SRC_LIBRI/*.wav")
srcs.sort()

with open("test/txts/u2s.txt", "w") as f:
    for src in srcs:
        for tgt in tgts:
            title = f"{src.split('/')[-1][:-4]}-{tgt.split('/')[-1][:-4]}"
            tgt_spk = tgt.split("/")[-1].split("_")[1]
            tgt_basename = tgt.split("/")[-1][2:-4]
            tgt_emb = f"{emb_dir}/{tgt_spk}/{tgt_basename}.npy"
            f.write(f"{title}|{src}|{tgt}|{tgt_spk}|{tgt_emb}\n")

# esd_en
srcs = glob("test/TEST_SRC_ESD_EN/*.wav")
srcs.sort()

with open("test/txts/esd_en.txt", "w") as f:
    for src in srcs:
        for tgt in tgts:
            title = f"{src.split('/')[-1][:-4]}-{tgt.split('/')[-1][:-4]}"
            tgt_spk = tgt.split("/")[-1].split("_")[1]
            tgt_basename = tgt.split("/")[-1][2:-4]
            tgt_emb = f"{emb_dir}/{tgt_spk}/{tgt_basename}.npy"
            f.write(f"{title}|{src}|{tgt}|{tgt_spk}|{tgt_emb}\n")

# esd_zh
srcs = glob("test/TEST_SRC_ESD_ZH/*.wav")
srcs.sort()

with open("test/txts/esd_zh.txt", "w") as f:
    for src in srcs:
        for tgt in tgts:
            title = f"{src.split('/')[-1][:-4]}-{tgt.split('/')[-1][:-4]}"
            tgt_spk = tgt.split("/")[-1].split("_")[1]
            tgt_basename = tgt.split("/")[-1][2:-4]
            tgt_emb = f"{emb_dir}/{tgt_spk}/{tgt_basename}.npy"
            f.write(f"{title}|{src}|{tgt}|{tgt_spk}|{tgt_emb}\n")
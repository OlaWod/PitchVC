from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import argparse
import torch
import librosa
from tqdm import tqdm
from glob import glob
from torchmetrics.text import CharErrorRate
from torchmetrics.text import WordErrorRate


CER = CharErrorRate()
WER = WordErrorRate()


def norm(text, chinese=False):
    if chinese:
        text = text.replace(" ", "")
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace("-", "")
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.lower()
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default="PROPOSED")
    parser.add_argument("--title", type=str, default="1")
    parser.add_argument("--gt", type=str, default="src_vctk")
    parser.add_argument("--lang", type=str, default="english")
    args = parser.parse_args()
    
    # load model and processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").cuda()
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")

    # gt
    gt_dict = {}
    with open(f"result/{args.gt}.txt", "r") as f:
        for line in f.readlines():
            title, text = line.strip().split("|")
            gt_dict[title] = text
    
    # get transcriptions
    wavs = glob(f'{args.wavdir}/*.wav')
    wavs.sort()
    trans_dict = {}
    
    with open(f"result/{args.title}.txt", "w") as f:
        for path in tqdm(wavs):
            wav, sr = librosa.load(path, sr=16000)
            input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.cuda() # text
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            title = os.path.basename(path)[:-4]
            title = title.replace("F_", "")
            title = title.replace("M_", "")
            f.write(f"{title}|{text}\n")
            trans_dict[title] = text
    
    # calc
    wers, cers = {}, {}
    for key in trans_dict.keys():
        text = trans_dict[key]
        gttext = gt_dict[key.split("-")[0]]
        text = norm(text, args.lang=="chinese")
        gttext = norm(gttext, args.lang=="chinese")
        wer = WER(text, gttext).numpy().tolist()
        cer = CER(text, gttext).numpy().tolist()
        wers[key] = min(wer, 1)
        cers[key] = min(cer, 1)
    
    wer = sum(wers.values()) / len(wers)
    cer = sum(cers.values()) / len(cers)
    with open(f"result/{args.title}-cer.txt", "w") as f:
        f.write(f"wer: {wer}\n")
        f.write(f"cer: {cer}\n")
    print("WER:", wer)
    print("CER:", cer)
    
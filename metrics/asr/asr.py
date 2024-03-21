from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import argparse
import shutil
import torch
import librosa
from tqdm import tqdm
from glob import glob
from torchmetrics.text import CharErrorRate
from torchmetrics.text import WordErrorRate
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp


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


def process_one(path, model, processor, forced_decoder_ids, title):
    wav, sr = librosa.load(path, sr=16000)
    input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.cuda() # text
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    basename = os.path.basename(path)[:-4]
    basename = basename.replace("F_", "")
    basename = basename.replace("M_", "")
    
    with open(f"tmp/{title}+{basename}.txt", "w") as f:
        f.write(text)


def process_batch(batch, title, lang):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model and processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").cuda()
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")

    # process
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0

    for line in tqdm(batch, position=rank):
        process_one(line, model, processor, forced_decoder_ids, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default="PROPOSED")
    parser.add_argument("--title", type=str, default="1")
    parser.add_argument("--gt", type=str, default="src_vctk")
    parser.add_argument("--lang", type=str, default="english")
    parser.add_argument("--n_processes", type=int, default=8, help="number of multiprocessing processes")
    args = parser.parse_args()
    
    # gt
    gt_dict = {}
    with open(f"result/{args.gt}.txt", "r") as f:
        for line in f.readlines():
            title, text = line.strip().split("|")
            gt_dict[title] = text
    
    # get transcriptions
    wavs = glob(f'{args.wavdir}/*.wav')
    wavs.sort()

    # process
    shutil.rmtree("tmp", ignore_errors=True)
    os.makedirs("tmp", exist_ok=True)
    mp.set_start_method("spawn", force=True)
    n_processes = args.n_processes
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        tasks = []
        for i in range(n_processes):
            start = int(i * len(wavs) / n_processes)
            end = int((i + 1) * len(wavs) / n_processes)
            batch = wavs[start:end]
            tasks.append(executor.submit(process_batch, batch, args.title, args.lang))
        for task in tqdm(tasks, position=0):
            task.result()

    # gather
    trans_dict = {}
    for txt in glob(f"tmp/{args.title}+*.txt"):
        basename = txt.split("/")[-1].split(".")[0]
        basename = basename.split("+")[-1]
        with open(txt, "r") as f:
            trans = f.read().strip()
        trans_dict[basename] = trans
    
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
    
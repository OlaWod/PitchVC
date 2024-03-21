from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import argparse
import torch
import librosa
from tqdm import tqdm
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="gt", help="path to tgt txt file")
    parser.add_argument("--wavdir", type=str, default="SOURCE")
    parser.add_argument("--lang", type=str, default="english")
    args = parser.parse_args()

    # load model and processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").cuda()
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")

    # get transcriptions
    wavs = glob(f'{args.wavdir}/*.wav')
    wavs.sort()

    os.makedirs("result", exist_ok=True)

    with open(f"result/{args.title}.txt", "w") as f:
        for path in tqdm(wavs):
            wav, sr = librosa.load(path, sr=16000)
            input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.cuda() # text
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            title = os.path.basename(path)[:-4]
            f.write(f"{title}|{text}\n")
    
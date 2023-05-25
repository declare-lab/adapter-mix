import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import glob

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    wav_tag = config["path"]["wav_tag"]
    txt_dir = os.path.join(in_dir, config["path"]["txt_dir"])
    wav_dir = os.path.join(in_dir, config["path"]["wav_dir"])
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    for spker_id, speaker in enumerate(tqdm(os.listdir(txt_dir))):
        for i, chapter_name in enumerate(tqdm(os.listdir(os.path.join(txt_dir, speaker)))):
            # base_name = txt_name.split(".")[0]
            # base_name_out = base_name.replace('_', '-')
            for i, txt_name in enumerate(tqdm(glob.glob(os.path.join(txt_dir,speaker,chapter_name,"*normalized.txt")))):
                base_name = txt_name.split("/")[-1]
                base_name = base_name.split(".")[0]
                with open(txt_name, "r") as f:
                    text = f.readline().strip("\n")
                text = _clean_text(text, cleaners)

                wav_path = os.path.join(os.path.join(wav_dir, speaker,chapter_name), "{}.wav".format(base_name))
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                        "w",
                    ) as f1:
                        f1.write(text)
                else:
                    print("[Error] No flac file:{}".format(wav_path))
                    continue

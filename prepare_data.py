import numpy as np
import os
import re
import pandas
import glob
import librosa
import wave
import random
from text import grapheme_to_phoneme
from utils.tools import get_configs_of
from model import PreDefinedEmbedder
from g2p_en import G2p
from tqdm import tqdm
import argparse
import torch
import os


def prepare_train_list(config):
    number_of_speakers = config["low_resource"]["number_of_speakers"]
    dataset_name = config["dataset"]
    max_train_duration = config["low_resource"]["max_train_duration"]
    max_val_duration = config["low_resource"]["max_val_duration"]
    out_dir = f"./Subsample/{dataset_name}/{max_train_duration}"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    def load_audio(wav_path,sampling_rate,filter_length,hop_length,trim_top_db):
        wav_raw, _ = librosa.load(wav_path, sampling_rate)
        _, index = librosa.effects.trim(wav_raw, top_db= trim_top_db, frame_length= filter_length, hop_length= hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)
    
    def compute_spker_embed(out_dir,sample_list,config):
        embed_path = out_dir + '/' + "spker_embed"
        if not os.path.exists(embed_path):
            os.makedirs(embed_path)
        speaker_emb_dict = {}
        root_dir = config["path"]["raw_path"]
        sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        hop_length = config["preprocessing"]["stft"]["hop_length"]
        filter_length = config["preprocessing"]["stft"]["filter_length"]
        trim_top_db = config["preprocessing"]["audio"]["trim_top_db"]
        
        speaker_emb = PreDefinedEmbedder(preprocess_config)
        for sample in tqdm(sample_list):
            basename, speaker,_,_ = sample.split('|')
            wav_path = root_dir + '/' + speaker +'/' + basename + ".wav"
            wav_raw, wav, duration =  load_audio(wav_path,sampling_rate,filter_length,hop_length,trim_top_db)
            spker_embed = speaker_emb(wav)
            if speaker not in speaker_emb_dict.keys():
                speaker_emb_dict[speaker] = [spker_embed]
            else:
                speaker_emb_dict[speaker].append(spker_embed)
        
        for speaker in speaker_emb_dict.keys():
            spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
            np.save(os.path.join(out_dir, 'spker_embed', spker_embed_filename), \
            np.mean(speaker_emb_dict[speaker], axis=0), allow_pickle=False)

    # root_dir = f'/home/yingting/Comprehensive-Transformer-TTS/raw_data/{dataset_name}/'
    root_dir = config["path"]["raw_path"]
    if dataset_name == 'LTS':
        with open("/data/yingting/libritts/LibriTTS/SPEAKERS.txt","r") as f:
            lines = f.readlines()
        speaker_info = {"ID":[],"GENDER":[],"DSET":[],"DURATION":[]}
        for line in lines[12:]:
            id,gender,dset,duration = line.split('|')[0:4]
            if dset.strip() == "train-clean-100":
                # print(id)
                speaker_info["ID"].append(id.strip())
                speaker_info["GENDER"].append(gender.strip())
                speaker_info["DSET"].append(dset.strip())
                speaker_info["DURATION"].append(float(duration.strip()))
        # INFO = pandas.DataFrame.from_dict(speaker_info)
    elif dataset_name == "VCTK":
        with open("/data/yingting/Dataset/VCTK/speaker-info.txt","r") as f:
            lines = f.readlines()
        speaker_info = {"ID":[],'AGE':[],"GENDER":[],"ACCENTS":[],"DURATION":[]}
        for line in lines[1:]:
            line = [x for x in line.split(' ') if x != '']
            speaker_info["ID"].append(f'p{line[0]}'.strip())
            speaker_info["AGE"].append(line[1].strip())
            speaker_info["GENDER"].append(line[2].strip())
            speaker_info["ACCENTS"].append(line[3].strip())
            speaker_info["DURATION"].append(0.0)
    elif dataset_name == "L2ARCTIC":
        with open("/data/yingting/Dataset/L2ARCTIC/speaker-info.txt","r") as f:
            lines = f.readlines()
        speaker_info = {"ID":[],"GENDER":[],"NATIVE":[],"NUM":[],"ANNOTATION":[]}
        for line in lines[2:]:
            id,gender,native,num,annotation = line.split('|')[1:6]
            speaker_info["ID"].append(id.strip())
            speaker_info["GENDER"].append(gender.strip())
            speaker_info["NATIVE"].append(native.strip())
            speaker_info["NUM"].append(num.strip())
            speaker_info["ANNOTATION"].append(annotation.strip())        
    INFO = pandas.DataFrame.from_dict(speaker_info)
        
    for index,row in tqdm(INFO.iterrows()):
        print("Comuting duration for speaker {}".format(row["ID"]))
        data_path = root_dir+row["ID"]
        wav_files = glob.glob(f"{data_path}/*.wav")
        total_duration = 0.0
        for wav in wav_files:
            with wave.open(wav) as mywav:
                duration_seconds = mywav.getnframes() / mywav.getframerate()
                # print(f"Length of the WAV file: {duration_seconds:.1f} s")
            total_duration+= duration_seconds
        if dataset_name == 'LTS':
            print("Old duration: {}, New duration: {}".format(row["DURATION"],total_duration/60))
        elif dataset_name == 'VCTK':
            print("Total duration: {}".format(total_duration/60))
            
        INFO.at[index,"DURATION"] = total_duration/60.00
    MALE_INFO = INFO.loc[INFO['GENDER'] == "M"]
    FEMALE_INFO = INFO.loc[INFO["GENDER"] == "F"]
    # MALE_INFO = MALE_INFO.sort_values(by=["DURATION"],ascending=False,ignore_index=True)[:number_of_speakers]
    # FEMALE_INFO = FEMALE_INFO.sort_values(by=["DURATION"],ascending=False,ignore_index=True)[:number_of_speakers]
    frames = [MALE_INFO,FEMALE_INFO]
    df = pandas.concat(frames)
    print(df.tail(10))
    g2p = G2p()
    # df0 = df.groupby()
    train_unsup = []
    val_unsup = []
    for index,row in tqdm(df.iterrows()):
        data_path = root_dir+'/'+row["ID"]
        wav_files = glob.glob(f"{data_path}/*.wav")
        txt_files = glob.glob(f"{data_path}/*.lab")
        train_batch_duration = 0
        train_batch = []
        while train_batch_duration <= max_train_duration:
            sample = random.sample(wav_files,1)[0]
            text_path = sample.split('.')[0]+".lab"
            with wave.open(sample) as mywav:
                duration_seconds = mywav.getnframes() / mywav.getframerate()
                # print(f"Length of the WAV file: {duration_seconds:.1f} s")
            train_batch_duration+=duration_seconds
            train_batch.append(sample)
            wav_files.remove(sample)
            
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")
            phone = grapheme_to_phoneme(raw_text, g2p)
            phones = "{" + "}{".join(phone) + "}"
            phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
            text_unsup = phones.replace("}{", " ")
            b = sample.split('/')[-1]
            if dataset_name == 'LTS':
                m,n,o,p = b.split('_')
                basename = "_".join([n,o,p])
            elif dataset_name == 'VCTK':
                basename = b
            TEXT = basename.split('.')[0]+"|"+row["ID"]+"|"+text_unsup+"|"+raw_text
            train_unsup.append(TEXT)
        
        val_batch_duration = 0
        val_batch = []
        while val_batch_duration <= max_val_duration:
            sample = random.sample(wav_files,1)[0]
            text_path = sample.split('.')[0]+".lab"
            with wave.open(sample) as mywav:
                duration_seconds = mywav.getnframes() / mywav.getframerate()
                # print(f"Length of the WAV file: {duration_seconds:.1f} s")
            val_batch_duration+=duration_seconds
            val_batch.append(sample)
            wav_files.remove(sample)
            
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")
            phone = grapheme_to_phoneme(raw_text, g2p)
            phones = "{" + "}{".join(phone) + "}"
            phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
            text_unsup = phones.replace("}{", " ")
            b = sample.split('/')[-1]
            if dataset_name == 'LTS':
                m,n,o,p = b.split('_')
                basename = "_".join([n,o,p])
            elif dataset_name == 'VCTK':
                basename = b
            TEXT = basename.split('.')[0]+"|"+row["ID"]+"|"+text_unsup+"|"+raw_text
            val_unsup.append(TEXT)
        
    # Write metadata
    random.shuffle(train_unsup)
    random.shuffle(val_unsup)
    compute_spker_embed(out_dir,train_unsup,config)
    with open(os.path.join(out_dir, "train_unsup.txt"), "w", encoding="utf-8") as f:
        for m in train_unsup:
            f.write(m + "\n")
            
    with open(os.path.join(out_dir, "val_unsup.txt"), "w", encoding="utf-8") as f:
        for m in val_unsup:
            f.write(m + "\n")
        
if __name__ == "__main__":
    # Move to preprocess config
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()
    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    prepare_train_list(preprocess_config)
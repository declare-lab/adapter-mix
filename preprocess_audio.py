# from encoder.audio import *
# from synthesizer.inference import Synthesizer
# import soundfile as sf
from scipy.io import wavfile
import noisereduce as nr
from glob import glob
from tqdm import tqdm
import argparse
import os
# try:
#     import webrtcvad
# except:
#     warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
#     webrtcvad=None
def denoise_audio(args):
    org_wav_dir = args.org_wav_dir
    denoised_wav_dir = args.denoised_wav_dir
    org_wav_files = f"{org_wav_dir}/*/*/*.wav"
    wav_files = glob(org_wav_files)
    for wav in tqdm(wav_files):
        if not "GT" in wav:
            rate, data = wavfile.read(wav)
            reduced_noise = nr.reduce_noise(y=data, sr=rate)
            file_list = wav.split("/")
            file_name = file_list[-1]
            target_folder = file_list[-2]
            model_type = file_list[-3]
            if os.path.exists(os.path.join(denoised_wav_dir,model_type,target_folder)):
                # print(os.path.join(converted_dir,target_folder,file_name))
                wavfile.write(os.path.join(denoised_wav_dir,model_type,target_folder,file_name), rate, reduced_noise)
            else:
                os.makedirs(os.path.join(denoised_wav_dir,model_type,target_folder))
                wavfile.write(os.path.join(denoised_wav_dir,model_type,target_folder,file_name), rate, reduced_noise)
            
    # wav_path = "/data/dataset/samplesMLVAE/200000/SVBI_Hindi_What was the object of your little sensation.wav"
    # num_generated = 0
    # sample_rate = 22050
    # generated_wav = preprocess_wav(wav_path,normalize=True,trim_silence=True)
    # # Save it on the disk
    # filename = "demo_output_%02d.wav" % num_generated
    # print(generated_wav.dtype)
    # sf.write(filename, generated_wav.astype(np.float32), sample_rate)
    # num_generated += 1
    # print("\nSaved output as %s\n\n" % filename)

    # load data
    # rate, data = wavfile.read(wav_path)
    # perform noise reduction
    # reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--org_wav_dir",
    type=str,
    required=True,
    help="Path to orignal wav files",
)
    parser.add_argument(
    "--denoised_wav_dir",
    type=str,
    required=True,
    help="Path to denoised  wav files",
)
    args = parser.parse_args()
    
    denoise_audio(args)
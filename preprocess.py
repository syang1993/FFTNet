from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import fnmatch
import numpy as np
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from hparams import hparams
from utils import audio

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def _process_wav(wav_path, audio_path, spc_path):
    wav = audio.load_wav(wav_path)
    # Extract mels
    spc = audio.melspectrogram(wav).astype(np.float32)

    # Align audios and mels
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    length_diff = len(spc) * hop_length - len(wav)
    wav = wav.reshape(-1,1)
    if length_diff > 0:
        wav = np.pad(wav, [[0, length_diff], [0, 0]], 'constant')
    elif length_diff < 0:
        wav = wav[: hop_length * spc.shape[0]]

    np.save(audio_path, wav)
    np.save(spc_path, spc)
    return (audio_path, spc_path, spc.shape[0])


def calc_stats(file_list, out_dir):
    scaler = StandardScaler()
    for i, filename in enumerate(file_list):
        feat = np.load(filename)
        scaler.partial_fit(feat)

    mean = scaler.mean_
    scale = scaler.scale_
    if hparams.feature_type == "mcc":
        mean[0] = 0.0
        scale[0] = 1.0
    
    np.save(os.path.join(out_dir, 'mean'), np.float32(mean))
    np.save(os.path.join(out_dir, 'scale'), np.float32(scale))


def build_from_path(in_dir, audio_out_dir, mel_out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wav_list = find_files(in_dir)
    for wav_path in wav_list:
        fid = os.path.basename(wav_path).replace('.wav','.npy')
        audio_path = os.path.join(audio_out_dir, fid)
        mel_path = os.path.join(mel_out_dir, fid)
        futures.append(executor.submit(partial(_process_wav, wav_path, audio_path, mel_path)))

    return [future.result() for future in tqdm(futures)]
    

def preprocess(args):
    in_dir = os.path.join(args.wav_dir)
    out_dir = os.path.join(args.output)
    audio_out_dir = os.path.join(out_dir, 'audios')
    mel_out_dir = os.path.join(out_dir, 'mels')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(mel_out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, audio_out_dir, mel_out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)

    spc_list = find_files(mel_out_dir, "*.npy")
    calc_stats(spc_list, out_dir)
     

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='database/audio')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import os
import time
import scipy.io.wavfile
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from tqdm import tqdm
from utils import infolog
from fftnet import FFTNet
from hparams import hparams, hparams_debug_string
from utils.utils import mu_law_encode, mu_law_decode

def display(string, variables):
    sys.stdout.write(f'\r{string}' % variables)

def prepare_data(lc_file, upsample_factor, receptive_field, read_fn=lambda x: x):
    samples = [128] * receptive_field
    local_condition = read_fn(lc_file)
    local_condition = np.repeat(local_condition, upsample_factor, axis=0)
    local_condition = np.pad(local_condition, [[receptive_field, 0], [0, 0]], 'constant')
    local_condition = local_condition[np.newaxis, :, :]
    return samples, torch.FloatTensor(local_condition).transpose(1, 2)

def write_wav(wav, sample_rate, filename):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(filename, sample_rate, wav.astype(np.int16))
    print('Updated wav file at {}'.format(filename))


def generate_fn(args):
    device = torch.device("cuda" if hparams.use_cuda else "cpu")
    upsample_factor = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)

    model = FFTNet(
        n_stacks=hparams.n_stacks,
        fft_channels=hparams.fft_channels,
        quantization_channels=hparams.quantization_channels,
        local_condition_channels=hparams.num_mels)

    model.load_state_dict(torch.load(
        args.checkpoint, map_location=lambda storage, loc: storage))

    model.to(device)

    model.eval()

    with torch.no_grad():
        samples, local_condition = prepare_data(args.lc_file, upsample_factor, 
                                                model.receptive_field, read_fn=lambda x: np.load(x))

        start = time.time()
        for i in range(local_condition.size(-1)):
            sample = torch.LongTensor(np.array(samples[-model.receptive_field:])).unsqueeze(0)
            sample = sample.unsqueeze(-1)
            h = local_condition[:, :, i : i + model.receptive_field]
            sample, h = sample.to(device), h.to(device)
            output = F.softmax(model(sample, h), dim=1)

            prediction = output.data[0, :, -1].cpu().numpy()
            temperature = 1.0
            np.seterr(divide='ignore')
            scaled_prediction = np.log(prediction) / temperature
            scaled_prediction = (scaled_prediction -
                                 np.logaddexp.reduce(scaled_prediction))
            scaled_prediction = np.exp(scaled_prediction)
            np.seterr(divide='warn')
            sample = np.random.choice(
                np.arange(hparams.quantization_channels),
                p=scaled_prediction)
            speed = (i + 1) / (time.time() - start)
            display('Generating: %i/%i, Speed: %.2f samples/sec', (i, local_condition.size(-1), speed))

            samples.append(sample)

        waveform = mu_law_decode(
            np.asarray(samples[model.receptive_field:]),
            hparams.quantization_channels)
        write_wav(waveform, hparams.sample_rate, "generated.wav")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
        help='Checkpoint path to restore model')
    parser.add_argument('--lc_file', type=str, required=True,
        help='Local condition file path.')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    hparams.parse(args.hparams)
    generate_fn(args)

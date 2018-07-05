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
    samples = [0.0] * receptive_field
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

    if hparams.feature_type == 'mcc':
        lc_channel = hparams.mcep_dim + 3
    else:
        lc_channel = hparams.num_mels

    model = FFTNet(
        n_stacks=hparams.n_stacks,
        fft_channels=hparams.fft_channels,
        quantization_channels=hparams.quantization_channels,
        local_condition_channels=lc_channel)
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    model.to(device)

    model.eval()

    with torch.no_grad():
        samples, local_condition = prepare_data(args.lc_file, upsample_factor, 
                                                model.receptive_field, read_fn=lambda x: np.load(x))

        start = time.time()
        for i in tqdm(range(local_condition.size(-1) - model.receptive_field)):
            sample = torch.FloatTensor(np.array(samples[-model.receptive_field:]).reshape(1,-1,1))
            h = local_condition[:, :, i+1 : i+1 + model.receptive_field]
            sample, h = sample.to(device), h.to(device)
            output = model(sample, h)
            output = F.softmax(output, dim=1)
            #sample = output.argmax(1)[0,:].cpu().numpy()
            output=output[0,:,0]
            dist = torch.distributions.Categorical(output)
            sample = dist.sample()
            sample = mu_law_decode(np.asarray(sample), 256)
            samples.append(sample)


        write_wav(np.asarray(samples), hparams.sample_rate, "generated.wav")


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

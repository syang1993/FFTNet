from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import os
import time
import scipy.io.wavfile
from sklearn.preprocessing import StandardScaler

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from torch.nn import functional as F
from tqdm import tqdm
from utils import infolog
from fftnet import FFTNet
from hparams import hparams, hparams_debug_string
from utils.utils import mu_law_encode, mu_law_decode, write_wav


def prepare_data(lc_file, upsample_factor, receptive_field, read_fn=lambda x: x, feat_transform=None):
    samples = [0.0] * receptive_field
    local_condition = read_fn(lc_file)
    uv = local_condition[:, 0]
    if feat_transform is not None:
        local_condition = feat_transform(local_condition)
    uv = np.repeat(uv, upsample_factor, axis=0)
    local_condition = np.repeat(local_condition, upsample_factor, axis=0)
    uv = np.pad(uv, [receptive_field, 0], 'constant')
    local_condition = np.pad(local_condition, [[receptive_field, 0], [0, 0]], 'constant')
    local_condition = local_condition[np.newaxis, :, :]
    return samples, torch.FloatTensor(local_condition).transpose(1, 2), uv


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
    
    if hparams.feature_type == "mcc":
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(args.data_dir, 'mean.npy'))
        scaler.scale_ = np.load(os.path.join(args.data_dir, 'scale.npy'))
        feat_transform = transforms.Compose([lambda x: scaler.transform(x)])
    else:
        feat_transform = None

    with torch.no_grad():
        samples, local_condition, uv = prepare_data(args.lc_file, upsample_factor, 
                                                model.receptive_field, read_fn=lambda x: np.load(x), feat_transform=feat_transform)

        start = time.time()
        for i in tqdm(range(local_condition.size(-1) - model.receptive_field)):
            sample = torch.FloatTensor(np.array(samples[-model.receptive_field:]).reshape(1,-1,1))
            h = local_condition[:, :, i+1 : i+1 + model.receptive_field]
            sample, h = sample.to(device), h.to(device)
            output = model(sample, h)
           
            if hparams.feature_type == "mcc":
                if uv[i+model.receptive_field] == 0:
                    output = output[0, :, -1]
                    outprob = F.softmax(output, dim=0).cpu().numpy()
                    sample = np.random.choice(
                        np.arange(hparams.quantization_channels),
                        p=outprob)
                else:
                    output = output[0, :, -1] * 2
                    outprob = F.softmax(output, dim=0).cpu().numpy()
                    sample = outprob.argmax(0)
            else:
                outprob = output[0, :, -1].cpu().numpy()

            sample = mu_law_decode(sample, hparams.quantization_channels)
            samples.append(sample)


        write_wav(np.asarray(samples), hparams.sample_rate, os.path.join(os.path.dirname(args.checkpoint), "generated-t2-argmaxvoice-samplingunvoice-{}.wav".format(os.path.basename(args.checkpoint))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
        help='Checkpoint path to restore model')
    parser.add_argument('--lc_file', type=str, required=True,
        help='Local condition file path.')
    parser.add_argument('--data_dir', type=str, default='training_data_mcc', 
        help='data dir')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    hparams.parse(args.hparams)
    generate_fn(args)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import io
import numpy as np
import os
import torch


from torch.utils.data import Dataset
from utils.utils import mu_law_encode

class CustomDataset(Dataset):

    def __init__(self,
                 meta_file,
                 receptive_field,
                 sample_size=5000,
                 upsample_factor=200,
                 quantization_channels=256,
                 use_local_condition=True,
                 noise_injecting=True,
                 feat_transform=None):
        with open(meta_file, encoding='utf-8') as f:
            self.metadata = [line.strip().split('|') for line in f]
        self.receptive_field = receptive_field
        self.sample_size = sample_size
        self.upsample_factor = upsample_factor
        self.quantization_channels = quantization_channels
        self.use_local_condition = use_local_condition
        self.feat_transform = feat_transform
        self.noise_injecting = noise_injecting
        self.audio_buffer, self.local_condition_buffer = self._load_data(
                           self.metadata, use_local_condition, post_fn=lambda x: np.load(x))

    def __len__(self):
        return len(self.audio_buffer)

    def __getitem__(self, index):
        audios = self.audio_buffer[index]
        rand_pos = np.random.randint(0, len(audios) - self.sample_size)

        if self.use_local_condition:
            local_condition = self.local_condition_buffer[index]
            local_condition = np.repeat(local_condition, self.upsample_factor, axis=0)
            local_condition = local_condition[rand_pos : rand_pos + self.sample_size]
        else:
            audios = np.pad(audios, [[self.receptive_field, 0], [0, 0]], 'constant')
            local_condition = None

        audios = audios[rand_pos : rand_pos + self.sample_size]
        target = mu_law_encode(audios, self.quantization_channels)
        if self.noise_injecting:
            noise = np.random.normal(0.0, 1.0/self.quantization_channels, audios.shape)
            audios = audios + noise

        audios = np.pad(audios, [[self.receptive_field, 0], [0, 0]], 'constant')
        local_condition = np.pad(local_condition, [[self.receptive_field, 0], [0, 0]], 'constant')

        return torch.FloatTensor(audios), torch.LongTensor(target), torch.FloatTensor(local_condition)

    def _load_data(self, metadata, use_local_condition, post_fn=lambda x: x):
        audio_buffer = []
        local_condition_buffer = []
        for x in metadata:
            tmp_data = post_fn(x[0])
            if len(tmp_data) - self.sample_size - self.receptive_field > 0:
                audio_buffer.append(tmp_data)
                if use_local_condition:
                    feat = post_fn(x[1])
                    if self.feat_transform is not None:
                        feat = self.feat_transform(feat)
                    local_condition_buffer.append(feat)
        return audio_buffer, local_condition_buffer


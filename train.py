from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from fftnet import FFTNet
from dataset import CustomDataset
from utils.utils import apply_moving_average, ExponentialMovingAverage
from utils import infolog
from hparams import hparams, hparams_debug_string
from tensorboardX import SummaryWriter
log = infolog.log


def save_checkpoint(device, hparams, model, step, checkpoint_dir, ema=None):
    model = model.module if isinstance(model, nn.DataParallel) else model

    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}.pt".format(step))
    torch.save(model.state_dict(), checkpoint_path)
    log("Saved checkpoint: {}".format(checkpoint_path))

    if ema is not None:
        averaged_model = clone_as_averaged_model(device, hparams, model, ema)
        checkpoint_path = os.path.join(
            checkpoint_dir, "model.ckpt-{}.ema.pt".format(step))
        torch.save(averaged_model.state_dict(), checkpoint_path)
        log("Saved averaged checkpoint: {}".format(checkpoint_path))


def clone_as_averaged_model(device, hparams, model, ema):
    assert ema is not None
    averaged_model = create_model(hparams).to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model

def create_model(hparams):
    return FFTNet(n_stacks=hparams.n_stacks,
                  fft_channels=hparams.fft_channels,
                  quantization_channels=hparams.quantization_channels,
                  local_condition_channels=hparams.num_mels)


def train_fn(args):
    device = torch.device("cuda" if hparams.use_cuda else "cpu")
    upsample_factor = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)

    model = create_model(hparams)

    if args.resume is not None:
        log("Resume checkpoint from: {}:".format(args.resume))
        model.load_state_dict(torch.load(
            args.resume, map_location=lambda storage, loc: storage))
        global_step = int(re.findall(r"\d+\d*", args.resume)[-1])
    else:
        global_step = 0

    log("receptive field: {0} ({1:.2f}ms)".format(
        model.receptive_field, model.receptive_field / hparams.sample_rate * 1000))

    dataset = CustomDataset(meta_file=args.input, 
                            receptive_field=model.receptive_field,
                            sample_size=hparams.sample_size,
                            upsample_factor=upsample_factor,
                            quantization_channels=hparams.quantization_channels,
                            use_local_condition=hparams.use_local_condition)

    dataloader = DataLoader(dataset, batch_size=hparams.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)

    ema = ExponentialMovingAverage(args.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    writer = SummaryWriter(args.checkpoint_dir)

    while global_step < hparams.training_steps:
        for i, data in enumerate(dataloader, 0):
            audio, local_condition = data
            target = audio.squeeze(-1)
            local_condition = local_condition.transpose(1, 2)
            audio, target, h = audio.to(device), target.to(device), local_condition.to(device)

            optimizer.zero_grad()
            output = model(audio, h)
            loss = criterion(output[:, :, 1:], target[:, model.receptive_field:])
            log('step [%3d]: loss: %.3f' % (global_step, loss.item()))
            writer.add_scalar('loss', loss.item(), global_step)

            loss.backward()
            optimizer.step()

            # update moving average
            if ema is not None:
                apply_moving_average(model, ema)

            global_step += 1

            if global_step % hparams.checkpoint_interval == 0:
                save_checkpoint(device, hparams, model, global_step, args.checkpoint_dir, ema)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--input', default='training_data/train.txt',
        help='Metadata file which contains the keys of audio and melspec')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
        help='Moving average decay rate.')
    parser.add_argument('--num_workers',type=int, default=4, 
        help='Number of dataloader workers.')
    parser.add_argument('--resume', type=str, default=None, 
        help='Checkpoint path to resume')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', 
        help='Directory to save checkpoints.')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    infolog.init(os.path.join(args.checkpoint_dir, 'train.log'), 'FFTNET')
    hparams.parse(args.hparams)
    train_fn(args)

from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn, optim
import torch

from scheduler import CycleScheduler
import distributed as dist

from tqdm import tqdm
from time import time
import argparse
import sys
import os

# MODEl ARCHITECTURE, ie. nn.Module CODE
#
# This is a modification of licensed source code that has
# been released into the public domain by its author(s)
#
# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TRAINING CODE
#
# This is a modification of licensed source code that has
# been released into the public domain by its author(s)
#
# MIT License
#
# Copyright (c) 2019 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class Quantizer(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.n_embed = n_embed
        self.decay = decay
        self.dim = dim
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        embed_avg = embed.clone()
        cluster_size = torch.zeros(n_embed)

        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed_avg)
        self.register_buffer('cluster_size', cluster_size)

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist.all_reduce(embed_onehot_sum)
            dist.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ])

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=4,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = nn.Sequential(
            Encoder(channel, channel, n_res_block, n_res_channel, stride=4),
            Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        )
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantizer(embed_dim, n_embed)
        self.dec_t = nn.Sequential(
            Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2),
            Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=4),
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantizer(embed_dim, n_embed)
        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=4, padding=0),
        )
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

    def train_epoch(self, epoch, loader, optimizer, scheduler, device, sample_path):
        if dist.is_primary():
            loader = tqdm(loader)

        criterion = nn.MSELoss()

        latent_loss_weight = 0.25
        sample_size = 25

        mse_sum = 0
        mse_n = 0

        for i, (img, label) in enumerate(loader):
            self.zero_grad()

            img = img.to(device)
            out, latent_loss = self(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss
            loss.backward()

            if not scheduler is None:
                scheduler.step()

            optimizer.step()

            part_mse_sum = recon_loss.item() * img.shape[0]
            part_mse_n = img.shape[0]
            comm = {'mse_sum': part_mse_sum, 'mse_n': part_mse_n}
            comm = dist.all_gather(comm)

            for part in comm:
                mse_sum += part['mse_sum']
                mse_n += part['mse_n']

            if dist.is_primary():
                lr = optimizer.param_groups[0]['lr']

                loader.set_description((
                    f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                    f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                    f'lr: {lr:.5f}'
                ))

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    os.path.join(sample_path, f'{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'),
                    nrow = sample_size,
                    normalize = True,
                    range = (-1, 1),
                )

                model.train()

    def train(self, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.distributed = dist.get_world_size() > 1

        transform = [transforms.ToTensor()]

        if args.normalize:
            transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        transform = transforms.Compose(transform)

        dataset = datasets.ImageFolder(args.path, transform = transform)
        sampler = dist.data_sampler(dataset, shuffle = True, distributed = args.distributed)
        loader = DataLoader(dataset, batch_size = args.batch_size // args.n_gpu, sampler = sampler, num_workers = args.num_workers)

        self = self.to(device)

        if args.distributed:
            self = nn.parallel.DistributedDataParallel(
                self,
                device_ids = [dist.get_local_rank()],
                output_device = dist.get_local_rank()
            )

        optimizer = args.optimizer(self.parameters(), lr = args.lr)
        schedular = None
        if args.sched == 'cycle':
            scheduler = CycleScheduler(
                optimizer,
                args.lr,
                n_iter = len(loader) * args.epoch,
                momentum = None,
                warmup_proportion = 0.05,
            )

        start = str(time())
        run_path = os.path.join('runs', start)
        sample_path = os.path.join(run_path, 'sample')
        checkpoint_path = os.path.join(run_path, 'checkpoint')
        os.mkdir(run_path)
        os.mkdir(sample_path)
        os.mkdir(checkpoint_path)

        for epoch in range(args.epoch):
            self.train_epoch(epoch, loader, optimizer, scheduler, device, sample_path)

            if dist.is_primary():
                torch.save(self.state_dict(), os.path.join(checkpoint_path, f'vqvae_{str(i + 1).zfill(3)}.pt'))

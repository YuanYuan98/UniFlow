# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

import copy

# sinusoidal positional embeds

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        print(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_att=None):
        x = self.conv1(x, edge_index, edge_att)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_att)
        return x

class Memory(nn.Module):
    """ Memory prompt
    """
    def __init__(self, num_memory, memory_dim, temporal_agg = False, args=None):
        super().__init__()

        self.args = args

        self.num_memory = num_memory
        self.memory_dim = memory_dim
        self.temporal_agg = temporal_agg

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C

        self.memMatrix.requires_grad = True
        self.keyMatrix.requires_grad = True

        self.x_proj = nn.Linear(memory_dim, memory_dim)

        if temporal_agg:
            encdoer_layer = nn.TransformerEncoderLayer(d_model=memory_dim, nhead=4, dim_feedforward=memory_dim, batch_first = True)
            self.encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)
        
        self.initialize_weights()

        print("model initialized memory")


    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x,Type='',shape=None):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product

        if self.temporal_agg:
            x = self.encoder(x).mean(dim=1)

        assert x.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "dimension mismatch"

        x_query = torch.tanh(self.x_proj(x))

        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]

        return dict(out=out, att_weight=att_weight)



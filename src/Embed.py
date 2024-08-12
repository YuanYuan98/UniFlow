import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
from Prompt_network import Memory, GCN


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(TokenEmbedding, self).__init__()
        kernel_size = [t_patch_size,patch_size, patch_size]
        self.tokenConv = nn.Conv3d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=kernel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.tokenConv(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*H*W, C]
        return x
    

class SpatialPatchEmb(nn.Module):
    def __init__(self, c_in, d_model, patch_size):
        super(SpatialPatchEmb, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        self.d_model = d_model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.reshape(B, self.d_model, H*W//self.patch_size**2).permute(0,2,1)
        return x

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, t_patch_size = 1, hour_size=48, weekday_size = 7):
        super(TemporalEmbedding, self).__init__()

        hour_size = hour_size
        weekday_size = weekday_size

        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.timeconv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=t_patch_size, stride=t_patch_size)

    def forward(self, x):
        x = x.long()
        hour_x = self.hour_embed(x[:,:,1])
        weekday_x = self.weekday_embed(x[:,:,0])
        timeemb = self.timeconv(hour_x.transpose(1,2)+weekday_x.transpose(1,2)).transpose(1,2)

        return timeemb

class GraphEmbedding(nn.Module):
    def __init__(self, c_in, d_model, GridEmb, dropout=0.1, args=None, size1 = 48, size2=7):
        super(GraphEmbedding, self).__init__()
        self.temporal_embedding = TemporalEmbedding(t_patch_size = args.t_patch_size, d_model=d_model, hour_size  = size1, weekday_size = size2)
        self.dropout = nn.Dropout(p=dropout)
        self.c_in = c_in
        self.d_model = d_model
        self.MIN, self.MID, self.MAX = [2,2,100]
        self.gcn = GCN(d_model, d_model, d_model)
        self.DataEmb = GridEmb
        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=args.t_patch_size, stride=args.t_patch_size)
        encdoer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=2, dim_feedforward=self.d_model//2, batch_first = True)
        self.spatial_attn = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)
        self.args = args

    def temporal_emb(self, x_mark, hour_num):
        if '24' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_24.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        elif '48' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_48.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        elif '288' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_288.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        elif '96' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_96.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        
        return TimeEmb

    def forward(self, x, x_mark, edges=None, node_split=None, is_time=1, patch_size=None, hour_num = None):
        '''
        x: N, T, 1, H, 1
        x_mark: N, T, D
        '''
        N, _, T, H, _ = x.shape
        if '24' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_24(x_mark) # N * seqlen//t_patch_size * D
        elif '48' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_48(x_mark)
        elif '288' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_288(x_mark)
        elif '96' in hour_num:
            TimeEmb = self.DataEmb.temporal_patch_96(x_mark)

        x = x.squeeze((1,4)).permute(0,2,1).reshape(N * H, 1, self.args.his_len + self.args.pred_len)
        temporal_value_emb = self.temporal_conv(x).permute(0,2,1).reshape(N, H, -1, self.d_model).permute(0,2,1,3).reshape(-1, H, self.d_model)
        
        TokenEmb = self.gcn(temporal_value_emb, edges.permute(1,0)).reshape(N, -1, H, self.d_model) # N * seqlen//t_patch_size * H * D

        TokenEmb = torch.cat([torch.mean(torch.gather(TokenEmb, 2 ,group.view(1, 1, group.shape[0], 1).expand(TokenEmb.shape[0], TokenEmb.shape[1], group.shape[0], TokenEmb.shape[3]).to(x).long()),dim=2) for group in node_split],2).reshape(N, -1, TokenEmb.shape[-1])

        assert TokenEmb.shape[1] == TimeEmb.shape[1] * len(node_split)
        TimeEmb = torch.repeat_interleave(TimeEmb, TokenEmb.shape[1]//TimeEmb.shape[1], dim=1)
        assert TokenEmb.shape == TimeEmb.shape
        if is_time==1:
            x = TokenEmb + TimeEmb
        else:
            x = TokenEmb
        if self.training:
            return self.dropout(x), TimeEmb
        else:
            return x, TimeEmb


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, args=None, size1 = 48, size2=7):
        super(DataEmbedding, self).__init__()
        self.args = args
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, t_patch_size = args.t_patch_size,  patch_size=args.patch_size)
        self.temporal_embedding = TemporalEmbedding(t_patch_size = args.t_patch_size, d_model=d_model, hour_size  = size1, weekday_size = size2)
        self.dropout = nn.Dropout(p=dropout)
        self.c_in = c_in
        self.d_model = d_model
        self.MIN, self.MID, self.MAX = [2,2,100]

    def temporal_emb(self, x_mark, hour_num):
        if '24' in hour_num:
            TimeEmb = self.temporal_patch_24.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        elif '48' in hour_num:
            TimeEmb = self.temporal_patch_48.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        elif '288' in hour_num:
            TimeEmb = self.temporal_patch_288.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        elif '96' in hour_num:
            TimeEmb = self.temporal_patch_96.hour_embed(x_mark[:,:,1]) + self.temporal_embedding.weekday_embed(x_mark[:,:,0])
        return TimeEmb

    def multi_patch(self):
        self.value_patch_1 = TokenEmbedding(c_in=self.c_in, d_model=self.d_model, t_patch_size = self.args.t_patch_size,  patch_size=1)

        self.value_patch_2 = TokenEmbedding(c_in=self.c_in, d_model=self.d_model, t_patch_size = self.args.t_patch_size,  patch_size=2)

        self.value_patch_4 = TokenEmbedding(c_in=self.c_in, d_model=self.d_model, t_patch_size = self.args.t_patch_size,  patch_size=4)

        self.temporal_patch_48 = TemporalEmbedding(d_model=self.d_model, t_patch_size = self.args.t_patch_size, hour_size=48, weekday_size = 7)
        self.temporal_patch_24 = TemporalEmbedding(d_model=self.d_model, t_patch_size = self.args.t_patch_size, hour_size=24, weekday_size = 7)
        self.temporal_patch_288 = TemporalEmbedding(d_model=self.d_model, t_patch_size = self.args.t_patch_size, hour_size=288, weekday_size = 7)
        self.temporal_patch_96 = TemporalEmbedding(d_model=self.d_model, t_patch_size = self.args.t_patch_size, hour_size=96, weekday_size = 7)


    def forward(self, x, x_mark, edges = None, is_time=1, patch_size=None, hour_num = None):
        '''
        x: N, T, C, H, W
        x_mark: N, T, D
        '''
        N, T, C, H, W = x.shape
        if patch_size == 1:
            TokenEmb = self.value_patch_1(x)
        elif patch_size == 2:
            TokenEmb = self.value_patch_2(x)
        elif patch_size == 4:
            TokenEmb = self.value_patch_4(x)
        if '24' in hour_num:
            TimeEmb = self.temporal_patch_24(x_mark)
        elif '48' in hour_num:
            TimeEmb = self.temporal_patch_48(x_mark)
        elif '288' in hour_num:
            TimeEmb = self.temporal_patch_288(x_mark)
        elif '96' in hour_num:
            TimeEmb = self.temporal_patch_96(x_mark)
        assert TokenEmb.shape[1] == TimeEmb.shape[1] * H // patch_size * W // patch_size
        TimeEmb = torch.repeat_interleave(TimeEmb, TokenEmb.shape[1]//TimeEmb.shape[1], dim=1)
        assert TokenEmb.shape == TimeEmb.shape
        if is_time==1:
            x = TokenEmb + TimeEmb
        else:
            x = TokenEmb
        if self.training:
            return self.dropout(x), TimeEmb
        else:
            return x, TimeEmb


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size1, grid_size2, res, cls_token=False, device="cpu"
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim]
    """
    # res = torch.FloatTensor(res).to(device)
    res = res
    grid_h = torch.arange(grid_size1, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size2, dtype=torch.float32, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    
    pos_embed = pos_embed.reshape(h * w, embed_dim).numpy()
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed_from_grid_with_resolution(embed_dim, pos, res):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = pos * res
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

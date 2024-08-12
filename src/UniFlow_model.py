from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

from Embed import GraphEmbedding, DataEmbedding, TokenEmbedding, SpatialPatchEmb, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid_with_resolution
from mask_strategy import *
import copy

from Prompt_network import Memory, GCN


class TransformerDecoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.linear = nn.Linear(d_model, d_model)  # Adjust the output dimension as needed

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = self.linear(output)
        return output

def model_select(args, **kwargs):
    if args.size == 'small': 
        model = UniFlow(
            embed_dim=128,
            depth=4,
            decoder_embed_dim = 128,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model

    elif args.size == 'middle': 
        model = UniFlow(
            embed_dim=256,
            depth=4,
            decoder_embed_dim = 256,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model
 
    elif args.size == 'large': 
        model = UniFlow(
            embed_dim=256,
            depth=6,
            decoder_embed_dim = 256,
            decoder_depth=6,
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias= qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x, attn_bias = {}):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_bias!={}:
            
            if 'bias_t' in attn_bias:
                T = attn.shape[-1] // attn_bias['bias_t'].shape[-1]
            elif 'bias_f' in attn_bias:
                T = attn.shape[-1] // attn_bias['bias_f'].shape[-1]
            elif 'topo' in attn_bias:
                T = attn.shape[-1] // attn_bias['topo'].shape[-1]

            if 'bias_t' in attn_bias:
                attn_bias_t = attn_bias['bias_t'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
                attn_bias_t = attn_bias_t.repeat(1, self.num_heads, T, 1, T, 1)
                attn_bias_t = attn_bias_t.reshape(attn_bias_t.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])

                assert attn.shape == attn_bias_t.shape

                attn += attn_bias_t

            if 'bias_f' in attn_bias:

                attn_bias_f = attn_bias['bias_f'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
                attn_bias_f = attn_bias_f.repeat(1, self.num_heads, T, 1, T, 1)
                attn_bias_f = attn_bias_f.reshape(attn_bias_f.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])

                assert attn.shape == attn_bias_f.shape
                
                attn += attn_bias_f

            if 'topo' in attn_bias:

                attn_bias_topo = attn_bias['topo'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
                attn_bias_topo = attn_bias_topo.repeat(1, self.num_heads, T, 1, T, 1)
                attn_bias_topo = attn_bias_topo.reshape(attn_bias_topo.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])

                assert attn.shape == attn_bias_topo.shape
                
                attn += attn_bias_topo

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attn_bias = {}):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias = attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class UniFlow(nn.Module):
    def __init__(self,  in_chans=1,
                 embed_dim=1024, decoder_embed_dim=512, depth=24, decoder_depth=4, num_heads=16,  decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, pos_emb = 'trivial', args=None, ):
        super().__init__()

        self.args = args

        self.pos_emb = pos_emb

        self.Embedding_patch = DataEmbedding(1, embed_dim, args=self.args)
        self.Embedding_patch_graph = GraphEmbedding(1, embed_dim, GridEmb = self.Embedding_patch, args=self.args)

        # mask

        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.in_chans = in_chans
        

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, embed_dim)
        )

        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50,  decoder_embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer

        
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        encdoer_layer2 = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=2, dim_feedforward=decoder_embed_dim//2, batch_first = True)
        self.spatial_attn_spec_tmp = nn.TransformerEncoder(encoder_layer=encdoer_layer2, num_layers=1)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_model = TransformerDecoderModel(d_model=decoder_embed_dim, dim_feedforward = decoder_embed_dim//2, nhead=2, num_decoder_layers=1)
        self.pred_model_linear_GraphBJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphNJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphSH  = nn.Linear(decoder_embed_dim, self.t_patch_size * 210 * in_chans)

        self.initialize_weights_trivial()

        print("model initialized")

    def init_multiple_patch(self):
        
        self.Embedding_patch.multi_patch()

        self.head_layer_1 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 1**2 * self.in_chans, bias= not self.args.no_qkv_bias)
        ])

        self.head_layer_2 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 2**2 * self.in_chans, bias= not self.args.no_qkv_bias)
        ])
        
        self.head_layer_4 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 4**2 * self.in_chans, bias= not self.args.no_qkv_bias)
        ])

        self.initialize_weights_trivial()

    def init_prompt(self):
        self.spec_mlp = nn.Sequential(*[
            nn.Linear(self.args.his_len+2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        ])
        self.prompt_spatial_patch_t_1 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 1)
        self.prompt_spatial_patch_f_1 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 1)
        self.prompt_spatial_patch_t_2 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 2)
        self.prompt_spatial_patch_f_2 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 2)
        self.prompt_spatial_patch_t_4 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 4)
        self.prompt_spatial_patch_f_4 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 4)
        self.temporal_patch = nn.Conv1d(in_channels=1, out_channels=self.embed_dim, kernel_size=self.args.t_patch_size, stride=self.args.t_patch_size)
        encdoer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=2, dim_feedforward=self.embed_dim, batch_first = True)
        self.temporal_attn_encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)
        self.temporaltokenConv = nn.Conv1d(in_channels=1, out_channels=self.embed_dim, kernel_size=3, stride=1,  padding = 1, padding_mode='circular', bias=False)
        self.gcn_t = GCN(self.embed_dim, self.embed_dim, self.embed_dim)
        self.gcn_f = GCN(self.embed_dim, self.embed_dim, self.embed_dim)
        self.gcn_topo_t = GCN(self.embed_dim, self.embed_dim, self.embed_dim)
        self.gcn_topo_f = GCN(self.embed_dim, self.embed_dim, self.embed_dim)

        self.spec_liner = nn.Linear(2*(self.args.his_len//2+1), self.embed_dim)

        self.enc_memory_t = Memory(num_memory=self.args.num_memory, memory_dim=self.embed_dim, args=self.args)
        self.enc_memory_f = Memory(num_memory=self.args.num_memory, memory_dim=self.embed_dim, args=self.args)
       
        self.prompt_spatial_patch_t_1.apply(self._init_weights)
        self.prompt_spatial_patch_f_1.apply(self._init_weights)
        self.prompt_spatial_patch_t_2.apply(self._init_weights)
        self.prompt_spatial_patch_f_2.apply(self._init_weights)
        self.temporal_patch.apply(self._init_weights)
        self.gcn_t.apply(self._init_weights)
        self.gcn_f.apply(self._init_weights)
        self.gcn_topo_t.apply(self._init_weights)
        self.gcn_topo_f.apply(self._init_weights)

        self.enc_memory_t.apply(self._init_weights)
        self.enc_memory_f.apply(self._init_weights)

    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
            )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.embed_dim)
        )

        pos_embed_spatial.data.copy_(torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.tensor(pos_temporal_emb, dtype=torch.float32).unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(pos_embed_temporal)

    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

        torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = patch_size
        u = self.args.t_patch_size
        assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        #self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x


    def pos_embed_enc(self, ids_keep, batch, input_size):

        pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

        pos_embed = pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
        pos_embed = pos_embed.to(ids_keep.device)

        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )

        return pos_embed_sort

    def pos_embed_dec(self, ids_keep, batch, input_size):

        _, _, decoder_pos_embed_spatial, decoder_pos_embed_temporal  = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

        decoder_pos_embed = decoder_pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                decoder_pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed

    def forward_encoder(self, x, x_mark, mask_ratio, mask_strategy, seed=None, data=None, mode='backward',prompt = {}, patch_size = 1, split_nodes=None):
        # embed patches
        N, _, T, H, W = x.shape

        origin_x = x.clone()

        edges = prompt['topo']

        if 'Graph' not in data:
            x, TimeEmb = self.Embedding_patch(x, x_mark, edges, is_time = self.args.is_time_emb, patch_size=patch_size, hour_num = data)
        else:
            x, TimeEmb = self.Embedding_patch_graph(x, x_mark, edges, split_nodes, is_time = self.args.is_time_emb, patch_size=patch_size, hour_num = data)
        
        _, L, C = x.shape

        T = T // self.args.t_patch_size

        assert mode in ['backward','forward']

        x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)

        if 'Graph' not in data:
            input_size = (T, H//patch_size, W//patch_size)
        else:
            input_size = (T, len(split_nodes), 1)

        pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)
        assert x.shape == pos_embed_sort.shape

        x_attn = x + pos_embed_sort
        
        prompt_save = {}
        attn_bias = {}

        if self.args.is_prompt==1 and 'graph' in self.args.prompt_content:
            
            prompt_t = self.enc_memory_t(prompt['t'].reshape(-1,prompt['t'].shape[-1]))
            prompt_t = prompt_t['out'].reshape(prompt['t'].shape)

            prompt_f = self.enc_memory_f(prompt['f'].reshape(-1,prompt['f'].shape[-1]))
            prompt_f = prompt_f['out'].reshape(prompt['f'].shape)

            adp_t = F.softmax(F.relu(prompt_t @ prompt_t.transpose(1, 2)), dim=-1) # N * (H*W) * (H*W)
            adp_f = F.softmax(F.relu(prompt_f @ prompt_f.transpose(1, 2)), dim=-1)

            data_list_t = []
            data_list_f = []
            edge_att_t, edge_att_f = [], []
            for i in range(adp_t.size(0)):
                edge_index_t = adp_t[i].nonzero().t().contiguous() + i * adp_t.shape[1]
                data_list_t.append(edge_index_t)
                edge_att_t.append(adp_t[i][adp_t[i]!=0])

                edge_index_f = adp_f[i].nonzero().t().contiguous() + i * adp_f.shape[1]
                data_list_f.append(edge_index_f)
                edge_att_f.append(adp_f[i][adp_f[i]!=0])

            edge_t = torch.cat(data_list_t, dim=-1)
            edge_att_t = torch.cat(edge_att_t, dim=0)
            edge_f = torch.cat(data_list_f, dim=-1)
            edge_att_f = torch.cat(edge_att_f, dim=0)

            if 'Graph' not in data:
                prompt_t = self.gcn_t(prompt_t.reshape(-1,prompt_t.shape[-1]), edge_t, edge_att_t).reshape(N, H*W//patch_size**2, self.embed_dim)
                prompt_f = self.gcn_f(prompt_f.reshape(-1,prompt_f.shape[-1]), edge_f, edge_att_f).reshape(N, H*W//patch_size**2, self.embed_dim)

            else:
                prompt_t = self.gcn_topo_t(prompt_t.reshape(-1,prompt_t.shape[-1]), edge_t, edge_att_t).reshape(N, len(split_nodes), self.embed_dim)
                prompt_f = self.gcn_topo_f(prompt_f.reshape(-1,prompt_f.shape[-1]), edge_f, edge_att_f).reshape(N, len(split_nodes), self.embed_dim)

            prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len//self.args.t_patch_size, 1,1).reshape(N, x_attn.shape[1], self.embed_dim)
            prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len//self.args.t_patch_size, 1,1).reshape(N, x_attn.shape[1], self.embed_dim)

            assert prompt_t.shape == prompt_f.shape == x_attn.shape

            prompt_save['t'] = prompt_t.clone()
            prompt_save['f'] = prompt_f.clone()


        if self.args.is_prompt==1 and 'node' in self.args.prompt_content:

            prompt_t = self.enc_memory_t(prompt['t'].reshape(-1,prompt['t'].shape[-1]))
            prompt_t = prompt_t['out'].reshape(prompt['t'].shape)

            prompt_f = self.enc_memory_f(prompt['f'].reshape(-1,prompt['f'].shape[-1]))
            prompt_f = prompt_f['out'].reshape(prompt['f'].shape)

            prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len//self.args.t_patch_size, 1,1).reshape(N, x_attn.shape[1], self.embed_dim)
            prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len//self.args.t_patch_size, 1,1).reshape(N, x_attn.shape[1], self.embed_dim)

            assert prompt_t.shape == prompt_f.shape == x_attn.shape

            prompt_save['node_t'] = prompt_t.clone()
            prompt_save['node_f'] = prompt_f.clone()

        for index, blk in enumerate(self.blocks):
            x_attn = blk(x_attn, attn_bias = attn_bias)
            
        return x_attn, mask, ids_restore, input_size, TimeEmb,  prompt_save

    def forward_decoder(self, x, x_mark, mask, ids_restore, mask_strategy, TimeEmb, input_size=None,  data=None, prompt_graph = {}):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)
            
        C = x.shape[-1]

        x = causal_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)

        decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)

        # add pos embed
        assert x.shape == decoder_pos_embed.shape == TimeEmb.shape

        if self.args.is_time_emb==1:
            x_attn = x + decoder_pos_embed + TimeEmb
        else:
            x_attn = x + decoder_pos_embed

        attn_bias = prompt_graph
        
        if self.args.is_prompt == 1 and 'graph' in self.args.prompt_content:

            prompt_t, prompt_f = prompt_graph['t'], prompt_graph['f']
            prompt_t = prompt_t.reshape(N, -1, H*W, prompt_t.shape[-1])[:,:1].repeat(1,(self.args.his_len+self.args.pred_len)//self.args.t_patch_size,1,1).reshape(N, -1, prompt_t.shape[-1])
            prompt_f = prompt_f.reshape(N, -1, H*W, prompt_f.shape[-1])[:,:1].repeat(1,(self.args.his_len+self.args.pred_len)//self.args.t_patch_size,1,1).reshape(N, -1, prompt_f.shape[-1])

            assert x_attn.shape == prompt_t.shape == prompt_f.shape

            if 'graph_t' in self.args.prompt_content:
                x_attn = x_attn + prompt_t
            elif 'graph_f' in self.args.prompt_content:
                x_attn = x_attn + prompt_f
            else:
                x_attn += prompt_f + prompt_t

        if self.args.is_prompt == 1 and 'node' in self.args.prompt_content:
            prompt_t, prompt_f = prompt_graph['node_t'], prompt_graph['node_f']
            prompt_t = prompt_t.reshape(N, -1, H*W, prompt_t.shape[-1])[:,:1].repeat(1,(self.args.his_len+self.args.pred_len)//self.args.t_patch_size,1,1).reshape(N, -1, prompt_t.shape[-1])
            prompt_f = prompt_f.reshape(N, -1, H*W, prompt_f.shape[-1])[:,:1].repeat(1,(self.args.his_len+self.args.pred_len)//self.args.t_patch_size,1,1).reshape(N, -1, prompt_f.shape[-1])

            assert x_attn.shape == prompt_t.shape == prompt_f.shape
            if 'node_t' in self.args.prompt_content:
                x_attn = x_attn + prompt_t
            elif 'node_f' in self.args.prompt_content:
                x_attn = x_attn + prompt_f
            else:
                x_attn += prompt_f + prompt_t

        # apply Transformer blocks
        for index, blk in enumerate(self.decoder_blocks):
            x_attn = blk(x_attn, attn_bias = attn_bias)
        x_attn = self.decoder_norm(x_attn)

        return x_attn

    def forward_loss(self, imgs, pred, mask, patch_size):
        """
        imgs: [N, 1, T, H, W]
        pred: [N, t*h*w, u*p*p*1]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """

        target = self.patchify(imgs, patch_size)

        assert pred.shape == target.shape

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss1 = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss2 = (loss * (1-mask)).sum() / (1-mask).sum()
        return loss1, loss2, target

    def graph_loss(self,pred, target):
        assert pred.shape == target.shape
        assert pred.shape[1] == self.args.his_len + self.args.pred_len

        loss1 = ((pred[:,self.args.his_len:] - target[:,self.args.his_len:]) ** 2).mean()

        loss2 = ((pred[:,:self.args.his_len] - target[:,:self.args.his_len]) ** 2).mean()

        mask = torch.ones_like(target)

        mask[:,:self.args.his_len] = 0

        return loss1, loss2, target, mask


    def adpative_graph(self, img, img_mark, DataEmbedding, data, node_split=None, patch_size = 2):
        N, _, T, H, W = img.shape
        # img_mark : N * T * 2

        img_origin = img.clone().squeeze(dim=1).reshape(N, T, H*W)
        img_origin = img_origin.permute(0,2,1).reshape(N*H*W, T) # (N*H*W) * T 
        img_origin = img_origin[:,:self.args.his_len] # only use history data

        img_spec = torch.fft.rfft(img_origin, n = img_origin.shape[-1], norm = "ortho", dim = -1) # [N, K] K = T//2 + 1
        img_spec = img_spec.reshape(N, H, W, self.args.his_len//2+1)

        img_spec = torch.cat((img_spec.real, img_spec.imag), dim = -1) # [N, H, W, 2(his_len//2+1)]

        img_spec = self.spec_liner(img_spec)

        img_tmp = img_origin.unsqueeze(1)

        img_tmp = self.temporaltokenConv(img_tmp).permute(0,2,1) # N * T * Embed

        img_mark = img_mark[:,:self.args.his_len].unsqueeze(dim=1).repeat(1,H*W,1, 1).reshape(N*H*W, self.args.his_len, 2)

        temporal_emb = DataEmbedding.temporal_emb(img_mark, data)

        assert img_tmp.shape == temporal_emb.shape
        img_tmp += temporal_emb

        img_tmp = torch.cat([self.temporal_attn_encoder(img_tmp[index:index+H*W]) for index in range(0, img_tmp.shape[0], H*W)],axis=0)[:,0]
        
        img_tmp = img_tmp.reshape(N, H, W, img_tmp.shape[-1])

        if 'Graph' not in data:
            if patch_size == 1:
                img_spec = self.prompt_spatial_patch_f_1(img_spec.permute(0,3,1,2))
                img_tmp = self.prompt_spatial_patch_t_1(img_tmp.permute(0,3,1,2))
            elif patch_size == 2:
                img_spec = self.prompt_spatial_patch_f_2(img_spec.permute(0,3,1,2))
                img_tmp = self.prompt_spatial_patch_t_2(img_tmp.permute(0,3,1,2))
            elif patch_size == 4:
                img_spec = self.prompt_spatial_patch_f_4(img_spec.permute(0,3,1,2))
                img_tmp = self.prompt_spatial_patch_t_4(img_tmp.permute(0,3,1,2))
        else:
            # patchify
            max_len = max([len(i) for i in node_split])
            n_group = len(node_split)

            img_tmp = torch.cat([torch.mean(torch.gather(img_tmp, 1, group.view(1, group.shape[0], 1, 1).expand(img_tmp.shape[0], group.shape[0], img_tmp.shape[2], img_tmp.shape[3]).to(img_tmp).long()),dim=1,keepdim=True) for group in node_split],dim=1).squeeze(dim=2)

            img_spec = torch.cat([torch.mean(torch.gather(img_spec, 1, group.view(1, group.shape[0], 1, 1).expand(img_spec.shape[0], group.shape[0], img_spec.shape[2], img_spec.shape[3]).to(img_tmp).long()),dim=1,keepdim=True) for group in node_split],dim=1).squeeze(dim=2)
            
        return img_tmp, img_spec


    # Create target key padding mask
    def create_padding_mask(self, seq_lengths, max_len):
        padding_mask = torch.zeros((len(seq_lengths), max_len), dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            padding_mask[i, length:] = True
        return padding_mask

    def forward(self, imgs, mask_ratio=0.5, mask_strategy='causal',seed=520, data='none',  mode='backward',topo = None, subgraphs = None, patch_size = 100):
        '''
        backward: 没有特定evaluation约束，forward: 有特定evaluation约束
        '''
        imgs, imgs_mark = imgs

        if self.args.is_prompt == 1:
            if 'Graph' not in data:
                img_tmp, img_spec = self.adpative_graph(imgs, imgs_mark, self.Embedding_patch, data=data, patch_size = patch_size)
            else:
                img_tmp, img_spec = self.adpative_graph(imgs, imgs_mark, self.Embedding_patch_graph, data=data, node_split = subgraphs, patch_size = patch_size)
        else:
            img_tmp = None
            img_spec = None

        T, H, W = imgs.shape[2:]
        latent, mask, ids_restore, input_size, TimeEmb, prompt = self.forward_encoder(imgs, imgs_mark, mask_ratio, mask_strategy, seed=seed, data=data, mode=mode, prompt = {'t': img_tmp, 'f':img_spec,'topo':topo}, patch_size = patch_size, split_nodes=subgraphs)

        pred = self.forward_decoder(latent, imgs_mark, mask, ids_restore, mask_strategy, TimeEmb, input_size = input_size, data = data, prompt_graph = prompt)  # [N, L, p*p*1]
        L = pred.shape[1]

        if 'Graph' not in data:
            if patch_size == 1:
                pred = self.head_layer_1(pred)
            elif patch_size == 2:
                pred = self.head_layer_2(pred)
            elif patch_size == 4:
                pred = self.head_layer_4(pred)

        else:
            seq_lengths = [len(i) for i in subgraphs] 
            max_len = max(seq_lengths)

            if 'GraphBJ' in data:
                pred = self.pred_model_linear_GraphBJ(pred).reshape(pred.shape[0],T//self.args.t_patch_size,len(subgraphs),self.args.t_patch_size, -1).permute(0,1,3,2,4)
            elif 'GraphSH' in data:
                pred = self.pred_model_linear_GraphSH(pred).reshape(pred.shape[0],T//self.args.t_patch_size,len(subgraphs),self.args.t_patch_size, -1).permute(0,1,3,2,4)
            elif 'GraphNJ' in data:
                pred = self.pred_model_linear_GraphNJ(pred).reshape(pred.shape[0],T//self.args.t_patch_size,len(subgraphs),self.args.t_patch_size, -1).permute(0,1,3,2,4)
           
            pred = pred.reshape(pred.shape[0],T, len(subgraphs), -1)

            pred = torch.cat([pred[:,:,g,:seq_lengths[g]] for g in range(pred.shape[2])],dim=2)

            target = imgs.squeeze(dim=(1,4))

            target = torch.cat([torch.gather(target, 2, group.view(1, 1, group.shape[0]).expand(target.shape[0], target.shape[1], group.shape[0]).to(target).long()) for group in subgraphs],dim=2)

        if 'Graph' not in data:
            loss1, loss2, target = self.forward_loss(imgs, pred, mask, patch_size)

        else:
            loss1, loss2, target, mask = self.graph_loss(pred, target)

        return loss1, loss2, pred, target, mask


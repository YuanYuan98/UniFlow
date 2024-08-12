import torch
import numpy as np



def causal_masking(x, mask_ratio, T, mask_strategy):
    N, L, D = x.shape
    x = x.reshape(N, T, L//T, D)
    N, T, L, C = x.shape

    len_keep = int(T * (1 - mask_ratio))

    if mask_strategy == 'causal':
        # noise = torch.ones(N, T, device=x.device)  # noise in [0, 1]
        # noise[:,:len_keep] = 0
        noise = torch.arange(T).unsqueeze(dim=0).repeat(N,1)
        noise = noise.to(x)
    
    elif mask_strategy == 'frame':
        noise = torch.rand(N, T, device=x.device)

    ids_shuffle = torch.argsort(
        noise, dim=1
    )  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(2).unsqueeze(-1).repeat(1, 1, L, D))

    assert (x_masked == x[:,:len_keep]).all()

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, T, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore.unsqueeze(2).repeat(1,1,L)).reshape(N,-1)

    ids_keep = ids_keep.unsqueeze(2).repeat(1,1,L).reshape(N,-1)
    x_masked = x_masked.reshape(N, -1, x_masked.shape[-1])

    return x_masked, mask, ids_restore, ids_keep

def causal_restore(x, ids_restore, N, T, H, W, C, mask_token):
    x = x.reshape(N, -1, H * W, x.shape[-1])
    mask_tokens = mask_token.repeat(N, T - x.shape[1] , H * W, 1)
    x_ = torch.cat([x, mask_tokens], dim=1) 
    x_ = torch.gather(x_, dim=1, index = ids_restore.unsqueeze(2).unsqueeze(-1).repeat(1, 1, H * W, x_.shape[-1]))
    x = x_.view([N, T * H * W, C])
    return x

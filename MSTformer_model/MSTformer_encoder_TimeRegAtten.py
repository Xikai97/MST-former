import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from MSTformer_model.attn_TimeRegAtten import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil, floor, sqrt
import ipdb

class PatchMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    New revision: segment merging represents extract image feature at different scales now.
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        scale_ratio = floor(sqrt(win_size))
        self.multi_scale_embed = nn.Conv2d(in_channels=d_model, 
                                       out_channels=d_model,
                                       kernel_size=scale_ratio,
                                       stride=scale_ratio)   # scale_ratio is the number of adjcent patches to be merged
        # self.linear_trans = nn.Linear(win_size * d_model, d_model)
        # self.norm = norm_layer(win_size * d_model)
        self.norm = norm_layer(d_model)

    def forward(self, x):
        """
        x: B, L, N, d_model
        """
        batch_size, seq_len, patch_num, d_model = x.shape
        pad_num = patch_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)
        
        h_1 = floor(sqrt(patch_num))
        # w_1 = h_1
        x_reshape = rearrange(x, "b l (h w) d -> (b l) d h w", h=h_1)
        x_reshape = self.multi_scale_embed(x_reshape)  # (b l) d h w -> (b l) d h_1 w_1
        x_reshape = x_reshape.flatten(start_dim=2).transpose(-1, -2)  # (b l) d h_1 w_1 -> (b l) (h_1 w_1) d
        x = rearrange(x_reshape, "(b l) n d -> b l n d", l=seq_len)
        x = self.norm(x)
        return x

class scale_block(nn.Module):
    
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = PatchMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(d_model, n_heads, d_ff, dropout))
    
    def forward(self, x, Tmatrix=None):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            x = layer(x, Tmatrix=Tmatrix)        
        
        return x

class Encoder(nn.Module):
    '''
    The Encoder of MSTformer.
    '''
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout))

    def forward(self, x, Tmatrix=None):
        encode_x = []
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x, Tmatrix=Tmatrix)
            encode_x.append(x)

        return encode_x
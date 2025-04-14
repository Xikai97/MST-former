import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import ipdb

from math import sqrt

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1, causality=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.relu = nn.ReLU()
        self.causality = causality
        
    def forward(self, queries, keys, values, Tmatrix=None, debug=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        
        if debug:
            ipdb.set_trace()
            
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if Tmatrix is not None:
            Tmatrix = repeat(Tmatrix, 'b l s-> b h l s', h=H)
            A = self.dropout(torch.softmax(scale * self.relu(scores) * Tmatrix, dim=-1))
        else:
            A = self.dropout(torch.softmax(scale * scores, dim=-1))

        if self.causality is True:
            padding_val = -2 ** 32 + 1
            # torch.nn.MultiheadAttention()
            # mask = torch.triu(torch.full((L, S), float('-inf'), device=queries.device), diagonal=1)
            mask = torch.tril(torch.full((L, S), float(1), device=queries.device), diagonal=0)
            mask = repeat(mask, 'l s -> b h l s', b=B, h=H)
            paddings = torch.ones_like(mask) * padding_val
            A_out = torch.where(mask>0, A, paddings)
        else:
            A_out = A

        V = torch.einsum("bhls,bshd->blhd", A_out, values)
        
        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1, causality=True):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout, causality=causality)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, Tmatrix=None, debug=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        if debug:
            ipdb.set_trace()
            # print('shape of query in cross atten:{}'.format(queries.shape))
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            Tmatrix=Tmatrix,
            debug=debug
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)
    
    
class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Seq_len(l), Patch_num(n), d_model]   # b l n d_model
    '''
    # def __init__(self, seg_num, d_model, n_heads, d_ff = None, dropout=0.1):
    def __init__(self, d_model, n_heads, d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.spatio_attention = AttentionLayer(d_model, n_heads, dropout = dropout, causality=False)
        self.temporal_attention = AttentionLayer(d_model, n_heads, dropout = dropout, causality=True)
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), 
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), 
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x, Tmatrix=None):
        #Cross Image Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        B, L, N, D = x.shape
        time_in = rearrange(x, 'b l n d_model -> (b l) n d_model')
        time_enc = self.spatio_attention(time_in, time_in, time_in, Tmatrix=None)  # time-sensitive attention is not proper for spatio dimension 
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))  
        dim_in = self.norm2(dim_in) 
        dim_send = rearrange(dim_in, '(b l) n d_model -> (b n) l d_model', b = batch)
        if Tmatrix is not None:
            Tmatrix_1 = repeat(Tmatrix, 'b l s -> b n l s', n = N)
            Tmatrix_2 = rearrange(Tmatrix_1, 'b n l s -> (b n) l s')
        else:
            Tmatrix_2 = Tmatrix
        dim_receive = self.temporal_attention(dim_send, dim_send, dim_send, Tmatrix=Tmatrix_2)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc)) 
        dim_enc = self.norm4(dim_enc)  

        final_out = rearrange(dim_enc, '(b n) l d_model -> b l n d_model', b = batch)

        return final_out

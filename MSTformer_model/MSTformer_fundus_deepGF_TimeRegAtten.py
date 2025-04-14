import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
sys.path.append('add path to your project')
from MSTformer_model.MSTformer_encoder_TimeRegAtten import Encoder
from MSTformer_model.MSTformer_decoder_TimeRegAtten import Decoder
from MSTformer_model.attn_TimeRegAtten import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil
import ipdb
import numpy as np

class MSTformer(nn.Module):
    def __init__(self, img_len, patch_size=14, win_size = 4, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, block_depth=1,
                dropout=0.0, device=torch.device('cuda:0'), useTime = True, useAttenMap=True):
        super(MSTformer, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.img_len = img_len
        self.merge_win = win_size

        self.device = device

        self.useTime = useTime

        self.useAttenMap = useAttenMap

        self.block_depth = block_depth
        
        # Patch_num calculation (n)
        self.patch_num = (224 // patch_size) * (224 // patch_size)
        
        # patch embedding for Fundus image
        self.patch_embeddings_img = nn.Conv2d(in_channels=3, 
                                       out_channels=d_model,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # Embedding
        # self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, img_len, self.patch_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = self.block_depth, dropout = dropout)
        
        # Decoder
        self.decoder = Decoder(2, e_layers + 1, d_model, n_heads, d_ff, dropout) # choose the output with size (B x L x 1 x 2)
        # self.final_linear_pred = nn.Linear(self.img_len, 2) # whether the future is glaucoma or non-glaucoma
        
        self.feat_label_convert = nn.AdaptiveAvgPool2d((2, 1))
        # self.feat_label_convert = nn.AdaptiveAvgPool2d((1, 1))
        self.prob_norm = nn.LayerNorm(d_model)

    
    def pos_embed_enc(self, input, time, max_time=50):
        B, L, N, D = input.shape
        year_pos_all = np.array([[pos / np.power(10000, (i-i%2)/D) for i in range(D)] for pos in range(max_time)])
        year_pos_all = torch.from_numpy(year_pos_all)
        year_pos_all = year_pos_all.to(time.device, dtype=torch.float)
        year_pos_all = repeat(year_pos_all, 'max_t n -> b max_t n', b=B)
        index = repeat(time.long(), 'b l -> b l d', d=D)
        # ipdb.set_trace()
        year_pos = torch.gather(year_pos_all,dim=1,index=index)
        year_pos = repeat(year_pos, 'b l d -> b l n d', n=N)
        
        patch_pos_all = np.array([[patch_idx / np.power(10000, (i-i%2)/D) for i in range(D)] for patch_idx in range(N)])
        patch_pos_all = torch.from_numpy(patch_pos_all)
        patch_pos_all = patch_pos_all.to(time.device, dtype=torch.float)
        patch_pos = repeat(patch_pos_all, 'n d -> b l n d', l=L, b=B)
        final_pos_encoding = year_pos + patch_pos
        
        return final_pos_encoding
    
    
    def pos_embed_dec(self, time, max_time=50):
        B, L = time.shape
        year_pos_all = np.array([[pos / np.power(10000, (i-i%2)/self.d_model) for i in range(self.d_model)] for pos in range(max_time)])
        year_pos_all = torch.from_numpy(year_pos_all)
        year_pos_all = year_pos_all.to(time.device, dtype=torch.float)
        year_pos_all = repeat(year_pos_all, 'max_t d -> b max_t d', b=B)
        index = repeat(time.long(), 'b l -> b l d', d=self.d_model)
        year_pos = torch.gather(year_pos_all,dim=1,index=index)
        year_pos = repeat(year_pos, 'b l d -> b l dummy d', dummy=1)
        return year_pos
        
        
    # def forward(self, input, atten_map, polar_map, year, seq_label):
    def forward(self, input, atten_map=None, polar_map=None, year=None, future_year_tag=None, seq_label=None, Tmatrix=None):
        # if self.useAttenMap is True:
        #     input = input * atten_map # overlap attention image
        B, L, C, H, W = input.shape
        embedding_img = rearrange(input, "b l c h w -> (b l) c h w")
        patch_emb_img = self.patch_embeddings_img(embedding_img)
        patch_emb_img = patch_emb_img.flatten(start_dim=2).transpose(-1, -2)
        patch_emb_img_seq = rearrange(patch_emb_img, "(b l) n d -> b l n d", l=L)
        x_seq = patch_emb_img_seq
        
        if self.useTime and year is not None:
            # year_pos = repeat(year, 'b l -> b l n d', n=self.patch_num, d=self.d_model)
            year_pos = self.pos_embed_enc(x_seq, year)
            x_seq += year_pos
            x_seq += self.enc_pos_embedding
            
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq, Tmatrix=Tmatrix)
        
        enc_last = self.prob_norm(enc_out[-1])
        
        enc_pred = self.feat_label_convert(enc_last).reshape(B,L,2)

        if seq_label is None:
            pos_prob = enc_pred
            pesudo_seq_label_prob = F.softmax(enc_pred,dim=2)
            _, pesudo_seq_label = torch.max(pesudo_seq_label_prob,dim=2)
        else:
            pesudo_seq_label = seq_label
        
        seq_label_embed = repeat(seq_label.unsqueeze(-1), 'b l dummy -> b l dummy d_less', d_less=self.d_model)       
        dec_in = torch.cat((seq_label_embed,enc_out[-1]), dim=2)
        if self.useTime and year is not None:
            dec_pos_embedding = self.pos_embed_enc(dec_in, year)
            dec_in = dec_in + dec_pos_embedding
        
        predict_y_prob = self.decoder(dec_in, enc_out, Tmatrix=Tmatrix)
        
        if seq_label is None:
            return predict_y_prob, pos_prob, pesudo_seq_label
        else:
            return predict_y_prob, pesudo_seq_label
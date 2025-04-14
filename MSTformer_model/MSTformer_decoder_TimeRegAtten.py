import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from MSTformer_model.attn_TimeRegAtten import FullAttention, AttentionLayer, TwoStageAttentionLayer
import ipdb

class DecoderLayer(nn.Module):
    '''
    The decoder layer of MSTformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # self.self_attention = TwoStageAttentionLayer(d_model, n_heads, d_ff, dropout)    # keep decoder TSA
        # self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout, causality=False)
        self.self_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout, causality=False)  
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), 
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, 2)

    def forward(self, x, cross, Tmatrix=None):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''
        batch, L, N, D = x.shape
        x_in = rearrange(x, 'b l dummy d_model -> (b dummy) l d_model')
        if Tmatrix is not None:
            Tmatrix_expand = repeat(Tmatrix, 'b l s-> (repeat b) l s', repeat=N)
        else:
            Tmatrix_expand = None
        x_MSA = self.self_attention(x_in, x_in, x_in, Tmatrix=Tmatrix_expand, debug=False)   # keep decoder TSA 
        x_mid = x_in + self.dropout1(x_MSA)
        x_mid = self.norm1(x_mid)
        x_mid = rearrange(x_mid, '(b dummy) l d_model -> (b l) dummy d_model', dummy=N)
        cross = rearrange(cross, 'b l n d_model -> (b l) n d_model')
        tmp = self.cross_attention(x_mid, cross, cross, debug=False) 
        x_mid = x_mid + self.dropout2(tmp)  
        y = x_out = self.norm2(x_mid) 
        y = self.MLP1(y) 
        dec_output = self.norm3(x_out+y)
        
        dec_output = rearrange(dec_output, '(b l) dummy d_model -> b l dummy d_model', b = batch)
        layer_predict = self.linear_pred(dec_output)  # b l n d_model
        # layer_predict = rearrange(layer_predict, 'b l dummy cls -> b l (dummy cls)')
        # layer_predict = layer_predict[:,:,0,:].squeeze(2) # the first version
        layer_predict = layer_predict.mean(2) # change 2

        return dec_output, layer_predict

class Decoder(nn.Module):
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout))

    def forward(self, x, cross, Tmatrix=None):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc, Tmatrix=Tmatrix)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        
        # final_predict = rearrange(final_predict, 'b (l seg_num) seg_len -> b (seg_num seg_len) l', l = ts_d)

        return final_predict


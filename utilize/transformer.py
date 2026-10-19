from utilize.modules import *
import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self, src_feature_size, trg_feature_size, embed_size= 512, num_layers= 6, forward_expansion= 4, heads= 8, dropout= 0, device= "cuda", max_length= 100):
        super(Transformer, self).__init__()
        self.encoder= Encoder(src_feature_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.decoder= Decoder(trg_feature_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.device= device
    ### For sensor data, we donnot replace 0 value for masking.
    def make_src_mask(self, src):
        src_mask= (src != 0).any(dim= -1)
        src_mask= src_mask.unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    def make_trg_mask(self, trg):
        n, trg_len, _= trg.shape
        trg_mask= torch.tril(torch.ones((trg_len, trg_len), device= trg.device, dtype= torch.bool)).expand(n, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    def forward(self, src, trg):
        src_mask= None
        trg_mask= self.make_trg_mask(trg)
        encoder_src= self.encoder(src, src_mask)
        out= self.decoder(trg, encoder_src, src_mask, trg_mask)
        return out
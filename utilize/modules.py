import torch.nn as nn
import torch, math

class PositionalEncoding(nn.Module): 
    def __init__(self, embed_size, max_length= 5000): 
        super(PositionalEncoding, self).__init__() 
        pe= torch.zeros(max_length, embed_size)
        position= torch.arange(0, max_length, dtype= torch.float).unsqueeze(1)
        ### div-term represents different frequency.
        div_term= torch.exp(torch.arange(0, embed_size, 2, dtype= torch.float)* (-math.log(10000.0)/ embed_size)) 
        pe[:, 0::2]= torch.sin(position* div_term)
        pe[:, 1::2]= torch.cos(position* div_term)
        pe= pe.unsqueeze(0) 
        self.register_buffer("pe", pe) 
    def forward(self, x):
        sequence_len= x.size(1)
        assert sequence_len<= self.pe.size(1), f"Sequence length {sequence_len} exceeds PE max length {self.pe.size(1)}"
        return x+ self.pe[:, :sequence_len, :]

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size= embed_size
        self.heads= heads
        self.head_dim= embed_size// heads
        assert (self.head_dim* heads== embed_size), "embedding size needs to be divisible by heads"
        self.v= nn.Linear(embed_size, embed_size)
        self.k= nn.Linear(embed_size, embed_size)
        self.q= nn.Linear(embed_size, embed_size)
        self.fc= nn.Linear(embed_size, embed_size)
    def forward(self, q, k, v, mask):
        n= q.shape[0]
        v_len, k_len, q_len= v.shape[1], k.shape[1], q.shape[1]
        v_vector= self.v(v).reshape(n, v_len, self.heads, self.head_dim)
        k_vector= self.k(k).reshape(n, k_len, self.heads, self.head_dim)
        q_vector= self.q(q).reshape(n, q_len, self.heads, self.head_dim)
        qk= torch.einsum("nqhd, nkhd->nhqk", [q_vector, k_vector])
        if mask is not None:
            qk= qk.masked_fill(mask== 0, float("-1e20"))
        attention= torch.softmax(qk/ (self.head_dim** (1/ 2)), dim= 3)
        out= torch.einsum("nhql, nlhd->nqhd", [attention, v_vector]).reshape(n, q_len, self.heads* self.head_dim)
        out= self.fc(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.SelfAttention= SelfAttention(embed_size, heads)
        self.norm1= nn.LayerNorm(embed_size)
        self.norm2= nn.LayerNorm(embed_size)
        self.feed_forward= nn.Sequential(
            nn.Linear(embed_size, forward_expansion* embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion* embed_size, embed_size)
        )
        self.dropout= nn.Dropout(dropout)
    def forward(self, q, k, v, mask):
        attention= self.SelfAttention(q, k, v, mask)
        x= self.dropout(self.norm1(attention+ q))
        forward= self.feed_forward(x)
        out= self.dropout(self.norm2(forward+ x))
        return out

class Encoder(nn.Module):
    def __init__(self, feature_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Encoder, self).__init__()
        self.embed_size= embed_size
        self.device= device
        self.num_layers= num_layers
        self.feature_embeding= nn.Linear(feature_size, embed_size)
        self.position_encoding= PositionalEncoding(embed_size, max_length)
        self.layers= nn.ModuleList(
            [TransformerBlock(
                embed_size,
                heads,
                dropout= dropout,
                forward_expansion= forward_expansion
            )
            for _ in range(num_layers)
            ]
        )
        self.dropout= nn.Dropout(dropout)
    def forward(self, x, mask):
        out= self.feature_embeding(x) 
        out= self.position_encoding(out) 
        out= self.dropout(out) 
        for layer in self.layers: 
            out= layer(out, out, out, mask) 
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm= nn.LayerNorm(embed_size)
        self.attention= SelfAttention(embed_size, heads= heads)
        self.transformer_block= TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout= nn.Dropout(dropout)
    def forward(self, x, k, v, src_mask, trg_mask):
        attention= self.attention(x, x, x, trg_mask)
        query= self.dropout(self.norm(attention+ x))
        # q, k, v, mask
        out= self.transformer_block(query, k, v, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, feature_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device= device
        self.feature_embeding= nn.Linear(feature_size, embed_size)
        self.position_encoding= PositionalEncoding(embed_size, max_length)
        self.layers= nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc= nn.Linear(embed_size, feature_size)
        self.dropout= nn.Dropout(dropout)
    def forward(self, x, encoder_out, src_mask, trg_mask):
        x= self.feature_embeding(x) 
        x= self.position_encoding(x) 
        x= self.dropout(x)
        for layer in self.layers:
            ### k, v from encoder and q from decoder.
            x= layer(x, encoder_out, encoder_out, src_mask, trg_mask)
        out= self.fc(x)
        return out
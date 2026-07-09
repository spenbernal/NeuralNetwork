import numpy as np
import layers as l
import nn as nn

class EncoderBlock:
    def __init__(self, mha: l.MultiHeadAttention, ln_1: l.LayerNorm, ffn: nn.Sequential, ln_2: l.LayerNorm)-> None:
        self.mha = mha
        self.ln_1 = ln_1
        self.ffn = ffn
        self.ln_2 = ln_2
        
    def forward(self, X):
        # X: N, T, D
        # self attention in encoderblock
        Z = self.ln_1.forward(self.mha.forward(X, X, X) + X)
        E = self.ln_2.forward(self.ffn.forward(Z) + Z)
        return E
    
    def backwards(self, delta):
        d_E = self.ln_2.backwards(delta)
        d_Z = self.ffn.backwards(d_E) + d_E
        d_A = self.ln_1.backwards(d_Z)
        d_q, d_k, d_v = self.mha.backwards(d_A)
        d_X = d_q + d_k + d_v + d_A
        return d_X
    
    def zero_grad(self):
        self.mha.zero_grad()
        self.ffn.zero_grad()

    def update(self, eta):
        self.mha.update(eta)
        self.ffn.update(eta)
class Encoder:
    def __init__(self, embed: l.Embedding, pos_encode: l.PositionalEncoding, encoder_blocks: list[EncoderBlock]) -> None:
        self.embed = embed
        self.pos_encode = pos_encode
        self.encoder_blocks = encoder_blocks
        
    def forward(self, tokens):
        # X: N, T tokens
        X = self.pos_encode.forward(self.embed.forward(tokens))
        for encoder_block in self.encoder_blocks:
            X = encoder_block.forward(X)
        return X
    
    def backwards(self, delta):
        for encoder_block in reversed(self.encoder_blocks):
            delta = encoder_block.backwards(delta)
            
        delta_pos = self.pos_encode.backwards(delta)
        self.embed.backwards(delta_pos)
        
    def update(self, eta):
        for encoder_block in self.encoder_blocks:
            encoder_block.update(eta)
        self.embed.update(eta)
    
    def zero_grad(self):
        for encoder_block in self.encoder_blocks:
            encoder_block.zero_grad()
        self.embed.zero_grad()
    
class DecoderBlock:
    
    def __init__(self, mha_1: l.MultiHeadAttention, mha_2: l.MultiHeadAttention, 
                 ln_1: l.LayerNorm, ln_2: l.LayerNorm, ln_3: l.LayerNorm, ffn: nn.Sequential)-> None:
        self.mha_1 = mha_1
        self.mha_2 = mha_2
        self.ln_1 = ln_1
        self.ln_2 = ln_2
        self.ln_3 = ln_3
        self.ffn = ffn
    
    def forward(self, Y, E):
        A = self.ln_1.forward(self.mha_1.forward(Y, Y, Y) + Y)
        Z = self.ln_2.forward(self.mha_2.forward(A, E, E) + A)
        D = self.ln_3.forward(self.ffn.forward(Z) + Z)
        return D
        
    def backwards(self, delta):
        D_l_3 = self.ln_3.backwards(delta)
        D_Z = self.ffn.backwards(D_l_3) + D_l_3
        D_l_2 = self.ln_2.backwards(D_Z)
        D_dec, D_enc_k, D_enc_v = self.mha_2.backwards(D_l_2)
        D_dec += D_l_2
        D_enc = D_enc_k + D_enc_v
        D_l_1 = self.ln_1.backwards(D_dec)
        d_q, d_k, d_v = self.mha_1.backwards(D_l_1)
        D_X_dec = d_q + d_k + d_v + D_l_1
        return D_X_dec, D_enc
    
    def zero_grad(self):
        self.mha_1.zero_grad()
        self.mha_2.zero_grad
        self.ffn.zero_grad()

    def update(self, eta):
        self.mha_1.update(eta)
        self.mha_2.update(eta)
        self.ffn.update(eta)

        
class Decoder:
    def __init__(self, embed: l.Embedding, pos_encode: l.PositionalEncoding, decoder_blocks: list[DecoderBlock]) -> None:
        self.embed = embed
        self.pos_encode = pos_encode
        self.decoder_blocks = decoder_blocks
        
    def forward(self, tokens, E):
        # X: N, T tokens
        self.E = E
        Y = self.pos_encode.forward(self.embed.forward(tokens))
        for decoder_block in self.decoder_blocks:
            Y = decoder_block.forward(Y, E)
        return Y
    
    def backwards(self, delta):
        delta_dec = delta
        delta_enc_total = np.zeros_like(self.E)
        for decoder_block in reversed(self.decoder_blocks):
            delta_dec, delta_enc = decoder_block.backwards(delta_dec)
            delta_enc_total += delta_enc
            
        delta_pos = self.pos_encode.backwards(delta_dec)
        self.embed.backwards(delta_pos)
        return delta_enc_total
        
    def update(self, eta):
        for decoder_block in self.decoder_blocks:
            decoder_block.update(eta)
        self.embed.update(eta)
    
    def zero_grad(self):
        for decoder_block in self.decoder_blocks:
            decoder_block.zero_grad()
        self.embed.zero_grad()
        
class Transformer:
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, X, y):
        E = self.encoder.forward(X)
        out = self.decoder.forward(y, E)
        return out
    
    def backwards(self, delta):
        D_enc = self.decoder.backwards(delta)
        self.encoder.backwards(D_enc)
        
    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
    
    def update(self, eta):
        self.encoder.update(eta)
        self.decoder.update(eta)
        
        
        
    
    

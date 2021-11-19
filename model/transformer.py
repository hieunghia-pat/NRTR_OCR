import torch
import torch.nn as nn
import copy
from data_utils.utils import subsequent_mask
from model.utils import PositionwiseFeedForward
from model.modality_transformer import ModalityTransformer
from model.embedding import Embeddings, PositionalEncoding
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.attention import MultiHeadedAttention
from model.generator import Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generator(self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask))

    def get_predictions(self, images, src_mask, vocab, max_len):
        encoded_features = self.encode(images, src_mask)
        batch_size = images.shape[0]
        
        ys = torch.ones(size=(batch_size, 1)).fill_(vocab.stoi[vocab.sos_token]).long().cuda()
        for it in range(max_len):
            tgt_mask = subsequent_mask(ys.shape[-1]).long().cuda()
            outs = self.decode(encoded_features, src_mask, ys, tgt_mask)
            outs = self.generator(outs[:, -1])
            outs = outs.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, outs], dim=1)

        # refine the outputs
        for batch_idx in range(ys.size(0)):
            idx = 0
            while ys[batch_idx, idx] != vocab.eos_idx and idx < ys.size(1)-1:
                idx += 1
            ys[batch_idx, idx+1:] = vocab.padding_idx
        
        return ys

def make_model(vocab_size, N=4, 
               d_model=256, d_ff=1024, image_height=64, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(ModalityTransformer(imChannel=3, imgHeight=image_height, d_model=d_model)),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model
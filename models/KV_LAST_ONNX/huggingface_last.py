import copy
import math
import time
from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from .transformers import TrOCRConfig, TrOCRForCausalLM
from .encoder import Encoder
from .line_pos_enc import LineIntroducer

class LASTDecoder(nn.Module):
    def __init__(self, config: TrOCRConfig, nline: int):
        super().__init__()
        self.nline = nline
        self.config = config
        self.word_embed = nn.Sequential(nn.Embedding(config.vocab_size, config.d_model), 
                                        nn.LayerNorm(config.d_model))
        self.model = TrOCRForCausalLM(config)
        self.line_pos = LineIntroducer(config.d_model)
    
    def forward(self, encoder_input, task_seq_input,
                task_seq_pos, task_seq_li, 
                past_key_values):
        tgt=task_seq_input
        tgte = self.word_embed(tgt)
        inline_pos=task_seq_pos

        outline_pos = self.line_pos(task_seq_li)
        tgte+=outline_pos+inline_pos

        out = self.model(
            attention_mask = None,
            inputs_embeds = tgte,
            encoder_hidden_states = encoder_input,
            past_key_values = past_key_values,
            )
        return out


class Huggingface_LAST(nn.Module):
    def __init__(self, config: TrOCRConfig, nline: int, growth_rate, num_layers):
        super().__init__()
        self.encoder = Encoder(
            d_model=config.d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = LASTDecoder(
            config=config,
            nline=nline
        )
        self.config = config


if __name__ == "__main__":
    decoder = LASTDecoder(TrOCRConfig(vocab_size=451, max_length=80, d_model=256,
                                           decoder_layers=3,
                                           decoder_attention_heads=8,
                                           decoder_ffn_dim=1024, 
                                           max_position_embeddings=100,
                                           dropout=0.3,
                                           ),
                            nline=16)                               

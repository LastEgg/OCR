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
from .datamodule import BiMultinline, Plainline

class LASTDecoder(nn.Module):
    def __init__(self, config: TrOCRConfig, nline: int):
        super().__init__()
        self.nline = nline
        self.config = config
        self.word_embed = nn.Sequential(nn.Embedding(config.vocab_size, config.d_model), 
                                        nn.LayerNorm(config.d_model))
        self.model = TrOCRForCausalLM(config)
        self.line_pos = LineIntroducer(config.d_model)
    
    def forward(self, src, src_mask, task_seq, past_key_values):
        tgt=task_seq.input
        tgte = self.word_embed(tgt)
        inline_pos=task_seq.pos

        if not task_seq.task_name.startswith('plain'):
            outline_pos = self.line_pos(task_seq.li)
            tgte+=outline_pos+inline_pos
        else:
            tgte+=inline_pos
        out = self.model(
            attention_mask = None,
            inputs_embeds = tgte,
            encoder_hidden_states = src,
            past_key_values = past_key_values,
            )
        return out
    
    def bar(self, src, src_mask, task_name):
        BAR=BiMultinline(task_name, src.device, self.config.d_model, self.nline)
        ti=0
        n=0
        past_key_values = None
        while not BAR.is_done():
            BAR.make_input()
            st=time.perf_counter()
            outputs = self.forward(src, src_mask, BAR.batch, past_key_values)
            new_char_outputs= outputs.logits
            past_key_values = outputs.past_key_values
            # print("================")
            # print(len(past_key_values))
            # print(past_key_values[0][0].shape)
            # print(past_key_values[0][1].shape)
            # print(past_key_values[0][2].shape)
            # print(past_key_values[0][3].shape)
            et=time.perf_counter()
            ti+=et-st
            BAR.update(new_char_outputs)
            n+=1
        ans=BAR.return_ans()
        return ans, ti, n

    def ar(self, src, src_mask, task_name):
        AR=Plainline(task_name, src.device, self.config.d_model)
        ti=0
        n=0
        past_key_values = None
        while not AR.is_done():
            AR.make_input()
            st=time.perf_counter()
            outputs = self.forward(src, src_mask, AR.batch, past_key_values)
            new_char_outputs= outputs.logits
            past_key_values = outputs.past_key_values
            et=time.perf_counter()
            ti+=et-st
            AR.update(new_char_outputs)
            n+=1
        ans=AR.return_ans()
        return ans, ti, n


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
    
    # def forward(self, imgs, img_mask, task_seq):
    #     feature, f_mask = self.encoder(imgs, img_mask)
    #     out = self.decoder(feature, f_mask, task_seq, None)
    #     return out

    def bar(self, img, img_mask, task_name):
        st=time.perf_counter()
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        et=time.perf_counter()
        ans, t, n = self.decoder.bar(feature, mask, task_name)
        return ans, t+et-st, n
    
    def ar(self, img, img_mask, task_name):
        st=time.perf_counter()
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        et=time.perf_counter()
        ans, t, n = self.decoder.ar(feature, mask, task_name)
        return ans, t+et-st, n


if __name__ == "__main__":
    decoder = LASTDecoder(TrOCRConfig(vocab_size=451, max_length=80, d_model=256,
                                           decoder_layers=3,
                                           decoder_attention_heads=8,
                                           decoder_ffn_dim=1024, 
                                           max_position_embeddings=100,
                                           dropout=0.3,
                                           ),
                            nline=16)                               

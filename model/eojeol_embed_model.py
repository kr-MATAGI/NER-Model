import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from transformers import ElectraModel, ElectraPreTrainedModel

from model.crf_layer import CRF
from model.modeling_bert2 import Encoder

#===============================================================
class Eojeol_Embed_Model(ElectraPreTrainedModel):
#===============================================================
    def __init__(self, config):
        # init
        super().__init__(config)
        self.config = config

        self.max_seq_len = config.max_seq_len
        self.num_ne_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels
        self.pos_embed_out_dim = 128
        self.dropout_rate = 0.33
        self.max_eojeol_len = 50

        # for encoder
        self.d_model_size = config.hidden_size + (self.pos_embed_out_dim * 3)  # [768 + 128 * 3] = 1152
        self.enc_config = copy.deepcopy(config)
        self.enc_config.num_hidden_layers = 4
        self.enc_config.hidden_size = self.d_model_size
        self.enc_config.ff_dim = self.d_model_size
        self.enc_config.act_fn = "gelu"
        self.enc_config.dropout_prob = 0.1
        self.enc_config.num_heads = 12  # origin 12

        # structure
        self.electra = ElectraModel.from_pretrained(config.model_name, config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # POS
        self.eojeol_pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.eojeol_pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.eojeol_pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        # self.eojeol_pos_embedding_4 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        # One-Hot Embed
        self.one_hot_embedding = nn.Embedding(self.max_seq_len, self.max_eojeol_len)

        # Transformer Encoder
        self.encoder = Encoder(self.enc_config)

        # Classifier
        self.linear = nn.Linear(self.d_model_size, config.num_labels)

        # CRF
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    def _make_ont_hot_embeddig(
            self,
            last_hidden,
            eojeol_ids,
    ):
        # [64, 128, 768]
        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        new_eojeol_info_matrix = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for batch_idx in range(batch_size):
            cur_idx = 0
            eojeol_info_matrix = torch.zeros(max_seq_len, dtype=torch.long)
            for eojeol_idx, eojeol_tok_cnt in enumerate(eojeol_ids[batch_idx]):
                tok_cnt = eojeol_tok_cnt.detach().cpu().item()
                if 0 == tok_cnt:
                    break
                for _ in range(tok_cnt):
                    if max_seq_len <= cur_idx:
                        break
                    eojeol_info_matrix[cur_idx] = eojeol_idx
                    cur_idx += 1
            new_eojeol_info_matrix[batch_idx] = eojeol_info_matrix

        new_eojeol_info_matrix = new_eojeol_info_matrix.to(device)
        one_hot_emb = self.one_hot_embedding(new_eojeol_info_matrix)

        return one_hot_emb

    def _make_eojeol_tensor(
            self,
            last_hidden,
            pos_ids,
            eojeol_ids,
            one_hot_embed,
            max_eojeol_len=50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
              last_hidden.shape: [batch_size, token_seq_len, hidden_size]
              token_seq_len: [batch_size, ]
              pos_ids: [batch_size, eojeol_seq_len, pos_tag_size]
              eojeol_ids: [batch_size, eojeol_seq_len]
        '''

        # [64, 128, 768]
        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        all_eojeol_attention_mask = torch.zeros(batch_size, max_eojeol_len)

        # matmul
        # one_hot_embed.shape : [64, 50, 128] = [batch, max_eojeol_len, max_seq_len]
        # last_hidden.shape : [64, 128, 768] = [batch, max_seq_len, hidden]
        matmul_out_embed = one_hot_embed @ last_hidden # [64, 50, 768] = [batch, max_eojeol, hidden]

        # [ This, O, O, ... ], [ O, This, O, ... ], [ O, O, This, ...]
        eojeol_pos_1 = pos_ids[:, :, 0] # [64, eojeol_max_len]
        eojeol_pos_2 = pos_ids[:, :, 1]
        eojeol_pos_3 = pos_ids[:, :, 2]

        eojeol_pos_1 = self.eojeol_pos_embedding_1(eojeol_pos_1) # [batch_size, eojeol_max_len, pos_embed]
        eojeol_pos_2 = self.eojeol_pos_embedding_1(eojeol_pos_2)
        eojeol_pos_3 = self.eojeol_pos_embedding_1(eojeol_pos_3)
        concat_eojeol_pos_embed = torch.concat([eojeol_pos_1, eojeol_pos_2, eojeol_pos_3], dim=-1)
        # [batch_size, max_eojeol_len, hidd_size + (pos_embed * 3)]
        matmul_out_embed = torch.concat([matmul_out_embed, concat_eojeol_pos_embed], dim=-1)

        for batch_idx in range(batch_size):
            valid_eojeol_cnt = 0
            for eojeol_idx, eojeol_token_cnt in enumerate(eojeol_ids[batch_idx]):
                if 0 == eojeol_token_cnt:
                    break
                valid_eojeol_cnt += 1

            # end, eojeol loop
            eojeol_attention_mask = torch.zeros(max_eojeol_len)
            for i in range(valid_eojeol_cnt):
                eojeol_attention_mask[i] = 1
            all_eojeol_attention_mask[batch_idx] = eojeol_attention_mask
        #end, batch_loop
        return matmul_out_embed, all_eojeol_attention_mask.to(device)

    def forward(
            self,
            input_ids, attention_mask, token_type_ids, token_seq_len=None, # Unit: Token
            labels=None, pos_tag_ids=None, eojeol_ids=None # Unit: Eojeol
    ):
        # POS Embedding
        # pos_tag_ids : [batch_size, seq_len, num_pos_tags]
        electra_outputs = self.electra(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)

        el_last_hidden = electra_outputs.last_hidden_state

        # one-hot embed : [batch_size, max_seq_len, max_eojeol_len]
        one_hot_embed = self._make_ont_hot_embeddig(last_hidden=el_last_hidden,
                                                    eojeol_ids=eojeol_ids)

        # matmul one_hot @ plm outputs
        one_hot_embed_t = one_hot_embed.transpose(1, 2)
        eojeol_tensor, eojeol_attention_mask = self._make_eojeol_tensor(last_hidden=el_last_hidden,
                                                                        pos_ids=pos_tag_ids,
                                                                        eojeol_ids=eojeol_ids,
                                                                        one_hot_embed=one_hot_embed_t,
                                                                        max_eojeol_len=self.max_eojeol_len)

        eojeol_attention_mask = eojeol_attention_mask.unsqueeze(1).unsqueeze(2) # [64, 1, 1, max_eojeol_len]
        eojeol_attention_mask = eojeol_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        eojeol_attention_mask = (1.0 - eojeol_attention_mask) * -10000.0

        enc_outputs = self.encoder(eojeol_tensor, eojeol_attention_mask)
        enc_outputs = enc_outputs[-1] # [batch, max_eojeol_len, hidden]

        # 어절->Wordpiece
        enc_outputs = one_hot_embed @ enc_outputs

        trans_outputs = self.dropout(enc_outputs)

        # Classifier
        logits = self.linear(trans_outputs)  # [batch_size, seq_len, num_labels]

        # CRF
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, reduction="mean"), \
                                               self.crf.decode(logits)
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags
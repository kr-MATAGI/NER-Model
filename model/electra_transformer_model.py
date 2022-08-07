import numpy as np
import copy
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import ElectraModel, ElectraPreTrainedModel, AutoConfig
from model.crf_layer import CRF
from model.modeling_bert2 import Encoder

#==============================================================
class ELECTRA_Graph_Model(ElectraPreTrainedModel):
# ==============================================================
    def __init__(self, config):
        super(ELECTRA_Graph_Model, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.num_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels
        self.pos_embed_out_dim = 128
        self.dropout_rate = 0.33

        # for encoder
        # [768 + 128 * 3 + 128] = 1152 + 128
        self.concat_info_tensor_size = config.hidden_size + (self.pos_embed_out_dim * 3) + self.max_seq_len
        self.enc_config = copy.deepcopy(config)
        self.enc_config.num_hidden_layers = 4
        self.enc_config.hidden_size = self.concat_info_tensor_size
        self.enc_config.ff_dim = self.concat_info_tensor_size
        self.enc_config.act_fn = "gelu"
        self.enc_config.dropout_prob = 0.1
        self.enc_config.num_heads = 12  # origin 12

        # pos tag embedding
        self.pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        self.electra = ElectraModel.from_pretrained(config._name_or_path, config=config)
        self.dropout = nn.Dropout(0.3)

        # FFN
        self.linear_output_dim = 1024
        self.ffn_1 = nn.Linear(self.concat_info_tensor_size, self.linear_output_dim)

        # Transformer
        self.encoder = Encoder(self.enc_config)

        # Classifier
        self.linear = nn.Linear(self.d_model_size, config.num_labels)

        # CRF
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.pos_init()

    def _make_one_hot_embedding(
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

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            pos_tag_ids,
            eojeol_ids,
            token_seq_len=None,
            labels=None
    ):
        pos_tag_1 = pos_tag_ids[:, :, 0]  # [batch_size, seq_len]
        pos_tag_2 = pos_tag_ids[:, :, 1]  # [batch_size, seq_len]
        pos_tag_3 = pos_tag_ids[:, :, 2]  # [batch_size, seq_len]

        pos_embed_1 = self.pos_embedding_1(pos_tag_1)  # [batch_size, seq_len, pos_tag_embed]
        pos_embed_2 = self.pos_embedding_2(pos_tag_2)  # [batch_size, seq_len, pos_tag_embed]
        pos_embed_3 = self.pos_embedding_3(pos_tag_3)  # [batch_size, seq_len, pos_tag_embed]
        concat_pos_embed = torch.concat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)

        el_outputs = self.electra(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        el_outputs = el_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]

        # eojeol_one_hot_embed : [batch_size, max_seq_len, max_eojeol_len]
        # 어절 경계 원핫
        eojeol_one_hot_embed = self._make_one_hot_embedding(last_hidden=el_outputs,
                                                            eojeol_ids=eojeol_ids)

        # 모든 정보 concat
        concat_all_embed = torch.concat([el_outputs, concat_pos_embed, eojeol_one_hot_embed], dim=-1)

        extend_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extend_attention_mask = extend_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extend_attention_mask = (1.0 - extend_attention_mask) * -10000.0

        enc_outputs = self.encoder(concat_all_embed, extend_attention_mask)
        enc_outputs = enc_outputs[-1]
        enc_outputs = self.dropout(enc_outputs)

        # Classifier
        logits = self.linear(enc_outputs)

        # CRF
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, reduction="mean"), \
                                               self.crf.decode(logits)
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags
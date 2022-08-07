import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.crf_layer import CRF

#===============================================================
class Eojeol_Transformer_Encoder(nn.Module):
#===============================================================
    def __init__(self, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.33):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        '''
            d_model: input features 개수
            n_head: multiheadattetntion head 개수
            d_hid: the dimension of the feedforward network model (default=2048)
        '''
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=n_head,
                                                 dim_feedforward=d_hid,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

    def init_weights(self) -> None:
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.transformer_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return src

#===============================================================
class PositionalEncoding(nn.Module):
#===============================================================
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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

        # structure
        self.electra = ElectraModel.from_pretrained(config.model_name, config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # POS
        self.eojeol_pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.eojeol_pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.eojeol_pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        # self.eojeol_pos_embedding_4 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        # Transformer Encoder
        self.d_model_size = config.hidden_size + (self.pos_embed_out_dim * 3) # [768 + 128 * 3]
        self.transformer_encoder = Eojeol_Transformer_Encoder(d_model=self.d_model_size,
                                                              d_hid=config.hidden_size,
                                                              n_head=8, n_layers=1, dropout=0.33)

        # Classifier
        self.linear = nn.Linear(self.d_model_size, config.num_labels)

        # CRF
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    def _make_eojeol_tensor(
            self,
            last_hidden,
            pos_ids,
            eojeol_ids,
            max_eojeol_len=25
    ) -> torch.Tensor:
        '''
              last_hidden.shape: [batch_size, token_seq_len, hidden_size]
              token_seq_len: [batch_size, ]
              pos_ids: [batch_size, eojeol_seq_len, pos_tag_size]
              eojeol_ids: [batch_size, eojeol_seq_len]
        '''

        # [64, 128, 768]
        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        new_all_batch_tensor = torch.zeros(batch_size, max_eojeol_len, hidden_size + (self.pos_embed_out_dim * 3))

        # [ This, O, O, ... ], [ O, This, O, ... ], [ O, O, This, ...]
        eojeol_pos_1 = pos_ids[:, :, 0] # [64, 25]
        eojeol_pos_2 = pos_ids[:, :, 1]
        eojeol_pos_3 = pos_ids[:, :, 2]

        for batch_idx in range(batch_size):
            sent_eojeol_tensor = torch.zeros(max_eojeol_len, hidden_size + (self.pos_embed_out_dim * 3))
            start_idx = 0

            for eojeol_idx, eojeol_token_cnt in enumerate(eojeol_ids[batch_idx]):
                if 0 == eojeol_token_cnt:
                    break
                cpu_token_cnt = eojeol_token_cnt.detach().cpu().item()
                end_idx = start_idx + cpu_token_cnt

                sum_eojeol_hidden = last_hidden[batch_idx][start_idx:end_idx]
                sum_eojeol_hidden = torch.sum(sum_eojeol_hidden, dim=0).detach().cpu()
                sum_eojeol_hidden = sum_eojeol_hidden / cpu_token_cnt

                eojeol_pos_embed_1 = self.eojeol_pos_embedding_1(eojeol_pos_1[batch_idx][eojeol_idx])
                eojeol_pos_embed_2 = self.eojeol_pos_embedding_2(eojeol_pos_2[batch_idx][eojeol_idx])
                eojeol_pos_embed_3 = self.eojeol_pos_embedding_3(eojeol_pos_3[batch_idx][eojeol_idx])
                concat_eojeol_pos_embed = torch.concat([eojeol_pos_embed_1, eojeol_pos_embed_2,
                                                        eojeol_pos_embed_3], dim=-1).detach().cpu()
                sum_eojeol_hidden = torch.concat([sum_eojeol_hidden, concat_eojeol_pos_embed], dim=-1)
                sent_eojeol_tensor[eojeol_idx] = sum_eojeol_hidden

                start_idx = end_idx
            # end, eojeol loop
            new_all_batch_tensor[batch_idx] = sent_eojeol_tensor
        #end, batch_loop
        return new_all_batch_tensor.to(device)

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

        # make eojeol embedding
        eojeol_tensor = self._make_eojeol_tensor(last_hidden=el_last_hidden,
                                                 pos_ids=pos_tag_ids,
                                                 eojeol_ids=eojeol_ids)

        # forward to transformer
        # [batch_size, eojeol_len, 2304]
        trans_outputs = self.transformer_encoder(eojeol_tensor)
        trans_outputs = self.dropout(trans_outputs)

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
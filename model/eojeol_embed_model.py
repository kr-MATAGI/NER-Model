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
        self.pos_embed_out_dim = 256
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
        self.d_model_size = config.hidden_size + (self.pos_embed_out_dim * 3) # [1536]
        # self.transformer_encoder = Eojeol_Transformer_Encoder(d_model=self.d_model_size,
        #                                                       d_hid=config.hidden_size,
        #                                                       n_head=8, n_layers=3, dropout=0.33)
        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=self.d_model_size,
            hidden_size=config.hidden_size, # // 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.33
        )

        '''
        # LSTM Decoder
        self.src_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.33
        )
        '''

        # Classifier
        self.linear = nn.Linear(self.d_model_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def _make_eojeol_tensor(
            self,
            last_hidden,
            token_seq_len,
            pos_ids,
            eojeol_ids,
            max_eojeol_len=50
    ) -> torch.Tensor:
        '''
              last_hidden.shape: [batch_size, token_seq_len, hidden_size]
              token_seq_len: [batch_size, ]
              pos_ids: [batch_size, eojeol_seq_len, pos_tag_size]
              eojeol_ids: [batch_size, eojeol_seq_len]
        '''

        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        new_all_batch_tensor = torch.zeros(batch_size, max_eojeol_len, hidden_size + (self.pos_embed_out_dim * 3))

        eojoel_pos_tag_1 = pos_ids[:, :, 0] # [batch_size, eojeol_seq_len, 1]
        eojoel_pos_tag_2 = pos_ids[:, :, 1] # [batch_size, eojeol_seq_len, 1]
        eojoel_pos_tag_3 = pos_ids[:, :, 2] # [batch_size, eojeol_seq_len, 1]
        # eojoel_pos_tag_4 = pos_ids[:, :, 3] # [batch_size, eojeol_seq_len, 1]

        for batch_idx in range(batch_size):
            eojeol_hidden_list = []

            token_idx = 0
            for eojeol_idx, eojeol_bound in enumerate(eojeol_ids[batch_idx]):
                if 0 == eojeol_bound:
                    break
                token_end_idx = token_idx + eojeol_bound.item()
                if max_seq_len <= token_end_idx:
                    token_end_idx = max_seq_len-1

                # pre_eojeol_hidden = last_hidden[batch_idx][token_idx]
                # last_eojeol_hidden = last_hidden[batch_idx][token_end_idx]
                token_size = token_end_idx - token_idx + 1
                sum_eojeol_hidden = last_hidden[batch_idx][token_idx:token_end_idx] # [batch_size, word_token(가변), hidden_size]
                sum_eojeol_hidden = torch.sum(sum_eojeol_hidden, dim=0)
                sum_eojeol_hidden = (sum_eojeol_hidden / token_size).detach().cpu()

                # [1536]
                # concat_eojeol_hidden = torch.concat([pre_eojeol_hidden, last_eojeol_hidden], dim=-1).detach().cpu()

                # [eojeol_seq_len, embed_out]
                eojeol_pos_embed_1 = self.eojeol_pos_embedding_1(eojoel_pos_tag_1[batch_idx][eojeol_idx])
                eojeol_pos_embed_2 = self.eojeol_pos_embedding_2(eojoel_pos_tag_2[batch_idx][eojeol_idx])
                eojeol_pos_embed_3 = self.eojeol_pos_embedding_3(eojoel_pos_tag_3[batch_idx][eojeol_idx])
                # eojeol_pos_embed_4 = self.eojeol_pos_embedding_4(eojoel_pos_tag_4[batch_idx][eojeol_idx])
                eojeol_pos_concat = torch.concat([eojeol_pos_embed_1, eojeol_pos_embed_2,
                                                  eojeol_pos_embed_3], dim=-1).detach().cpu()

                # [1536]
                # eojeol_hidden = torch.concat([concat_eojeol_hidden, eojeol_pos_concat], dim=-1)
                eojeol_hidden = torch.concat([sum_eojeol_hidden, eojeol_pos_concat], dim=-1)
                eojeol_hidden_list.append(eojeol_hidden)

                # update token start idx
                token_idx = token_end_idx

            # 어절 길이 맞추기 (기준 max_output_eojeol_len)
            new_tensor = np.vstack(eojeol_hidden_list)
            if max_eojeol_len < new_tensor.shape[0]:
                new_tensor = new_tensor[:max_eojeol_len, :]
            else:
                diff_size = max_eojeol_len - new_tensor.shape[0]
                hidden_size = new_tensor.shape[1]
                for _ in range(diff_size):
                    new_tensor = np.vstack([new_tensor, [0] * hidden_size])
            new_tensor = torch.from_numpy(new_tensor)
            new_all_batch_tensor[batch_idx] = new_tensor

        new_all_batch_tensor = new_all_batch_tensor.to(device)
        return new_all_batch_tensor

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
                                                 token_seq_len=token_seq_len,
                                                 pos_ids=pos_tag_ids,
                                                 eojeol_ids=eojeol_ids)

        # forward to transformer
        # [batch_size, eojeol_len, 2304]
        trans_outputs = self.transformer_encoder(eojeol_tensor)
        trans_outputs = self.dropout(trans_outputs)

        # Classifier
        logits = self.linear(trans_outputs)  # [batch_size, seq_len, num_labels]

        loss = None
        if labels is not None:
            # logits_len = logits.shape[1]
            # labels = labels[:, :logits_len]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_ne_labels), labels.contiguous().view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )
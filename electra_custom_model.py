import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import ElectraModel, ElectraPreTrainedModel, AutoConfig

#================================================================================================================
class Multihead_Attention(nn.Module):
    def __init__(self, num_units, num_heads=1, dropout_rate=0, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(Multihead_Attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        return outputs

#================================================================================================================
class ELECTRA_LSTM_LAN(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ELECTRA_LSTM_LAN, self).__init__(config)
        self.pad_id = config.pad_token_id

        hp_hidden_dim = 200
        hp_dropout = 0.1
        lstm_hidden = hp_hidden_dim // 2
        label_embedding_scale = 0.0025
        num_attention_head = 5

        # PLM model
        self.electra = ElectraModel.from_pretrained("monologg/kocharelectra-base-discriminator")

        # label embedding
        self.label_dim = hp_hidden_dim
        self.label_embedding = nn.Embedding(config.num_labels, self.label_dim)

        self.lstm_first = nn.LSTM(config.hidden_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(hp_dropout)
        # self.self_attention_first = Multihead_Attention

    def forward(self, input_ids, token_type_ids, attention_mask, input_seq_len, labels=None):
        # label embedding
        # label_embs = self.label_embedding(input_label_seq_tensor)

        electra_output = self.electra(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        electra_output = electra_output.last_hidden_state

        '''
        First LSTM layer (input word only)
        '''
        pack_padded_output = pack_padded_sequence(input=electra_output, lengths=input_seq_len.detach().cpu(),
                                        batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm_first(pack_padded_output)

        # shape: [ batch_size, seq_len, hidden_size ]
        lstm_out = pad_packed_sequence(lstm_out, batch_first=True, padding_value=self.pad_id)[0]
        lstm_out = self.dropout_lstm(lstm_out)


### TEST ###
if "__main__" == __name__:
    config = AutoConfig.from_pretrained("monologg/kocharelectra-base-discriminator",
                                        num_labels=30)

    print(config)
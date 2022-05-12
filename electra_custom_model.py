import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import ElectraModel, ElectraPreTrainedModel, AutoConfig

#================================================================================================================
class Label_Embedding(nn.Module):
    def __init__(self, label_dim, num_labels, label_embedding_scale):
        super(Label_Embedding, self).__init__()

        self.label_embedding = nn.Embedding(num_labels, label_dim)
        self.label_embedding.weight.data.copy_(torch.from_numpy(
            self._random_embedding_label(num_labels, label_dim, label_embedding_scale))
        )

    def _random_embedding_label(self, vocab_size, embedding_dim, scale):
        pretrain_emb = np.empty([vocab_size, embedding_dim])

        for idx in range(vocab_size):
            pretrain_emb[idx, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, input_label_seq_tensor):
        label_embs = self.label_embedding(input_label_seq_tensor)
        return label_embs

#================================================================================================================
class Multihead_Attention(nn.Module):
    def __init__(self, num_units, num_heads=1, dropout_rate=0.0):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            num_heads: An int. Number of heads.
        '''
        super(Multihead_Attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values, last_layer=False):
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
        # bmm은 batch matrix multiplication으로 두 opearnd가 모두 batch일 때 사용
        # [B, n, m] x [B, m, p] = [B, n, p]
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        return outputs

#================================================================================================================
class LSTM_Attention(nn.Module):
    def __init__(self, input_size, lstm_hidden, num_heads, max_len,
                 bilstm_flg, dropout_rate, is_last_layer=False, pad_id=0):
        super(LSTM_Attention, self).__init__()
        self.is_last_layer = is_last_layer
        self.max_len = max_len
        self.pad_id = pad_id

        self.lstm = nn.LSTM(input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=bilstm_flg)
        self.label_attn = Multihead_Attention(lstm_hidden * 2, num_heads=num_heads, dropout_rate=dropout_rate)
        self.drop_lstm = nn.Dropout(dropout_rate)

    def forward(self, lstm_out, label_embs, word_seq_lengths, hidden):
        lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(),
                                        enforce_sorted=False, batch_first=True)
        lstm_out, hidden = self.lstm(lstm_out, hidden)
        lstm_out = pad_packed_sequence(lstm_out, total_length=self.max_len, padding_value=self.pad_id)[0]
        lstm_out = self.drop_lstm(lstm_out.transpose(1, 0))

        label_attention_output = self.label_attn(lstm_out, label_embs, label_embs, last_layer=self.is_last_layer)
        if self.is_last_layer:
           return label_attention_output
        else:
            lstm_out = torch.cat([lstm_out, label_attention_output], -1)
            return lstm_out

#================================================================================================================
class ELECTRA_LSTM_LAN(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ELECTRA_LSTM_LAN, self).__init__(config)
        self.pad_id = config.pad_token_id
        self.max_seq_len = config.max_seq_len

        hidden_dim = 400
        dropout_rate = 0.1
        lstm_hidden = hidden_dim // 2
        label_embedding_scale = 0.0025
        num_attention_head = 5

        # label embedding
        self.label_embedding = Label_Embedding(num_labels=config.num_labels, label_dim=hidden_dim,
                                               label_embedding_scale=label_embedding_scale)
        # PLM model
        self.electra = ElectraModel.from_pretrained("monologg/kocharelectra-base-discriminator")

        # LAN
        self.lstm_attn_1 = LSTM_Attention(input_size=config.hidden_size, lstm_hidden=lstm_hidden, bilstm_flg=True,
                                          dropout_rate=dropout_rate, num_heads=num_attention_head, max_len=self.max_seq_len)
        self.lstm_attn_2 = LSTM_Attention(input_size=lstm_hidden * 4, lstm_hidden=lstm_hidden, bilstm_flg=True,
                                          dropout_rate=dropout_rate, num_heads=num_attention_head, max_len=self.max_seq_len)

        # DO NOT Add dropout at last layer
        self.lstm_attn_last = LSTM_Attention(input_size=lstm_hidden * 4, lstm_hidden=lstm_hidden, bilstm_flg=True,
                                             dropout_rate=0.0, num_heads=1, is_last_layer=True, max_len=self.max_seq_len)

    def forward(self, input_ids, token_type_ids, attention_mask, input_seq_len, input_label_seq_tensor, labels=None):
        '''
        Args:
            input_label_seq_tensor: [batch_size, num_labels]
        '''
        # label embedding
        # shape: [ batch_size, num_labels, hidden_dim ]
        label_embs = self.label_embedding(input_label_seq_tensor) # [32, 31, 768]

        electra_output = self.electra(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        electra_output = electra_output.last_hidden_state # [batch_size, seq_len, config.hidden_size]

        # LAN layer
        hidden = None
        # [ batch_size, seq_len, hidden_dim * 2]
        lstm_out = self.lstm_attn_1(electra_output, label_embs, input_seq_len, hidden)
        lstm_out = self.lstm_attn_2(lstm_out, label_embs, input_seq_len, hidden)
        # [ batch_size, seq_len, num_labels]
        lstm_out = self.lstm_attn_last(lstm_out, label_embs, input_seq_len, hidden)

        if labels is None:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)

            outs = lstm_out.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            return tag_seq
        else:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)

            loss_func = nn.NLLLoss()
            outs = lstm_out.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_func(score, labels.view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            total_loss = total_loss / batch_size

            return total_loss, tag_seq

        return lstm_out

### TEST ###
if "__main__" == __name__:
    config = AutoConfig.from_pretrained("monologg/kocharelectra-base-discriminator",
                                        num_labels=30)

    print(config)
import torch
import copy
import math
import torch.nn as nn
import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#===============================================================
class Eojeol_Transformer_Encoder(nn.Module):
#===============================================================
    def __init__(self, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.3):
        super().__init__()
        self.model_type = "Transformer"
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

    def init_weights(self) -> None:
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
            Args:
                src: Tensor, shape [seq_len, batch_size]

            Returns:
                output Tensor of shape [seq_len, batch_size, ntoken]
        """
        output = self.transformer_encoder(src)
        return output

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
        self.pos_embed_out_dim = 100

        # structure
        self.electra = ElectraModel.from_pretrained(config.model_name, config=config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # POS
        self.eojeol_pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.eojeol_pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.eojeol_pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.eojeol_pos_embedding_4 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        # Transformer Encoder
        d_model_size = config.hidden_size + (self.pos_embed_out_dim * 4)
        self.transformer_encoder = Eojeol_Transformer_Encoder(d_model=d_model_size,
                                                              d_hid=config.hidden_size,
                                                              n_head=8, n_layers=3, dropout=0.33)

        # encoder
        # self.encoder = nn.LSTM()

        # decoder
        # self.decoder = nn.LSTM()

        # Classifier
        self.classifier = nn.Linear(d_model_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def _make_eojeol_tensor(
            self,
            last_hidden,
            token_seq_len,
            pos_ids,
            eojeol_ids,
            max_input_eojeol_len=100,
            max_output_eojeol_len=50
    ) -> torch.Tensor:
        '''
              last_hidden.shape: [batch_size, token_seq_len, hidden_size]
              token_seq_len: [batch_size, ]
              pos_ids: [batch_size, eojeol_seq_len, pos_tag_size]
              eojeol_ids: [batch_size, eojeol_seq_len]
        '''

        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        new_all_batch_tensor = torch.zeros(batch_size, max_output_eojeol_len, hidden_size + (self.pos_embed_out_dim * 4))

        eojoel_pos_tag_1 = pos_ids[:, :, 0] # [batch_size, eojeol_seq_len]
        eojoel_pos_tag_2 = pos_ids[:, :, 1] # [batch_size, eojeol_seq_len]
        eojoel_pos_tag_3 = pos_ids[:, :, 2] # [batch_size, eojeol_seq_len]
        eojoel_pos_tag_4 = pos_ids[:, :, 3] # [batch_size, eojeol_seq_len]

        for batch_idx in range(batch_size):
            eojeol_hidden_list = []

            token_idx = 0
            for eojeol_idx, eojeol_bound in enumerate(eojeol_ids[batch_idx]):
                if 0 == eojeol_bound:
                    break
                token_end_idx = token_idx + eojeol_bound

                slice_eojeol_hidden = last_hidden[batch_idx][token_idx:token_end_idx]
                eojeol_hidden = slice_eojeol_hidden.mean(dim=0).detach().cpu()
                eojeol_hidden = torch.nan_to_num(eojeol_hidden)

                # [eojeol_seq_len, embed_out]
                eojeol_pos_embed_1 = self.eojeol_pos_embedding_1(eojoel_pos_tag_1[batch_idx][eojeol_idx])
                eojeol_pos_embed_2 = self.eojeol_pos_embedding_2(eojoel_pos_tag_2[batch_idx][eojeol_idx])
                eojeol_pos_embed_3 = self.eojeol_pos_embedding_3(eojoel_pos_tag_3[batch_idx][eojeol_idx])
                eojeol_pos_embed_4 = self.eojeol_pos_embedding_4(eojoel_pos_tag_4[batch_idx][eojeol_idx])
                eojeol_pos_concat = torch.concat([eojeol_pos_embed_1, eojeol_pos_embed_2,
                                                  eojeol_pos_embed_3, eojeol_pos_embed_4], dim=-1).detach().cpu()
                eojeol_hidden = torch.concat([eojeol_hidden, eojeol_pos_concat], dim=-1)
                eojeol_hidden_list.append(eojeol_hidden)

                # update token start idx
                token_idx = token_end_idx

            # 어절 길이 맞추기 (기준 max_output_eojeol_len)
            new_tensor = np.vstack(eojeol_hidden_list)
            if max_output_eojeol_len < new_tensor.shape[0]:
                new_tensor = new_tensor[:max_output_eojeol_len, :]
            else:
                diff_size = max_output_eojeol_len - new_tensor.shape[0]
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
        trans_outputs = self.transformer_encoder(eojeol_tensor)

        # classifier
        logits = self.classifier(trans_outputs)

        loss = None
        if labels is not None:
            #print("AAAAAA :", labels.shape) # [55, 100]
            #print("BBBBBB :", labels.shape) # [55, 50]
            #print("CCCCCC :", logits.shape) # [55, 50, 31]
            #print("DDDDDD :", logits.view(-1, self.num_ne_labels).shape) # [2750, 31]
            #print("EEEEEE :", labels.reshape(-1).shape) # [2750]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_ne_labels), labels.reshape(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
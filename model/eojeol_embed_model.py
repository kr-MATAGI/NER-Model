import torch
import copy
import torch.nn as nn
import numpy as np

from transformers import ElectraModel, ElectraTokenizer, ElectraPreTrainedModel


class Eojeol_Embed_Model(ElectraPreTrainedModel):
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
        self.pos_embed_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embed_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embed_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embed_4 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

    def _make_eojeol_tensor(self, last_hidden, pos_ids, eojeol_ids) -> torch.Tensor:
        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        new_batch_tensor = torch.zeros(batch_size, max_seq_len, hidden_size).to(device)

        return

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
        eojeol_batch_tensor = self._make_eojeol_tensor(last_hidden=el_last_hidden,
                                                       pos_ids=pos_tag_ids,
                                                       eojeol_ids=eojeol_ids)


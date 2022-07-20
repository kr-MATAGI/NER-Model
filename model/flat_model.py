import collections
import numpy as np
import torch
import torch.nn as nn

from transformers import ElectraModel, ElectraPreTrainedModel
from model.crf_layer import CRF

#===============================================================
class Four_Pos_Fusion_Embedding(nn.Module):
#===============================================================
    def __init__(self, pe, four_pos_fusion, pe_ss, pe_se, pe_es, pe_ee, max_seq_len, hidden_size, mode):
        super().__init__()
        # self.mode = mode # debug / ???
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.pe = pe
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee

        self.four_pos_fusion = four_pos_fusion
        self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4, self.hidden_size),
                                                nn.ReLU(inplace=True))

    def forward(self, pos_s, pos_e):
        batch_size = pos_s.size(0)
        # max_seq_len = pos_s.size(1) # temp

        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.max_seq_len
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.max_seq_len
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.max_seq_len
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.max_seq_len

        pe_ss = pos_ss.view(size=[batch_size, self.max_seq_len, self.max_seq_len, -1])
        pe_se = pos_se.view(size=[batch_size, self.max_seq_len, self.max_seq_len, -1])
        pe_es = pos_es.view(size=[batch_size, self.max_seq_len, self.max_seq_len, -1])
        pe_ee = pos_ee.view(size=[batch_size, self.max_seq_len, self.max_seq_len, -1])

        pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
        pe_4 = pe_4.view(size=[-1, 4])
        pe_unique, inverse_indices = torch.unique(pe_4, sorted=True, return_inverse=True, dim=0)
        pos_unique_embedding = self.pe(pe_unique)
        pos_unique_embedding = pos_unique_embedding.view([pos_unique_embedding.size(0), -1])
        pos_unique_embedding_after_fusion = self.pos_fusion_forward(pos_unique_embedding)
        rel_pos_embedding = pos_unique_embedding_after_fusion[inverse_indices]
        rel_pos_embedding = rel_pos_embedding.view(size=[batch_size, self.max_seq_len, self.max_seq_len, -1])

        return rel_pos_embedding

#===============================================================
class Transformer_Encoder_Layer(nn.Module):
#===============================================================
    def __init__(self, hidden_size, num_heads,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=1, mode=collections.defaultdict(bool),
                 max_seq_len=-1, pe=None,
                 pe_ss=None, pe_se=None, pe_es=None, pe_ee=None,
                 device=None,
                 k_pro=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_ff=True, ff_activate="relu", lattice=False,
                 four_pos_shared=True, four_pos_fusion=None, four_pos_fusion_embedding=None):
        super().__init__()

#===============================================================
class Transformer_Encoder(nn.Module):
#===============================================================
    def __init__(self, hidden_size, num_heads, num_layers,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 mode=collections.defaultdict(bool), device=None, max_seq_len=-1,
                 pe=None, pe_ss=None, pe_se=None, pe_es=None, pe_ee=None,
                 k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_ff=True, ff_activate='relu', lattice=False,
                 four_pos_shared=True, four_pos_fusion=None, four_pos_fusion_shared=True):
        super().__init__()

        self.is_four_pos_fusion_shared = four_pos_fusion_shared
        self.is_four_pos_shared = four_pos_shared
        self.is_four_pos_fusion = four_pos_fusion

        self.pe = pe
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee

        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        if self.four_pos_fusion_shared:
            self.four_pos_fusion_embedding = Four_Pos_Fusion_Embedding(self.pe, self.four_pos_fusion,
                                                                       self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                                                       self.max_seq_len, self.hidden_size)
        else:
            self.four_pos_fusion.embedding = None

        self.lattice = lattice
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.relative_position =

    def forward(self, inp):
        output = inp

#===============================================================
class ELECTRA_FLAT(ElectraPreTrainedModel):
#===============================================================
    def __init__(self, config):
        super(ELECTRA_FLAT, self).__init__(config)

        # ELECTRA
        self.electra = ElectraModel.from_pretrained(config.__name_or_path, config=config)

        # Transformer
        self.encoder = Transformer_Encoder()

        # CRF
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, pos_tag_ids=None, labels=None):
        electra_output = self.electra(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)

        # [batch_size, seq_len, hidden_size]
        electra_output = electra_output.last_hidden_state

        # CRF
        logits = None
        if labels is not None:
            log_liklihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, mask=attention_mask.bool(),
                                                       reduction="mean"), self.crf.decode(logits, mask=attention_mask.bool())
            return log_liklihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags
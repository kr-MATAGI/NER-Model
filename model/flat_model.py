import copy
import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraModel, ElectraPreTrainedModel
from model.crf_layer import CRF

from fastNLP import seq_len_to_mask

#===============================================================
class Four_Pos_Fusion_Embedding(nn.Module):
#===============================================================
    def __init__(self, pe, four_pos_fusion, pe_ss, pe_se, pe_es, pe_ee, max_seq_len, hidden_size):
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
class MyDropout(nn.Module):
#===============================================================
    def __init__(self, ratio):
        super().__init__()
        assert 0 <= ratio <= 1
        self.ratio = ratio

    def forward(self, x):
        if self.training and self.ratio > 0.001:
            mask = torch.rand(x.size())
            mask = mask.to(x)
            mask = mask.lt(self.ratio)
            x = x.masked_fill(mask, 0) / (1-self.ratio)
        return x

#===============================================================
class Layer_Process(nn.Module):
#===============================================================
    def __init__(self, process_sequence, hidden_size, dropout=0):
        super().__init__()
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            self.dropout = MyDropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inp):
        output = inp
        for op in self.process_sequence:
            if op == 'a':
                output = output + inp
            elif op == 'd':
                output = self.dropout(output)
            elif op == 'n':
                output = self.layer_norm(output)
        return output

#===============================================================
class MultiHead_Attention_rel(nn.Module):
#===============================================================
    def __init__(self, hidden_size, num_heads, pe, scaled=True, max_seq_len=-1,
                 k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_dropout=None, ff_final=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))

        self.pe = pe
        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, key, query, value, seq_len):
        max_seq_len = torch.max(seq_len)
        rel_distance = self.seq_len_to_rel_distance(max_seq_len)
        rel_distance_flat = rel_distance.view(-1)
        rel_pos_embedding_flat = self.pe[rel_distance_flat + self.max_seq_len]
        rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len, max_seq_len, self.hidden_size])

        if self.k_proj:
            key = self.w_k(key)
        if self.q_proj:
            query = self.w_q(query)
        if self.v_proj:
            value = self.w_v(value)
        if self.r_proj:
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)

        # batch_size * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [max_seq_len, max_seq_len, self.num_heads, self.per_head_size])

        # batch_size * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)

        # A
        A_ = torch.matmul(query, key)

        # B
        rel_pos_embedding_for_b = rel_pos_embedding.unsqueeze(0).permute(0, 3, 1, 4, 2)
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        B_ = torch.matmul(query_for_b, rel_pos_embedding_for_b).squeeze(-2)

        # D
        rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        D_ = torch.matmul(rel_pos_embedding_for_d, v_for_d).squeeze(-1).squeeze(-1).permute(2, 0, 1).unsqueeze(0)

        # C
        # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        key_for_c = key
        C_ = torch.matmul(u_for_c, key)

        # att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        attn_score_raw = A_ + B_ + C_ + D_

        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(seq_len).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)
        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)
        value_weighted_sum = torch.matmul(attn_score, value)
        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)

        if hasattr(self,'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)
        return result

    def seq_len_to_rel_distance(self, max_seq_len):
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        #index = index.to(self.dvc)
        return index

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
class Positionwise_FeedForward(nn.Module):
#===============================================================
    def __init__(self, sizes, dropout=None, ff_activate='relu'):
        super().__init__()
        self.num_layers = len(sizes) - 1
        for i in range(self.num_layers):
            setattr(self, 'w' + str(i), nn.Linear(sizes[i], sizes[i + 1]))

        if dropout == None:
            dropout = collections.defaultdict(int)

        self.dropout = MyDropout(dropout['ff'])
        self.dropout_2 = MyDropout(dropout['ff_2'])
        if ff_activate == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif ff_activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, inp):
        output = inp
        for i in range(self.num_layers):
            if i != 0:
                output = self.activate(output)
            w = getattr(self, 'w' + str(i))
            output = w(output)
            if i == 0:
                output = self.dropout(output)
            if i == 1:
                output = self.dropout_2(output)
        return output

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
        self.is_pos_norm = False

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
        self.relative_position = relative_position
        if self.relative_position and self.lattice:
            assert four_pos_fusion is not None
        self.is_four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate
        self.max_seq_len = max_seq_len

        if self.relative_position and self.max_seq_len < 0:
            print("max_seq_len should be set if relative position encode")
            exit(1208)

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        if self.relative_position:
            if pe is None:
                pe = self.get_embedding(max_seq_len, hidden_size, rel_pos_init=0)
                pe_sum = pe.sum(dim=-1, keepdim=True)
                if self.pos_norm:
                    with torch.no_grad():
                        pe = pe / pe_sum
                self.pe = nn.Embedding(max_seq_len * 2 + 1, hidden_size, _weight=pe)
                if self.is_four_pos_shared:
                    self.pe_ss = self.pe
                    self.pe_se = self.pe
                    self.pe_es = self.pe
                    self.pe_ee = self.pe
                else:
                    self.pe_ss = nn.Embedding(max_seq_len * 2 + 1, hidden_size, _weight=pe)
                    self.pe_se = nn.Embedding(max_seq_len * 2 + 1, hidden_size, _weight=pe)
                    self.pe_es = nn.Embedding(max_seq_len * 2 + 1, hidden_size, _weight=pe)
                    self.pe_ee = nn.Embedding(max_seq_len * 2 + 1, hidden_size, _weight=pe)
            else:
                self.pe = pe
                self.pe_ss = pe_ss
                self.pe_se = pe_se
                self.pe_es = pe_es
                self.pe_ee = pe_ee
        if self.four_pos_fusion_embedding is None:
            self.four_pos_fusion_embedding = Four_Pos_Fusion_Embedding(self.pe, self.is_four_pos_fusion,
                                                                       self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                                                       self.max_seq_len, self.hidden_size)
        if dropout == None:
            dropout = collections .defaultdict(int)
        self.dropout = dropout

        if -1 == ff_size:
            ff_size = hidden_size
        self.ff_size = ff_size
        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence, self.hidden_size, self.dropout['pre'])
        self.layer_postprocess = Layer_Process(self.layer_postprocess_sequence, self.hidden_size, self.dropout['post'])
        if self.relative_position:
            if not self.lattice:
                self.attn = MultiHead_Attention_rel(self.hidden_size, self.num_heads,
                                                    pe=self.pe,
                                                    scaled=self.scaled,
                                                    max_seq_len=self.max_seq_len,
                                                    k_proj=self.k_proj,
                                                    q_proj=self.q_proj,
                                                    v_proj=self.v_proj,
                                                    r_proj=self.r_proj,
                                                    attn_dropout=self.dropout['attn'],
                                                    ff_final=self.attn_ff)
        self.ff = Posi

    def get_embedding(self, max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
        '''
            @NOTE
            rel_pos_init
                - 0: -max_len에서 max_len까지 상대 위치 인코딩 행렬을 0-2*max_len으로 초기화
                - 1: -max_len, max_len을 눌러(?) 초기화
        '''
        num_embeddings = 2 * max_seq_len + 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        if 0 == rel_pos_init:
            emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        else:
            emb = torch.arange(-max_seq_len, max_seq_len+1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if 1 == embedding_dim % 2:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

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
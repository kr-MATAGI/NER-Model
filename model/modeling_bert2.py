# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import math
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax, MSELoss

logger = logging.getLogger(__name__)

LayerNorm = nn.LayerNorm


def Linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.o_proj = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(config.dropout_prob)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention_scores : [64, 8, 25, 25]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.o_proj(context_layer)

        return attention_output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.ff_dim)
        self.fc2 = Linear(config.ff_dim, config.hidden_size)
        self.act_fn = ACT2FN[config.act_fn]
        self.dropout = Dropout(config.dropout_prob)

    def forward(self, input):
        intermediate = self.fc1(input)
        ff_out = self.dropout(self.fc2(self.act_fn(intermediate)))
        return ff_out


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.ffn = PositionWiseFeedForward(config)
        self.attn = Attention(config)

    def forward(self, x, attention_mask):
        # Attention
        h = x
        x = self.attn(x, attention_mask)
        x = h + x
        x = self.attention_norm(x)

        # FFN
        h = x
        x = self.ffn(x)
        x = x + h
        x = self.ffn_norm(x)

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for l in range(config.num_hidden_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class Enc_Config(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 act_fn="gelu",
                 hidden_size=768,
                 num_hidden_layers=12,
                 ff_dim=3072,
                 num_heads=12,
                 dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02
                 ):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.act_fn = act_fn
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.ff_dim = ff_dim
            self.num_heads = num_heads
            self.dropout_prob = dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        config = Enc_Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.dropout_prob)

        self.init_weights(config)

    def init_weights(self, config):
        self.word_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        #self.pooler = Pooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        #extended_attention_mask = extended_attention_mask.to(torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask)

        sequence_output1 = encoded_layers[6]
        sequence_output2 = encoded_layers[9]
        sequence_output3 = encoded_layers[-1]

        pooled_output1 = self.pooler(sequence_output1)
        pooled_output2 = self.pooler(sequence_output2)
        pooled_output3 = self.pooler(sequence_output3)

        return sequence_output1, sequence_output2, sequence_output3, pooled_output1, pooled_output2, pooled_output3
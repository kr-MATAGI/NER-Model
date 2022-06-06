import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoConfig, BertPreTrainedModel, AutoModelForTokenClassification
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy

from crf_layer import CRF

#==========================================================================================

class BERT_LSTM_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_LSTM_CRF, self).__init__(config)
        self.max_seq_len = 128

        self.bert = AutoModel.from_pretrained("klue/bert-base", config=config)
        self.lstm_hidden = 512
        self.dropout_rate = 0.1

        ## New Layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=self.lstm_hidden // 2, num_layers=1,
                            dropout=self.dropout_rate, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.lstm_hidden, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None,
                pad_id=0, using_pack_sequence=True):
        seq_len = torch.LongTensor([torch.max(input_ids[i, :].data.nonzero()) + 1
                                    for i in range(input_ids.size(0))])

        bert_outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output, pooled_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        sequence_output = self.dropout(sequence_output)

        if using_pack_sequence:
            pack_padded_output = pack_padded_sequence(sequence_output, seq_len.tolist(),
                                                      batch_first=True, enforce_sorted=False)
            lstm_output, hidden = self.lstm(pack_padded_output)
            lstm_output = pad_packed_sequence(lstm_output, batch_first=True, padding_value=pad_id,
                                              total_length=self.max_seq_len)[0]
        else:
            lstm_output, hidden = self.lstm(sequence_output)
        emissions = self.linear(lstm_output)

        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=emissions, tags=labels, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(emissions,
                                                                                           mask=attention_mask.bool())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, mask=attention_mask.bool())
            return sequence_of_tags

#====================================================================================
class BERT_IDCNN_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_IDCNN_CRF, self).__init__(config)
        self.filter_num = config.max_len
        self.idcnn_nums = config.idcnn_nums
        kernel_size = 3
        dropout_rate = 0.1

        self.bert = AutoModel.from_pretrained(config._name_or_path, config=config)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_num, out_channels=self.filter_num,
                      kernel_size=kernel_size, dilation=1,
                      padding=kernel_size//2 + 1 - 1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filter_num, out_channels=self.filter_num,
                      kernel_size=kernel_size, dilation=1,
                      padding=kernel_size // 2 + 1 - 1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filter_num, out_channels=self.filter_num,
                      kernel_size=kernel_size, dilation=2,
                      padding=kernel_size // 2 + 2 - 1),
            nn.ReLU(),
        )
        self.idcnn = nn.ModuleList([self.cnn for _ in range(self.idcnn_nums)])
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(config.hidden_size * self.idcnn_nums, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output, pooled_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        # [ batch_size, max_seq_len, hidden_size ]
        sequence_output = self.dropout(sequence_output)

        idcnn_outputs = [idcnn_layer(sequence_output) for idcnn_layer in self.idcnn]
        if 1 == self.idcnn_nums:
            idcnn_outputs = idcnn_outputs[0]
        else:
            # [ batch_size, max_seq_len, hidden * idcnn_nums ]
            idcnn_outputs = torch.concat(idcnn_outputs, dim=-1)
        emissions = self.linear(idcnn_outputs)
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=emissions, tags=labels, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(emissions, mask=attention_mask.bool())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags

#====================================================================================
class BERT_POS_LSTM(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_POS_LSTM, self).__init__(config)
        self.max_seq_len = 128
        self.num_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels
        self.pos_embed_out_dim = 100

        # pos tag embedding
        self.pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        # bert + lstm
        '''
            @ Note
                AutoModel.from_config()
                Loading a model from its configuration file does not load the model weights. 
                It only affects the model’s configuration. 
                Use from_pretrained() to load the model weights.
        '''
        self.bert = AutoModel.from_config(config=config)
        # self.lstm = nn.LSTM(input_size=config.hidden_size + (self.pos_embed_out_dim * 3),
        #                     hidden_size=config.hidden_size + (self.pos_embed_out_dim * 3),
        #                     num_layers=1, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size + (self.pos_embed_out_dim * 3), config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.post_init()

    def forward(self, input_ids, attention_mask, token_type_ids, pos_tag_ids, input_seq_len, labels=None):
        # pos embedding
        # pos_tag_ids : [batch_size, seq_len, num_pos_tags]
        pos_tag_1 = pos_tag_ids[:, :, 0] # [batch_size, seq_len]
        pos_tag_2 = pos_tag_ids[:, :, 1] # [batch_size, seq_len]
        pos_tag_3 = pos_tag_ids[:, :, 2] # [batch_size, seq_len]

        pos_embed_1 = self.pos_embedding_1(pos_tag_1) # [batch_size, seq_len, pos_tag_embed]
        pos_embed_2 = self.pos_embedding_2(pos_tag_2)  # [batch_size, seq_len, pos_tag_embed]
        pos_embed_3 = self.pos_embedding_3(pos_tag_3)  # [batch_size, seq_len, pos_tag_embed]

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        sequence_output = outputs[0] # [batch_size, seq_len, hidden_size]
        concat_embed = torch.concat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)
        concat_embed = torch.concat([sequence_output, concat_embed], dim=-1)
        # lstm_out, _ = self.lstm(concat_embed) # [batch_size, seq_len, hidden_size]
        lstm_out = self.dropout(concat_embed)
        logits = self.classifier(lstm_out)

        # crf
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(logits, mask=attention_mask.bool())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

### TEST ###
if "__main__" == __name__:
    config = AutoConfig.from_pretrained("klue/bert-base",
                                        num_labels=31)
    bert_config = AutoConfig.from_pretrained("klue/bert-base")
    print(config)
    print(bert_config)
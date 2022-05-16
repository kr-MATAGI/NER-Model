import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoConfig, BertPreTrainedModel,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy

from crf_layer import CRF

#==========================================================================================

class BERT_LSTM(BertPreTrainedModel):
    def __init__(self, config):
        config.output_attention = True
        self.max_seq_len = 128

        super(BERT_LSTM, self).__init__(config)
        self.bert = AutoModel.from_pretrained(config._name_or_path, config=config)
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
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags

#====================================================================================

### TEST ###
if "__main__" == __name__:
    config = AutoConfig.from_pretrained("klue/bert-base",
                                        num_labels=31)
    config.output_attention = True
    print(config)
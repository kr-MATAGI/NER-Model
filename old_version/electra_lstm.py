import torch
import torch.nn as nn
from torchcrf import CRF

from transformers import ElectraPreTrainedModel, ElectraModel

class Electra_BiLSTM(ElectraPreTrainedModel):
    def __init__(self, config, output_dim=256, num_layers=1):
        super(Electra_BiLSTM, self).__init__(config)

        self.electra = ElectraModel(config)
        self.rnn = nn.LSTM(config.hidden_size, output_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(output_dim * 2, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        electra_outputs = self.electra(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)
        lstm_outputs, _ = self.rnn(electra_outputs[0])
        dropout_outputs = self.dropout(lstm_outputs)
        emissions = self.linear(lstm_outputs)
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=emissions, tags=labels, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(emissions,
                                                                                           mask=attention_mask.bool())
        return log_likelihood, sequence_of_tags

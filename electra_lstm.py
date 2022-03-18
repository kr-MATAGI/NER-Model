import torch
import torch.nn as nn

from transformers import ElectraPreTrainedModel, ElectraModel

class Electra_BiLSTM(ElectraPreTrainedModel):
    def __init__(self, config):
        super(Electra_BiLSTM, self).__init__(config)

        self.electra = ElectraModel(config)
        self.rnn = nn.LSTM(config.hidden_size, 200,
                           num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(200 * 2, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        electra_outputs = self.electra(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask, labels=labels)
        lstm_outputs, _ = self.rnn(electra_outputs[0])
        output = self.linear(lstm_outputs).transpose(1, 2)
        loss = self.loss_fu(output, labels)
        return loss

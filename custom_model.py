import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BertPreTrainedModel


class BERT_LSTM(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_LSTM, self).__init__(config)
        self.bert = AutoModel.from_pretrained("klue/bert-base", config=config)
        self.lstm_hidden = 512

        ## New Layers
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=self.lstm_hidden//2, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.lstm_hidden, config.num_labels)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output, pooled_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        lstm_output, hidden = self.lstm(sequence_output)
        output = self.linear(lstm_output)
        sequence_of_tags = torch.argmax(output, dim=-1)

        if labels is not None:
            loss = self.criterion(sequence_of_tags.float(), labels.float())
            loss.requires_grad = True

        return loss, sequence_of_tags

### TEST ###
if "__main__" == __name__:
    config = AutoConfig.from_pretrained("klue/bert-base",
                                        num_labels=30)

    print(config)
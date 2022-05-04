import torch
import torch.nn as nn
from transformers import AutoModel


class BERT_LSTM(nn.Module):
    def __init__(self, config):
        super(BERT_LSTM, self).__init__()
        self.bert = AutoModel.from_pretrained("klue/bert-base", config=config)

        ## New Layers
        self.lstm = nn.LSTM(768, 30, batch_first=True, bidirectional=True)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output, pooled_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        lstm_output, (h, c) = self.lstm(sequence_output)  ## extract the 1st token's embeddings
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)

        ### assuming that you are only using the output of the last LSTM cell to perform classification
        #linear_output = self.linear(hidden.view(-1, 256 * 2))

        return hidden
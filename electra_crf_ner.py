import torch
import torch.nn as nn
from crf_layer import CRF

from transformers import ElectraPreTrainedModel, ElectraModel

class ElectraCRF_NER(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraCRF_NER, self).__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=emissions, tags=labels, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(emissions, mask=attention_mask.bool())
        return log_likelihood, sequence_of_tags
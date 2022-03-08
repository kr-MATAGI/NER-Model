import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from TorchCRF import CRF

from transformers import ElectraPreTrainedModel, ElectraModel
from transformers.modeling_outputs import TokenClassifierOutput

class MyElectraTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.hidden = torch.randn((config.batch_size, 512, self.num_labels), requires_grad=True).to(config.device)
        self.crf = CRF(self.num_labels)

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        #logits = self.classifier(discriminator_sequence_output)
        logits = self.crf.forward(self.hidden, labels, attention_mask)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import ElectraPreTrainedModel, ElectraModel, ElectraConfig
from transformers.modeling_outputs import TokenClassifierOutput

#==============================================================
class Transformer_Encoder(nn.Module):
#==============================================================
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.3):
        super().__init__()
        self.model_type = "Transformer"
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        output = self.transformer_encoder(src)
        return output

#==============================================================
class Electra_Trans_Model(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.num_ne_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels
        self.pos_embed_out_dim = 100
        self.span_embed_out_dim = 300

        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # POS Embedding
        self.pos_embed_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embed_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embed_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        # Span Embedding
        self.span_embed = nn.Embedding(config.max_seq_len, self.span_embed_out_dim)

        # transformer
        d_model_size = config.hidden_size + (self.pos_embed_out_dim * 3) + self.span_embed_out_dim
        self.transformer_encoder = Transformer_Encoder(d_model=d_model_size,
                                                       d_hid=config.hidden_size, nhead=8, nlayers=3, dropout=0.33)

        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(d_model_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids, attention_mask, token_type_ids,
            pos_tag_ids=None, span_ids=None, input_seq_len=None, labels=None
    ):
        # POS Embedding
        # pos_tag_ids : [batch_size, seq_len, num_pos_tags]
        pos_ids_1 = pos_tag_ids[:, :, 0] # [batch_size, seq_len]
        pos_ids_2 = pos_tag_ids[:, :, 1] # [batch_size, seq_len]
        pos_ids_3 = pos_tag_ids[:, :, 2] # [batch_size, seq_len]

        pos_embed_1 = self.pos_embed_1(pos_ids_1)
        pos_embed_2 = self.pos_embed_2(pos_ids_2)
        pos_embed_3 = self.pos_embed_3(pos_ids_3)

        # Span Embedding
        span_embed = self.span_embed(span_ids)

        # ELECTRA
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        sequence_outputs = outputs.last_hidden_state # [batch_size, seq_len, hidden_size]
        concat_embed = torch.concat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)
        concat_embed = torch.concat([concat_embed, span_embed], dim=-1)
        concat_embed = torch.concat([sequence_outputs, concat_embed], dim=-1)

        # Transformer
        trans_outputs = self.transformer_encoder(concat_embed)

        # Classifier
        trans_outputs = self.dropout(trans_outputs)
        logits = self.classifier(trans_outputs) # [batch_size, seq_len, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_ne_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

### Main
if "__main__" == __name__:
    config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    print(config)
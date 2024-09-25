from transformers.models.bert.modeling_bert import BertAttention
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.crossattention = BertAttention(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask
    ):
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        attention_output = self.crossattention(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )[0]
        return attention_output
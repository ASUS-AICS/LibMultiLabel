import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

from .modules import LabelwiseAttention, LabelwiseLinearOutput, LabelwiseMultiHeadAttention


class BERTAttention(nn.Module):
    """BERT + Label-wise Document Attention or Multi-Head Attention

    Args:
        num_classes (int): Total number of classes.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        lm_weight (str): Pretrained model name or path. Defaults to 'bert-base-cased'.
        lm_window (int): Length of the subsequences to be split before feeding them to
            the language model. Defaults to 512.
        num_heads (int): The number of parallel attention heads. Defaults to 8.
        attention_type (str): Type of attention to use (caml or multihead). Defaults to 'multihead'.
        attention_dropout (float): The dropout rate for the attention. Defaults to 0.0.
    """
    def __init__(
        self,
        num_classes,
        dropout=0.2,
        lm_weight='bert-base-cased',
        lm_window=512,
        num_heads=8,
        attention_type='multihead',
        attention_dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.lm_window = lm_window
        self.attention_type = attention_type

        self.lm = AutoModel.from_pretrained(lm_weight, torchscript=True)
        self.embed_drop = nn.Dropout(p=dropout)

        assert attention_type in ['singlehead', 'multihead'], "attention_type must be 'singlehead' or 'multihead'"
        if attention_type == 'singlehead':
            self.attention = LabelwiseAttention(self.lm.config.hidden_size, num_classes)
        else:
            self.attention = LabelwiseMultiHeadAttention(
                self.lm.config.hidden_size, num_classes, num_heads, attention_dropout)

        # Final layer: create a matrix to use for the #labels binary classifiers
        self.output = LabelwiseLinearOutput(self.lm.config.hidden_size, num_classes)

    def lm_feature(self, input_ids):
        """BERT takes an input of a sequence of no more than 512 tokens.
        Therefore, long sequence are split into subsequences of size `lm_window`, which is a number no greater than 512.
        If the split subsequence is shorter than `lm_window`, pad it with the pad token.

        Args:
            input_ids (torch.Tensor): Input ids of the sequence with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The representation of the sequence.
        """
        if input_ids.size(-1) <= self.lm_window:
            return self.lm(input_ids, attention_mask=input_ids != self.lm.config.pad_token_id)[0]
        else:
            inputs = []
            batch_indexes = []
            seq_lengths = []
            for token_id in input_ids:
                indexes = []
                seq_length = (token_id != self.lm.config.pad_token_id).sum()
                seq_lengths.append(seq_length)
                for i in range(0, seq_length, self.lm_window):
                    indexes.append(len(inputs))
                    inputs.append(token_id[i: i + self.lm_window])
                batch_indexes.append(indexes)

            padded_inputs = pad_sequence(inputs, batch_first=True)
            last_hidden_states = self.lm(
                padded_inputs, attention_mask=padded_inputs != self.lm.config.pad_token_id)[0]

            x = []
            for seq_l, mapping in zip(seq_lengths, batch_indexes):
                last_hidden_state = last_hidden_states[mapping].view(
                    -1, last_hidden_states.size(-1))[:seq_l, :]
                x.append(last_hidden_state)
            return pad_sequence(x, batch_first=True)

    def forward(self, input):
        input_ids = input['text'] # (batch_size, sequence_length)
        attention_mask = input_ids == self.lm.config.pad_token_id
        x = self.lm_feature(input_ids) # (batch_size, sequence_length, lm_hidden_size)
        x = self.embed_drop(x)

        # Apply per-label attention.
        logits, attention = self.attention(x, attention_mask)

        # Compute a probability for each label
        x = self.output(logits)
        return {'logits': x, 'attention': attention}

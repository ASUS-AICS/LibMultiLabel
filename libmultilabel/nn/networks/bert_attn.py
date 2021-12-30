import torch
from torch.nn.init import xavier_uniform_
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel


class BERTAttention(nn.Module):
    """BERT Attention model.

    Args:
        num_classes (int): Total number of classes.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        embedding_dim (int): Embedding dimension. Defaults to 512.
        lm_weight (str): Pretrained model name or path. Defaults to 'bert-base-cased'.
        lm_window (int): Length of the subsequences to be split before feeding them to
            the language model. Defaults to 512.
        num_heads (int): Number of parallel attention heads. Defaults to 2.
    """
    def __init__(
        self,
        num_classes,
        dropout=0.2,
        embedding_dim=512,
        lm_weight='bert-base-cased',
        lm_window=512,
        num_heads=2
    ):
        super().__init__()
        self.lm_window = lm_window
        self.embedding_dim = embedding_dim

        self.lm = AutoModel.from_pretrained(lm_weight, torchscript=True)
        self.lm_linear = nn.Linear(self.lm.config.hidden_size, embedding_dim)
        self.lm_final = nn.Linear(embedding_dim, num_classes)

        self.query = nn.Parameter(torch.Tensor(1, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.embed_drop = nn.Dropout(p=dropout)

        """Context vectors for computing attention with
        (in_features, out_features) = (lm_hidden_size, num_classes) -> lm_linear
        """
        self.U = nn.Linear(self.lm.config.hidden_size, num_classes)
        xavier_uniform_(self.U.weight)

        # Final layer: create a matrix to use for the #labels binary classifiers  -> lm_final
        self.final = nn.Linear(self.lm.config.hidden_size, num_classes)
        xavier_uniform_(self.final.weight)

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
        x = self.lm_feature(input_ids) # (batch_size, sequence_length, lm_hidden_size)

        # attention_mask = input_ids == self.lm.config.pad_token_id
        x = self.embed_drop(x)
        # x = self.lm_linear(x)  # (batch_size, sequence_length, embedding_dim)

        # k = v = x.permute(1, 0, 2) # (sequence_length, batch_size, embedding_dim)
        # q = self.query.repeat(1, input_ids.size(0), 1) # (1, batch_size, embedding_dim)

        # output, mulit_head_alpha = self.attention(query=q, key=k, value=v, key_padding_mask=attention_mask)
        # logits = self.lm_final(output.squeeze(0))

        """Apply per-label attention. The shapes are:
           - U.weight: (num_classes, lm_hidden_size)
           - matrix product of U.weight and x: (batch_size, num_classes, length)
           - alpha: (batch_size, num_classes, length)
        """
        alpha = torch.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        # Document representations are weighted sums using the attention
        m = alpha.matmul(x)  # (batch_size, num_classes, lm_hidden_size)

        # Compute a probability for each label
        x = self.final.weight.mul(m).sum(dim=2).add(
            self.final.bias)  # (batch_size, num_classes)

        return {'logits': x, 'attention': alpha}
        # return {'logits': logits}

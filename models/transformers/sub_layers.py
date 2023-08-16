import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v):
        """
        d_model: dimension model
        @d_q: dimension query
        @d_v: dimension value
        @d_k: dimension key

        """
        super(SelfAttention, self).__init__()
        # assert d_model % n_head == 0 ,"d_model must divibe by n_head"
        # self.d_k = d_model // n_head
        assert d_q == d_k, "d_q not equal d_k"
        self.d_q = d_q
        self.d_v = d_v
        self.linear_q = nn.Linear(d_model, self.d_q)
        self.linear_k = nn.Linear(d_model, self.d_q)
        self.linear_v = nn.Linear(d_model, self.d_v)

    def forward(self, inp_q, inp_k, inp_v, mask=None):
        """
        @inp_q: input query, shape [batch_size,seq_len_1, d_model]
        @inp_k: input key, shape [batch_size, seq_len_2, d_model]
        @inp_v: input value, shape [batch_size, seq_len_3, d_model]

        if mask-selfattention: seq_len_2 = seq_len_3
        else: seq_len1 = seq_len2 = seq_len3
        """
        query = self.linear_q(inp_q)  # shape: [batch_size, seq_len_1, d_q]
        key = self.linear_k(inp_k)  # shape: [batch_size, seq_len_2, d_k]
        value = self.linear_v(inp_v)  # shape: [batch_size, seq_len_3, d_v]
        matrix_score = torch.bmm(query, key.transpose(-1, -2)) / np.sqrt(
            self.d_q
        )  # shape: [batch_size, seq_len_1,seq_len_2]
        if mask is not None:
            matrix_score = matrix_score.masked_fill(mask == 0, -1e9)
        weight = torch.softmax(
            matrix_score, dim=-1
        )  # shape: [batch_size, seq_len_1,seq_len_2]
        out = torch.bmm(weight, value)  # shape: [batch_size, seq_len_1 , d_model]
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        """
        @d_model: dimension model
        @n_head: number of heads attention
        """

        super(MultiheadAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.multihead = nn.ModuleList(
            [
                SelfAttention(d_model, self.d_k, self.d_k, self.d_k)
                for _ in range(n_head)
            ]
        )

    def forward(self, inp_q, inp_k, inp_v, mask=None):
        """
        @inp_q: input query
        @inp_k: input key
        @inp_v: input value

        """
        outps = []
        for layer in self.multihead:
            out = layer(inp_q, inp_k, inp_v, mask)
            outps.append(out)
        output_multihead = torch.cat(outps, dim=-1)
        return output_multihead

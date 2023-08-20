import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .sub_layers import MultiheadAttention


class Norm(nn.Module):
    """Calibrate normalisation"""

    def __init__(self, d_model, eps=1e-6):
        """_summary_

        Args:
            d_model (_type_): _description_
            eps (_type_, optional): _description_. Defaults to 1e-6.
        """
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


class EncoderLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, n_head, n_hidden, dropout=0.1):
        """
        @d_model: dimension model
        @n_head: number of heads attention
        @n_hidden: hidden state of linear layer
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, n_head)
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.linear2 = nn.Linear(n_hidden, d_model)
        self.layer_norm = Norm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, inp, mask=None):
        """
        @inp: input encode layer, shape [batch_size, seq_length, d_model]
        """
        out_attention = self.attention(
            inp, inp, inp, mask
        )  # shape [batch_size, seq_length, d_model]
        out_attention = self.dropout1(out_attention)
        out_sublayer = inp + out_attention  # shape [batch_size, seq_length, d_model]
        out_layernorm = self.layer_norm(out_sublayer)
        out_feedforward = self.linear1(
            out_layernorm
        )  # shape [batch_size, seq_length, n_hidden]
        out_feedforward = self.relu(out_feedforward)
        outp = self.linear2(out_feedforward)  # shape [batch_size, seq_length, d_model]
        outp = self.layer_norm(self.dropout2(outp) + out_layernorm)
        return outp  # shape [batch_size, seq_length, d_model]


class DecoderLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, n_head, n_hidden, dropout=0.1):
        """
        @d_model: dimension model
        @n_head: number of heads attention
        @n_hidden: hidden state of linear layer
        """
        super(DecoderLayer, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.linear2 = nn.Linear(n_hidden, d_model)
        self.mask_attention = MultiheadAttention(d_model, n_head)
        self.att = MultiheadAttention(d_model, n_head)
        self.layer_norm1 = Norm(d_model)
        self.layer_norm2 = Norm(d_model)
        self.layer_norm3 = Norm(d_model)
        self.relu = nn.ReLU()

    def forward(self, inp_decode, inp_memory, mask_tgt=None, mask_src=None):
        """
        @inp_decode: shape [batch_size, seq_len_target, d_model]
        @inp_memory: shape [batch_size, seq_len_source, d_model]
        """
        out_maskatt = self.mask_attention(
            inp_decode, inp_decode, inp_decode, mask=mask_tgt
        )
        out_maskatt = self.layer_norm1(inp_decode + self.dropout1(out_maskatt))
        out_att = self.att(out_maskatt, inp_memory, inp_memory, mask=mask_src)
        out_att = self.layer_norm2(out_maskatt + self.dropout2(out_att))
        out_feedforward = self.linear1(out_att)
        out_feedforward = self.linear2(self.relu(out_feedforward))
        outp = self.layer_norm3(self.dropout3(out_feedforward) + out_att)
        return outp


class PositionalEncoding(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        d_model,
        device,
        dropout=0.1,
        max_len=1024,
    ):
        """
        parameters:
        @d_model: size of vector throughtout the transformer model
        @device: cuda or cpu
        @dropout: dropout probability
        @max_len: max length of input sentence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #         self.device = device
        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device)
        self.d_model = d_model

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = x * np.sqrt(self.d_model) + Variable(
            self.pe[:, : x.size(1)], requires_grad=False
        )
        return self.dropout(x)

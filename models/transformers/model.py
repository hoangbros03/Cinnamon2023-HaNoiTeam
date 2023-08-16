from __future__ import absolute_import
import torch.nn as nn
import torch
from .layers import EncoderLayer, DecoderLayer, PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self, n_vocab, d_model, n_hidden, n_head, n_layer, device="cpu", dropout=0.1
    ):
        """
        @n_vocab: size vocabulary of source language
        @d_model: size of vector throughtout the transformer model
        @n_hidden: size of hidden state in linear layers
        @n_head: number of head attention
        @n_layer: number of encoder layer
        @device: cuda or cpu
        @dropout: dropout probability
        """
        super(Encoder, self).__init__()
        self.embedd = nn.Embedding(n_vocab, d_model)
        self.pe = PositionalEncoding(d_model, device, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, n_hidden, dropout) for _ in range(n_layer)]
        )

    def forward(self, inp, mask=None):
        embedd = self.embedd(inp)  # shape: [batch_size, seq_len_src, d_model]
        out = self.pe(embedd)  # shape: [batch_size, seq_len_src, d_model]
        for layer in self.encoder_layers:
            out = layer(out, mask)
        return out  # shape: [batch_size, seq_len_src, d_model]


class Decoder(nn.Module):
    def __init__(
        self, n_vocab, d_model, n_hidden, n_head, n_layer, device="cpu", dropout=0.1
    ):
        """
        @n_vocab: size vocabulary of target language
        @d_model: size of vector throughtout the transformer model
        @n_hidden: size of hidden state in linear layers
        @n_head: number of head attention
        @n_layer: number of encoder layer
        @device: cuda or cpu
        @dropout: dropout probability
        """
        super(Decoder, self).__init__()
        self.embedd = nn.Embedding(n_vocab, d_model)
        self.pe = PositionalEncoding(d_model, device, dropout)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, n_hidden, dropout) for _ in range(n_layer)]
        )
        self.linear = nn.Linear(d_model, n_vocab)

    def forward(self, inp_decode, memory, tgt_mask=None, src_mask=None):
        embedd = self.embedd(inp_decode)  # shape: [batch_size, seq_len_tgt, d_model]
        out = self.pe(embedd)  # shape: [batch_size, seq_len_tgt, d_model]
        for layer in self.decoder_layers:
            out = layer(
                out, memory, tgt_mask, src_mask
            )  # shape: [batch_size, seq_len_tgt, d_model]
        out = self.linear(out)  # shape: [batch_size, seq_len_tgt, n_vocab]
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        n_vocab_src,
        n_vocab_tgt,
        d_model,
        n_hidden,
        n_head,
        n_layer,
        src_pad,
        device="cpu",
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            n_vocab_src, d_model, n_hidden, n_head, n_layer, device, dropout
        )
        self.decoder = Decoder(
            n_vocab_tgt, d_model, n_hidden, n_head, n_layer, device, dropout
        )
        self.device = device
        self.src_pad = src_pad
        # self.tgt_pad = tgt_pad
        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, inp_src, inp_tgt):
        src_mask = (inp_src != self.src_pad).unsqueeze(1).to(self.device)
        tgt_mask = (
            1 - torch.triu(torch.ones(inp_tgt.size(1), inp_tgt.size(1)), diagonal=1)
        ).to(self.device)
        outp_encode = self.encoder(inp_src)
        outp_decode = self.decoder(inp_tgt, outp_encode, tgt_mask, src_mask)
        return outp_decode


class LabelSmooth(nn.Module):
    def __init__(self, eps=0.1):
        super(LabelSmooth, self).__init__()
        self.eps = eps
        # self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, predicts, targets):
        """
        @inputs: shape [batch_size, tgt_max_length, n_vocab]
        @targets: shape [batch_size, tgt_max_length]
        """
        n_vocab = predicts.size(-1)
        predicts = predicts.reshape(-1, n_vocab)
        targets = targets.reshape(-1, 1)
        assert predicts.size(0) == targets.size(0), "shape not equal"
        # predicts = torch.softmax(predicts, dim= -1)
        one_hot = torch.zeros_like(predicts)
        one_hot.scatter_(1, targets, 1)
        one_hot = one_hot * (1 - self.eps) + self.eps / n_vocab
        loss = (-1 * one_hot * torch.log_softmax(predicts, dim=-1)).sum(dim=1)
        loss = torch.mean(loss)
        return loss

'''
Attention class
use different methods to calculate attention score
written by LAQ
2019.7.25
'''

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, mode, batch_size, seq_len, hidden_dim):
        super(Attention, self).__init__()
        assert mode in ['dot_product', 'bilinear', 'perceptron']
        self.mode = mode
        self.tanh = nn.Tanh()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        if self.mode == 'bilinear':
            self.w_bilinear = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
            self.w_bilinear = self.w_bilinear.repeat(self.batch_size, 1, 1)
        if self.mode == 'perceptron':
            self.w_1 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
            self.w_2 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
            self.v = nn.Parameter(torch.Tensor(1, self.hidden_dim))
            self.w_1 = self.w_1.repeat(self.batch_size, 1, 1)  # [batch_size, rnn_out_dim, rnn_out_dim]
            self.w_2 = self.w_2.repeat(self.batch_size, 1, 1)
            self.v = self.v.repeat(self.batch_size, 1, 1)  # [batch_size, 1, rnn_out_dim]

    def calculate_score(self, rnn_out, state):
        assert self.batch_size == rnn_out.size(0)
        assert self.seq_len == rnn_out.size(1)
        assert self.hidden_dim == rnn_out.size(2)

        if self.mode == 'dot_product':
            weights = torch.bmm(rnn_out, state)
            # print(weights.size())
        if self.mode == 'bilinear':
            weights = torch.bmm(torch.bmm(rnn_out, self.w_bilinear), state)
        if self.mode == 'perceptron':
            rnn_out_tran = torch.transpose(torch.bmm(self.w_1, torch.transpose(rnn_out, 1, 2)), 1, 2)
            # print(rnn_out_tran.size())
            state_tran = torch.bmm(self.w_2, state).squeeze(2).unsqueeze(1).repeat(1, self.seq_len, 1)
            # print(state_tran.size())
            weight = rnn_out_tran + state_tran
            # print(weight.size())
            weights = torch.transpose(torch.bmm(self.v, self.tanh(torch.transpose((weight), 1, 2))), 1, 2)
            # print(weights.size())
        return weights


    def forward(self, rnn_out):  # rnn_out.size:[batch_size, seq_len, hidden_size]
        attn_out = torch.zeros_like(rnn_out)
        seq_len = rnn_out.size(1)
        for i in range(seq_len):
            state = rnn_out[:, i, :].unsqueeze(0)
            # print('state',state.size())
            # print(state)
            merged_state = torch.cat([s for s in state], 1)
            # print('merged_state',merged_state.size())
            # print(merged_state)
            merged_state = merged_state.squeeze(0).unsqueeze(2)
        # print(merged_state.size())
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
            weights = self.calculate_score(rnn_out, merged_state)
            weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # print('weight',weights.size())
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
            attn = torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)
            attn_out[:, i, :] = attn
    # print('attn',attn.size())
        return attn_out


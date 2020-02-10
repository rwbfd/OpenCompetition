# !/user/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from file_utils import TaskModelBase
from .multi_attention_type import Attention


class StackedBiLSTM(TaskModelBase):

    def __init__(self, input_size, hidden_size, num_layers, dropout_prob,
                 mode="dot_product", residual=True, attention=True):
        super(StackedBiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.mode = mode
        self.residual = residual
        self.attention = attention

        self.drop = nn.Dropout(dropout_prob)

        for i in range(num_layers):
            lstm_layer = nn.LSTM(input_size, hidden_size[i], batch_first=True,
                                 bidirectional=True, dropout=dropout_prob)
            setattr(self, f'lstm_layer_{i}', lstm_layer)

    def get_lstm_layer(self, i):
        return getattr(self, f'lstm_layer_{i}')

    def forward(self, input):

        for i in range(self.num_layers):
            if i == 0:
                lstm_in = input
            else:
                if self.config.residual:
                    lstm_in = input + lstm_out
                else:
                    lstm_in = lstm_out
            lstm_layer = self.get_lstm_layer(i)
            lstm_out, _ = lstm_layer(lstm_in)  # [batch, seq_len, 2*hidden]
            two2emblin = nn.Linear(2 * self.hidden_size[i], self.input_size)
            lstm_out = self.drop(two2emblin(lstm_out))  # [batch, seq_len, emb_dim]

            if self.attention:
                attention = Attention(self.mode, lstm_out.size(0), lstm_out.size(1), lstm_out.size(2))
                attn_out = attention(lstm_out)  # [batch, seq_len, emb_dim]
                lstm_attn_out = torch.cat([attn_out, lstm_out], 2)  # [batch, seq_len, 2*emb_dim]
                lin_layer = nn.Linear(2 * self.input_size, self.input_size)
                lstm_out = self.drop(lin_layer(lstm_attn_out))

        return lstm_out



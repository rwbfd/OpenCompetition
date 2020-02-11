# -- coding: utf-8 --
# @Time : 2019/7/23
# @Author : lha
# @File : main.py 

import os
import sys
import torch
import torch.nn as nn
import argparse
from crf import CRF
from file_utils import TaskModelBase


class Bilstm_Crf(TaskModelBase):
    def __init__(self, args, vocab_size, embed_dim, vec_list, label2id):
        super(Bilstm_Crf, self).__init__()
        self.input_size = embed_dim
        self.hidden_size = args.hidden_size
        self.out_size = args.out_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.label2id = label2id

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(vec_list))
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.out_size)
        self.drop = nn.Dropout(0.5)
        self.crf = CRF(len(self.label2id))

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.bilstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=128)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)

        return x

    def loss(self, feats, tags):
        return self.crf.loss(feats, tags)

    def predict(self, x, lengths):
        feats = self.forward(x, lengths=lengths)
        return self.crf(feats)

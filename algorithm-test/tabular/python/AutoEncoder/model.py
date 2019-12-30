import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self, n_inp):
        super(DenoisingAutoencoder, self).__init__()
        n_hidden = 600
        self.encoder = nn.Linear(n_inp, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_inp)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = F.sigmoid(self.decoder(encoded))
        return decoded
import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import torch
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def log_likelihood_samples_mean_sigma(samples, mean, logvar, dim):

    constant = torch.log(torch.FloatTensor(np.asarray([np.pi]))*2)
    return - constant[0] * samples.shape[dim] * 0.5  - \
               torch.sum(((samples-mean)/torch.exp(logvar*0.5))**2 + logvar, dim=dim) * 0.5

def prior_z(samples, dim):

    constant = torch.log(torch.FloatTensor(np.asarray([np.pi]))*2)
    return - constant[0]*samples.shape[dim] * 0.5 - torch.sum(samples**2, dim=dim) * 0.5


def log_mean_exp(x, dim=1):

    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m), dim=dim, keepdim=True))



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
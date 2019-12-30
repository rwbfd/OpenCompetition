import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import matplotlib.pyplot as plt
import numpy as np

from model import DenoisingAutoencoder

# global constants
BATCH_SIZE = 32
N_INP = 784
N_EPOCHS = 10
NOISE = 0.5

# MNIST data loading
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=False)

auto_encoder = DenoisingAutoencoder(N_INP)
optimizer = optim.Adam(auto_encoder.parameters(), lr=1e-3)

# set plot and view data for visualization
N_COLS = 8
N_ROWS = 4
view_data = []
for i in range(N_ROWS * N_COLS):
    noise = np.random.choice([1, 0], size=(28, 28), p=[NOISE, 1 - NOISE])
    view_data.append(test_set[i][0] * torch.FloatTensor(noise))
plt.figure(figsize=(20, 4))

for epoch in range(N_EPOCHS):
    for b_index, (x, _) in enumerate(train_loader):
        y = x.view(x.size()[0], -1)
        noise = np.random.choice([1, 0], size=(BATCH_SIZE, N_INP), p=[NOISE, 1 - NOISE])
        inp = y * torch.FloatTensor(noise)
        inp = Variable(inp)
        decoded = auto_encoder(inp)
        MSE_loss = (y - decoded) ** 2
        loss = MSE_loss.view(1, -1).sum(1) / BATCH_SIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: [%3d], Loss: %.4f" %(epoch + 1, loss.data))

for i in range(N_ROWS * N_COLS):
    # original image
    r = i // N_COLS
    c = i % N_COLS + 1
    ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c)
    plt.imshow(view_data[i].squeeze())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstructed image
    ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c + N_COLS)
    x = Variable(view_data[i])
    y = auto_encoder(x.view(1, -1))
    plt.imshow(y.detach().squeeze().numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
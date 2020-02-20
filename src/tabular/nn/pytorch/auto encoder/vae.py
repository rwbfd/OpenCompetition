
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from src.common.pytorch.activation.activation import gelu,swish,mish,gelu_bert

# VAE model
class VAE(nn.Module):
    def __init__(self, args, net):
        super(VAE, self).__init__()
        self.layer1 = net(args.image_size, args.h_dim)
        self.layer2 = net(args.h_dim, args.z_dim)
        self.layer3 = net(args.h_dim, args.z_dim)
        self.layer4 = net(args.z_dim, args.h_dim)
        self.layer5 = net(args.h_dim, args.image_size)


    def encode(self, x, activation='relu'):
        if activation == 'relu':
            h = F.relu(self.layer1(x))
            return self.layer2(h), self.layer3(h)
        elif activation == 'hardtanh':
            h = F.hardtanh(self.layer1(x))
            return self.layer2(h), self.layer3(h)
        elif activation == 'relu6':
            h = F.relu6(self.layer1(x))
            return self.layer2(h), self.layer3(h)
        elif activation == 'leaky_relu':
            h = F.leaky_relu(self.layer1(x))
            return self.layer2(h), self.layer3(h)

        elif activation == 'gelu':
            h = gelu(self.layer1(x))
            return self.layer2(h), self.layer3(h)
        elif activation == 'swish':
            h = swish(self.layer1(x))
            return self.layer2(h), self.layer3(h)
        elif activation == 'mish':
            h = mish(self.layer1(x))
            return self.layer2(h), self.layer3(h)
        elif activation == 'gelu_bert':
            h = gelu_bert(self.layer1(x))
            return self.layer2(h), self.layer3(h)



    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, activation='relu'):
        if activation == 'relu':
            h = F.relu(self.layer4(z))
            return F.sigmoid(self.layer5(h))
        elif activation == 'hardtanh':
            h = F.hardtanh(self.layer4(z))
            return F.sigmoid(self.layer5(h))
        elif activation == 'relu6':
            h = F.relu6(self.layer4(z))
            return F.sigmoid(self.layer5(h))
        elif activation == 'leaky_relu':
            h = F.leaky_relu(self.layer4(z))
            return F.sigmoid(self.layer5(h))

        elif activation == 'gelu':
            h = gelu(self.layer4(z))
            return F.sigmoid(self.layer5(h))
        elif activation == 'swish':
            h = swish(self.layer4(z))
            return F.sigmoid(self.layer5(h))
        elif activation == 'mish':
            h = mish(self.layer4(z))
            return F.sigmoid(self.layer5(h))
        elif activation == 'gelu_bert':
            h = gelu_bert(self.layer4(z))
            return F.sigmoid(self.layer5(h))


    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


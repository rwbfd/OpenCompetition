import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


class NormedLinear(nn.Model):
    def __init__(self,input_dim,out_dim,momentum=0.1):
        super(NormedLinear,self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.momentum = momentum
        self._build_model()

    def _build_model(self):
        self.linear = nn.utils.weight_norm(nn.Linear(self.input_dim,self.out_dim))
        self.bias = Parameter(torch.Tensor)
        self.register_buffer('running_mean',torch.zeros(self.out_dim))
        self.reset_parameter()


    def reset_parameter(self):
        self.running_mean.zero_()
        self.bias.data.zero_()

    def forward(self,inputs):
        inputs = self.linear(inputs)

        if self.training:
            avg = torch.mean(inputs,dim=0)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * avg.data
        else:
            avg = Variable(self.running_mean,requires_grad=False)
        out = inputs - avg + self.bias
        return out
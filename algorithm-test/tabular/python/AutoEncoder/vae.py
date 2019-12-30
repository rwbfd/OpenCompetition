import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image
# 加载数据集
def get_data():
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_data = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)    # 均值
        self.fc22 = nn.Linear(400, 20)    # 方差
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        x = F.tanh(self.fc4(h3))
        return x
    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()   # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()    # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

if __name__ == '__main__':
    # 超参数设置
    batch_size = 128
    lr = 1e-3
    epoches = 100

    model = VAE()
    if torch.cuda.is_available():
        model.cuda()

    train_data = get_data()

    reconstruction_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        for img, _ in train_data:
            img = img.view(img.size(0), -1)
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            # forward
            output, mu, logvar = model(img)
            loss = loss_function(output, img, mu, logvar)/img.size(0)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch=", epoch, loss.data.float())
        if (epoch+1) % 10 == 0:
            print("epoch = {}, loss is {}".format(epoch+1, loss.data))
            pic = to_img(output.cpu().data)
            if not os.path.exists('./vae_img1'):
                os.mkdir('./vae_img1')
            save_image(pic, './vae_img1/image_{}.png'.format(epoch + 1))
    torch.save(model, './vae.pth')
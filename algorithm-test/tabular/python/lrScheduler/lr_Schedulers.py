import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim, nn

from optimizer.torchtools.lr_scheduler import DelayerScheduler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#-----------------Fully connected neural network with one hidden layer-------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        nn.BatchNorm1d(3)
        #self.relu6 = nn.ReLU6()
        #self.elu = nn.ELU()
        #self.prelu = nn.PReLU()
        #self.leakyrelu = nn.LeakyReLU(0.1)
        nn.Dropout(p=0.5)
        # nn.Dropout2d(p=0.5, inplace=False)
        # nn.Dropout3d(p=0.5, inplace=False)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.001) # define here your optimizer, the lr that you set will be the one used for the initial delay steps

delay_epochs = 10
total_epochs = 20
base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, delay_epochs) # delay the scheduler for 10 steps
delayed_scheduler = DelayerScheduler(optimizer, total_epochs - delay_epochs, base_scheduler)


criterion = nn.CrossEntropyLoss()
total_step = len(train_loader)
for epoch in range(total_epochs):
	# train(...)
    delayed_scheduler.step()
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))



# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


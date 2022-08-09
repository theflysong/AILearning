import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import network
import torch.optim as optim
import torch.nn.functional as F
import os

# 参数

batch_size_training = 64
batch_size_testing = 1000
momentum = 0.5
learning_rate = 0.02
log_freq = 10
random_seed = 114514
global_average = 0.1307
standard_error = 0.3081

DoTrain = input("Do training?(y/N):") == 'y'
if DoTrain:
    epochs = int(input("Input training epochs:"))
else:
    epochs = 3

model_state_dict_name = "model.pth"
optimizer_state_dict_name = "optimizer.pth"

# 配置
torch.manual_seed(random_seed)

loader_training = DataLoader(
    torchvision.datasets.MNIST(
        './data/', train=True, download=True, transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((global_average,), (standard_error,))
        ]),
    ),
    batch_size = batch_size_training, shuffle=True
)

loader_testing = DataLoader(
    torchvision.datasets.MNIST(
        './data/', train=False, download=True, transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((global_average,), (standard_error,))
        ]),
    ),
    batch_size = batch_size_testing, shuffle=True
)

# 构建网络

net = network.Network()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

if os.path.exists(model_state_dict_name):
    network_state_dict = torch.load(model_state_dict_name)
    net.load_state_dict(network_state_dict)

if os.path.exists(optimizer_state_dict_name):
    optimizer_state_dict = torch.load(optimizer_state_dict_name)
    optimizer.load_state_dict(optimizer_state_dict)

# 训练
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(loader_training.dataset) for i in range(epochs + 1)]

def doLog(epoch, index, data, loss):
    print(
        f"Train Epoch: {epoch} [{index * len(data)}/{len(loader_training.dataset)} \
            ({100 * index / len(loader_training):.0f}%)]    Loss: {loss.item():.6f}"
    )

    train_losses.append(loss.item())

    train_counter.append(
        (index * batch_size_training) + ((epoch - 1) * len(loader_training.dataset)))

    torch.save(net.state_dict(), model_state_dict_name)
    torch.save(optimizer.state_dict(), optimizer_state_dict_name)

def train(epoch):
    net.train()
    for index, (data, target) in enumerate(loader_training):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if index % log_freq == 0:
            doLog(epoch, index, data, loss)

def test():
    net.eval()
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in loader_testing:
            output = net(data)
            test_loss += F.nll_loss(output, target)
            predict = output.data.max(1, keepdim=True)[1]
            acc += predict.eq(target.data.view_as(predict)).sum()
    test_loss /= len(loader_testing.dataset)
    test_losses.append(test_loss)
    print(f"Test set: Avg. loss: {test_loss:.4f}, Accuracy: {acc}/{len(loader_testing.dataset)} ({acc/len(loader_testing.dataset) * 100:.0f}%)\n")

def doPred():
    examples = enumerate(loader_testing)
    index, (eg_data, eg_targets) = next(examples)
    with torch.no_grad():
        output = net(eg_data)
    figure = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(eg_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

test()
doPred()

if DoTrain:
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
        doPred()

exit(0)
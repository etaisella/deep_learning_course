import torch as tr
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

training_data = datasets.FashionMNIST("", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))

train_set = tr.utils.data.DataLoader(training_data, batch_size=10, shuffle=True)

for data in train_set:
    # viewing the first item in the training batch
    print(data[1][0])
    plt.imshow(data[0][0].view(28,28))
    plt.show()
    break

lenet = nn.Sequential()
lenet.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2))
lenet.add_module("tanh1", nn.Tanh())
lenet.add_module("avg_pool1", nn.AvgPool2d(kernel_size=2, stride=2))
lenet.add_module("conv2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1))
lenet.add_module("avg_pool2", nn.AvgPool2d(kernel_size=2, stride=2))
lenet.add_module("conv3", nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5,stride=1))
lenet.add_module("tanh2", nn.Tanh())
lenet.add_module("flatten", nn.Flatten(start_dim=1))
lenet.add_module("fc1", nn.Linear(in_features=120 , out_features=84))
lenet.add_module("tanh3", nn.Tanh())
lenet.add_module("fc2", nn.Linear(in_features=84, out_features=10))
print(lenet)

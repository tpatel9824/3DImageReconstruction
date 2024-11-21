import torch.nn as nn
import torch.nn.functional as F
import  torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(15,30,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(30,40,4)
        self.pool2 = nn.MaxPool2d(12, 12)
        self.fc1=nn.Linear(2560,140)
        self.fc2=nn.Linear(140,90)
        self.fc3=nn.Linear(90,9)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x


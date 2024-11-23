import torch.nn.functional as F
import  torch
import torch.nn as nn
import torch.optim as optim
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1=nn.Conv2d(3,64,7)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(64,128,3)
        self.pool2=nn.MaxPool2d(2,2)
        self.conv3=nn.Conv2d(128,256,3)
        self.conv4=nn.Conv2d(256,512,3)
        self.conv5=nn.Conv2d(512,1024)
    def forward(self):
        pass
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()



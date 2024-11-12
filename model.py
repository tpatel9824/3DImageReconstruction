import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CNN2Dto3D(nn.Module):
    def __init__(self):
        super(CNN2Dto3D, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        
        self.output = nn.Linear(512, 28 * 28 * 28)  
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        x = self.output(x)
        
        x = x.view(-1, 28, 28, 28) 
        return x


def train(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in data_loader:
           
            images, targets = images.cuda(), targets.cuda()
            
        
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss = running_loss + loss.item()
        
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}')

num_epochs = 10
learning_rate = 0.001

model = CNN2Dto3D().cuda()  
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, data_loader, criterion, optimizer, num_epochs)

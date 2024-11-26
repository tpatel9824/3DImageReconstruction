import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(100352, 1024)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2, batch_first=True)
        #self.fc_voxel = nn.Linear(512, 50 * 50 * 50)  # Flatten voxel grid to a vector
        self.fc = nn.Linear(1024, 512 * 4 * 4 * 4)  # Map latent vector to initial 3D feature map

        self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)  # 8x8x8
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)  # 16x16x16
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)  # 32x32x32
        self.deconv4 = nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1)  # 50x50x50

    def forward(self, x):
        #x = x.unsqueeze(1)
        #lstm_out, (h_n, c_n) = self.lstm(x)
        #lstm_out = lstm_out[:, -1, :]  # Take the last time-step output
        #voxel = self.fc_voxel(lstm_out)
        #voxel = voxel.view(-1, 1, 50, 50, 50)  # Reshape to voxel grid
        x = self.fc(x)  # (batch_size, 512 * 4 * 4 * 4)
        x = x.view(-1, 512, 4, 4, 4)  # Reshape to (batch_size, channels, depth, height, width)

        # 3D transposed convolutions with ReLU activation
        x = F.relu(self.deconv1(x))  # Output size: (batch_size, 256, 8, 8, 8)
        x = F.relu(self.deconv2(x))  # Output size: (batch_size, 128, 16, 16, 16)
        x = F.relu(self.deconv3(x))  # Output size: (batch_size, 64, 32, 32, 32)
        x = torch.sigmoid(self.deconv4(x))  # Final voxel grid, (batch_size, 1, 50, 50, 50)
        x = x[:, :, 7:57, 7:57, 7:57]

        return x

class R2N2(nn.Module):
    def __init__(self):
        super(R2N2, self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
    def forward(self,x):
        features=self.encoder(x)
        voxels=self.decoder(features)
        return voxels





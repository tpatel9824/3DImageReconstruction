import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from main import  DDataset
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import os

class VoxelReconstructionModel(nn.Module):
    def __init__(self, voxel_dim):
        super(VoxelReconstructionModel, self).__init__()
        self.voxel_dim = voxel_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.feature_size = None
        self.view_aggregator = None

        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, voxel_dim ** 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_views, c, h, w = x.shape

        x = x.view(-1, c, h, w)
        features = self.feature_extractor(x)

        if self.feature_size is None:
            self.feature_size = features.size(1) * features.size(2) * features.size(3)
            self.view_aggregator = nn.Linear(self.feature_size, 1024).to(features.device)
            print(f"Initialized view_aggregator with input size {self.feature_size}")

        features = features.view(features.size(0), -1)

        features = features.view(batch_size, num_views, -1).mean(dim=1)

        if self.view_aggregator is None:
            raise RuntimeError("view_aggregator is not initialized")
        features = self.view_aggregator(features)

        voxels = self.decoder(features)
        voxels = voxels.view(-1, self.voxel_dim, self.voxel_dim, self.voxel_dim)
        return voxels


def train_model(model, dataloader,validation_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.01)
    Loss1=[]
    Loss2=[]
    Epoch=[]
    for epoch in range(epochs):
        Epoch.append(epoch)
        total_loss = 0
        count=0
        valid_loss=0
        for images, voxels,label in dataloader:
            images, voxels,label = images.to(device), voxels.to(device),label.to(device)
            voxels = voxels.squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, voxels)
            loss.backward()
            optimizer.step()
            count+=1
            total_loss += loss.item()
        Loss1.append(total_loss)
        for val_image,voxel_val,label_val in validation_loader:
            val_images, val_voxels, val_label = val_image.to(device), voxel_val.to(device), label_val.to(device)
            target=model(val_images)
            valid_loss+=criterion(target,val_voxels.squeeze(1)).item()
        Loss2.append(valid_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {total_loss / len(dataloader)}, Validation Loss : {valid_loss / len(validation_loader)}")

    fig,axs=plt.subplots(1,2)
    axs[0].plot(Epoch,Loss1,color='blue')
    axs[0].set_title("Training Loss Over Epochs")
    axs[1].plot(Epoch,Loss2,color='red')
    axs[1].set_title("Validation Loss Over Epochs")
    for ax in axs.flat:
        ax.set(xlabel='Epoch',ylabel='Loss')
    plt.show()



def predict(model, image_batch, device):
    model.eval()
    with torch.no_grad():
        image_batch = image_batch.to(device)
        predictions = model(image_batch)
        return predictions.cpu().numpy()




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    directory = r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Training'
    DataFrame = pd.read_csv(r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Training\Annotations.csv')
    Data = DDataset(DataFrame, directory)
    train_loader = torch.utils.data.DataLoader(Data, batch_size=7, num_workers=2, shuffle=True)

    directory1 = r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Validation'
    DataFrame1 = pd.read_csv(r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Validation\Annotations.csv')
    Data1 = DDataset(DataFrame1, directory1)
    validation_loader = torch.utils.data.DataLoader(Data1, batch_size=5, num_workers=2, shuffle=True)

    voxel_dim = 50
    model = VoxelReconstructionModel(voxel_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader,validation_loader, epochs=5, lr=0.001, device=device)
    directory = r'C:\Users\91875\OneDrive\Desktop\3D_OBJECT\3DImageReconstruction\SAVED_MODELS'
    os.makedirs(directory, exist_ok=True)

    PATH = os.path.join(directory, "NERF.pth")

    torch.save(model.state_dict(), PATH)

    for images, voxels, label in train_loader:
        predicted_voxels = predict(model, images, device)
        print(predicted_voxels.shape)
        predicted_voxels=predicted_voxels[4]
        print(label[0])
        voxel_slice = predicted_voxels[:, :, 16]  # Slice at depth 16
        plt.imshow(voxel_slice, cmap='gray')
        plt.show()
        voxel_grid = predicted_voxels
        threshold = threshold_otsu(voxel_grid)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get the indices of occupied voxels (value > threshold, e.g., 0.5)
        x, y, z = np.where(voxel_grid > 0.45)
        # Plot occupied voxels
        ax.scatter(x, y, z, c='blue', marker='o', s=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("3D Object Visualization")
        plt.show()

import torch
import pandas as pd
from main import DDataset
import plotly.graph_objects as go
from VOXEL import R2N2
import torch.optim as optim
import torch.nn as nn
from skimage.filters import threshold_otsu

import numpy as np
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    directory = r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Training'
    DataFrame = pd.read_csv(r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Training\Annotations.csv')
    Data = DDataset(DataFrame, directory)
    train_loader = torch.utils.data.DataLoader(Data, batch_size=10, num_workers=2, shuffle=True)

    directory1 = r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Validation'
    DataFrame1 = pd.read_csv(r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Validation\Annotations.csv')
    Data1 = DDataset(DataFrame1, directory1)
    validation_loader = torch.utils.data.DataLoader(Data1, batch_size=5, num_workers=2, shuffle=True)


    def iou_loss(pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return 1 - intersection / union
    Epoch=5
    net = R2N2().to(device)
    optimizer=optim.Adam(net.parameters(),lr=0.001)
    criterion = nn.BCELoss()

    for _ in range(0,Epoch):
        Running_Loss=0.0
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs,Voxels,label=data
            batch_size = inputs.shape[0]
            inputs, Voxels,label = inputs.to(device), Voxels.to(device),label.to(device)
            inputs = inputs[:, 0, :, :, :]
            outputs=net(inputs)
            Loss=criterion(outputs,Voxels)
            Loss.backward()
            optimizer.step()
            Running_Loss+=Loss.item()
        print(_)
    for i, data in enumerate(train_loader):
        inputs, Voxels, label = data
        batch_size = inputs.shape[0]
        inputs, Voxels, label = inputs.to(device), Voxels.to(device), label.to(device)
        #inputs = inputs.view(batch_size, 5 * 3, 214, 214)
        inputs = inputs[:, 0, :, :, :]
        outputs = net(inputs)
        voxel_grid=outputs[1]
        voxel_grid_np = voxel_grid.squeeze().detach().cpu().numpy()
        threshold = threshold_otsu(voxel_grid_np)
        print(voxel_grid_np.shape)  # Ensure it's (50, 50, 50)
        print(voxel_grid_np.max(), voxel_grid_np.min())

        # Get the coordinates of the active voxels
        x, y, z = np.where(voxel_grid_np > 0.45)  # Threshold to create the voxel points
        print(f"Active Voxel Count: {len(x)}")
        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=z, colorscale='Viridis')
        )])

        # Customize the layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, 50]),
                yaxis=dict(range=[0, 50]),
                zaxis=dict(range=[0, 50]),
            ),
            title="3D Voxel Grid",
            width=800,
            height=800
        )

        fig.show()

        break
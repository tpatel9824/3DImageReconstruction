import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import os

from CNN_CLASSIFIER import  Net
if __name__ == '__main__':
    from main import DDataset
    import torch
    import pandas as pd

    directory = r'C:\Users\91875\Downloads\pix3dorg'
    DataFrame = pd.read_csv(r'C:\Users\91875\Downloads\pix3dorg\Annotations.csv')
    Data = DDataset(DataFrame, directory)
    train_loader = torch.utils.data.DataLoader(Data, batch_size=10, num_workers=2, shuffle=True)

    # Ensure the code below is only executed after DataLoader is initialized
    train_features, train_labels = next(iter(train_loader))
    print("Feature Batch Shape", train_labels)
    for batch in train_loader:
        images,label=batch
        print("Size Of A Batch Is",images.shape)
        break
    net=Net()
    net.Temp(net)


    for epoch in range(100):
        running_loss=0.0
        for i,data in enumerate(train_loader):
            inputs,label=data
            batch_size=inputs.shape[0]
            inputs=inputs.view(batch_size, 5 * 3, 214, 214)


            running_loss += net.Backward(net,inputs,label,)
            if i==14:
                print(epoch,running_loss)

    directory = r'C:\Users\91875\OneDrive\Desktop\3D_OBJECT\3DImageReconstruction\SAVED_MODELS'
    os.makedirs(directory, exist_ok=True)

    PATH = os.path.join(directory, "model.pth")

    torch.save(net.state_dict(), PATH)
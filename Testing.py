from main import DDataset
import torch
import pandas as pd
from nerf import VoxelReconstructionModel
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from CNN_CLASSIFIER import Net
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")
    directory1 = r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Validation'
    DataFrame1 = pd.read_csv(r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Validation\Annotations.csv')
    Data1 = DDataset(DataFrame1, directory1)
    testing_loader = torch.utils.data.DataLoader(Data1, batch_size=5, num_workers=2, shuffle=True)
    net = Net().to(device)
    PATH = r'C:\Users\91875\OneDrive\Desktop\3D_OBJECT\3DImageReconstruction\SAVED_MODELS\cnn_model.pth'
    net.load_state_dict(torch.load(PATH, weights_only=True))
    Actual=[]
    Predicted=[]
    with torch.no_grad():
        for data in testing_loader:
            inputs, voxels, label = data
            batch_size = inputs.shape[0]

            inputs, label = inputs.to(device), label.to(device)
            inputs = inputs.view(batch_size, 5 * 3, 214, 214)
            Prediction=net(inputs)
            _,Prediction=torch.max(Prediction,1)
            Temp = [Actual.append(i.item()) for i in label]
            Temp=[Predicted.append(i.item()) for i in Prediction]
    Accuracy=accuracy_score(Actual,Predicted)
    Precision=precision_score(Actual,Predicted,average='weighted')
    Recall=recall_score(Actual,Predicted,average='weighted')

    print(Accuracy)
    print(Precision)
    print(Recall)






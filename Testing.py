from main import DDataset
import torch
import pandas as pd
from nerf import VoxelReconstructionModel
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import pandas as pd
from CNN_CLASSIFIER import Net
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    directory1 = r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Training'
    DataFrame1 = pd.read_csv(r'C:\Users\91875\OneDrive\Desktop\3D_RECONSTRUCTION\Training\Annotations.csv')
    Data1 = DDataset(DataFrame1, directory1)

    testing_loader = DataLoader(Data1, batch_size=5, num_workers=0, shuffle=True)

    net = Net().to(device)
    PATH = r'C:\Users\91875\OneDrive\Desktop\3D_OBJECT\3DImageReconstruction\SAVED_MODELS\cnn_model.pth'
    net.load_state_dict(torch.load(PATH, weights_only=True))

    Actual = []
    Predicted = []

    with torch.no_grad():
        for data in testing_loader:
            inputs, voxels, label = data
            batch_size = inputs.size(0)

            inputs, label = inputs.to(device), label.to(device)
            inputs = inputs.view(batch_size, 5 * 3, 214, 214)

            Prediction = net(inputs)
            _, Prediction = torch.max(Prediction, 1)

            Actual.extend(label.cpu().numpy())
            Predicted.extend(Prediction.cpu().numpy())

    Accuracy = accuracy_score(Actual, Predicted)
    Precision = precision_score(Actual, Predicted, average='weighted', zero_division=0)
    Recall = recall_score(Actual, Predicted, average='weighted', zero_division=0)
    print("Testing Results:")
    print("Accuracy:", Accuracy)
    print("Precision:", Precision)
    print("Recall:", Recall)
    cm=confusion_matrix(Actual,Predicted)
    disp=ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()






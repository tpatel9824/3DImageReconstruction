from main import DDataset
import torch
import pandas as pd
from nerf import VoxelReconstructionModel
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")





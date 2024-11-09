import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
import pandas
from PIL import  Image
import sys
from torchvision import transforms as tf
print(torch.__version__)

class DDataset(Dataset) :
    def __init__(self,dataframe,root_dir,transforms=None):
        self.dataframe=dataframe
        self.root=root_dir
        self.dataframe=dataframe
        self.transforms=transforms
        self.default_transforms=None
    def __getitem__(self, idx):
        Img_Path=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,2],self.dataframe.iloc[idx,3])
        Obj_Path=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,4],self.dataframe.iloc[idx,5])

        Mesh=trimesh.load_mesh(Obj_Path)

        Img1=cv2.imread(Img_Path)
        Img=Image.fromarray(Img1)

        return Img,Mesh


directory=r'C:\Users\91875\Downloads\Pix3D_organized'
DataFrame=pd.read_csv(r'C:\Users\91875\Downloads\Pix3D_organized\Annotations.csv')
Data=DDataset(DataFrame,directory)
Data.__getitem__(5)
for _ in range(0,19):
    Temp,Mesh=Data.__getitem__(_)
    Mesh.show()
    Temp.show()
    Array=np.array(Temp)
    print(Array.shape)

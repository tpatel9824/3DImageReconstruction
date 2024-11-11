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
        self.default_transforms=tf.Compose([tf.Resize((214, 214)),tf.ToTensor()])
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        dict={'bed':0,'bookcase':1,'chair':2,'desk':3,'misc':4,'sofa':5,'table':6,'tool':7,'wardrobe':8}
        Label=torch.tensor(dict[self.dataframe.iloc[idx,0]])
        Img_Path=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,2],self.dataframe.iloc[idx,3])
        Img_Path1=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,2],self.dataframe.iloc[idx,6])
        Img_Path2=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,2],self.dataframe.iloc[idx,7])
        Img_Path3=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,2],self.dataframe.iloc[idx,8])
        Img_Path4=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,2],self.dataframe.iloc[idx,9])

        Obj_Path=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,4],self.dataframe.iloc[idx,5])
        Mesh=trimesh.load_mesh(Obj_Path)


        Img1=cv2.imread(Img_Path)
        Img1=Image.fromarray(Img1)
        Img1=self.default_transforms(Img1)

        Img2 = cv2.imread(Img_Path1)
        Img2 = Image.fromarray(Img2)
        Img2 = self.default_transforms(Img2)

        Img3 = cv2.imread(Img_Path2)
        Img3 = Image.fromarray(Img3)
        Img3 = self.default_transforms(Img3)

        Img4 = cv2.imread(Img_Path3)
        Img4 = Image.fromarray(Img4)
        Img4 = self.default_transforms(Img4)

        Img5 = cv2.imread(Img_Path4)
        Img5 = Image.fromarray(Img5)
        Img5 = self.default_transforms(Img5)

        return (torch.stack([Img1,Img2,Img3,Img4,Img5]),Label)


#directory=r'C:\Users\91875\Downloads\pix3dorg'
#DataFrame=pd.read_csv(r'C:\Users\91875\Downloads\pix3dorg\Annotations.csv')
#Data=DDataset(DataFrame,directory)
#a,b=Data.__getitem__(8)
#print(a.shape,b)
#for _ in range(0,158):
    #Temp,Mesh=Data.__getitem__(_)
    #Mesh.show()
    #Temp.show()
    #print(Temp.shape)
#train_loader=torch.utils.data.DataLoader(Data,batch_size=10,num_workers=2,shuffle=True)

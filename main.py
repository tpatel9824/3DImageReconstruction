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
import matplotlib.pyplot as plt
print(torch.__version__)

Mean=[0.485, 0.456, 0.406]
Std=[0.229, 0.224, 0.225]
class DDataset(Dataset) :
    def __init__(self,dataframe,root_dir,transforms=None):
        self.dataframe=dataframe
        self.root=root_dir
        self.dataframe=dataframe
        self.transforms=transforms
        self.default_transforms=tf.Compose([tf.Resize((214, 214)),tf.ToTensor()])
        self.transforms2=tf.Compose([tf.Normalize(Mean,Std), tf.RandomHorizontalFlip(p=0.5),          # Data augmentation
    tf.RandomRotation(degrees=15)])
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

        Mask_Path=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,10],self.dataframe.iloc[idx,11])
        Mask_Path1=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,10],self.dataframe.iloc[idx,12])
        Mask_Path2=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,10],self.dataframe.iloc[idx,13])
        Mask_Path3=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,10],self.dataframe.iloc[idx,14])
        Mask_Path4=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,10],self.dataframe.iloc[idx,15])



        Obj_Path=os.path.join(self.root,self.dataframe.iloc[idx,0],self.dataframe.iloc[idx,1],self.dataframe.iloc[idx,4],self.dataframe.iloc[idx,5])
        Mesh=trimesh.load_mesh(Obj_Path)


        Img1=cv2.imread(Img_Path)
        Mask=cv2.imread(Mask_Path,cv2.IMREAD_GRAYSCALE)
        _,binary_mask=cv2.threshold(Mask, 128, 255, cv2.THRESH_BINARY)
        Img1=cv2.bitwise_and(Img1,Img1,mask=binary_mask)
        Img1=Image.fromarray(Img1)
        Img1=self.default_transforms(Img1)
        Mean,Std=Img1.mean([1,2]), Img1.std([1,2])
        Img1=self.transforms2(Img1)


        Img2 = cv2.imread(Img_Path1)
        Mask = cv2.imread(Mask_Path1, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(Mask, 128, 255, cv2.THRESH_BINARY)
        Img2 = cv2.bitwise_and(Img2, Img2, mask=binary_mask)
        Img2 = Image.fromarray(Img2)
        Img2 = self.default_transforms(Img2)
        Mean,Std=Img2.mean([1,2]), Img2.std([1,2])
        Img2=self.transforms2(Img2)




        Img3 = cv2.imread(Img_Path2)
        Mask = cv2.imread(Mask_Path2, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(Mask, 128, 255, cv2.THRESH_BINARY)
        Img3 = cv2.bitwise_and(Img3, Img3, mask=binary_mask)
        Img3 = Image.fromarray(Img3)
        Img3 = self.default_transforms(Img3)
        Mean,Std=Img3.mean([1,2]), Img3.std([1,2])
        Img3=self.transforms2(Img3)




        Img4 = cv2.imread(Img_Path3)
        Mask = cv2.imread(Mask_Path3, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(Mask, 128, 255, cv2.THRESH_BINARY)
        Img4 = cv2.bitwise_and(Img4, Img4, mask=binary_mask)
        Img4 = Image.fromarray(Img4)
        Img4 = self.default_transforms(Img4)
        Mean,Std=Img4.mean([1,2]), Img4.std([1,2])
        Img4=self.transforms2(Img4)


        Img5 = cv2.imread(Img_Path4)
        Mask = cv2.imread(Mask_Path4, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(Mask, 128, 255, cv2.THRESH_BINARY)
        Img5 = cv2.bitwise_and(Img5, Img5, mask=binary_mask)
        Img5 = Image.fromarray(Img5)
        Img5 = self.default_transforms(Img5)
        Mean,Std=Img5.mean([1,2]), Img5.std([1,2])
        Img5=self.transforms2(Img5)



        return (torch.stack([Img1,Img2,Img3,Img4,Img5]),Label)


#directory=r'C:\Users\91875\Downloads\pix3dorg'
#DataFrame=pd.read_csv(r'C:\Users\91875\Downloads\pix3dorg\Annotations.csv')
#Data=DDataset(DataFrame,directory)
#Obj=Data.__getitem__(2)
#Img,Lab=Obj[0],Obj[1]
#for i in range(0,158):
    #Obj = Data.__getitem__(i)
    #Img,Lab=Obj[0],Obj[1]
    #for _ in range(0,5):
        #Images=tf.ToPILImage()(Img[_])
        #plt.imshow(Images)
        #plt.show()


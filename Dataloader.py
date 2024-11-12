import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir)) 
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
       
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L") 
        
        if self.transform:
            image = self.transform(image)
        
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.png', '.npy'))
        label = np.load(label_path) 
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label

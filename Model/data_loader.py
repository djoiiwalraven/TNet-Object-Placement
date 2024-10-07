import torch
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np

class MyDataLoader(Dataset):
    def __init__(self,root_dir,x_dir,y_dir,image_size=12):
        self.root_dir = root_dir
        self.proj_dir = os.getcwd()
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.transform = Compose([
            Resize((image_size,image_size)),
            ToTensor()
        ])

        os.chdir(self.root_dir)
        self.input_images = [os.path.join(x_dir,img) for img in sorted(os.listdir(x_dir)) ]
        self.output_images = [os.path.join(y_dir,img) for img in sorted(os.listdir(y_dir)) ]
        os.chdir(self.proj_dir)

    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self,idx):
        os.chdir(self.root_dir)
        x_path = self.input_images[idx]
        y_path = self.output_images[idx]

        in_img = Image.open(x_path).convert('L')
        out_img = Image.open(y_path).convert('L')

        if self.transform:
            in_img = self.transform(in_img)
            out_img = self.transform(out_img)

        os.chdir(self.proj_dir)

        return in_img,out_img

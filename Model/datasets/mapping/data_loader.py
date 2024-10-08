import torch
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
from torch.utils.data import Dataset
import os

class MappingDataLoader(Dataset):
    def __init__(self,root_dir,data_dir,image_size=32):
        self.root_dir = root_dir
        self.proj_dir = os.getcwd()
        self.data_dir = data_dir
        self.image_size = image_size

        self.transform = Compose([
            Resize((image_size,image_size*2)),
            ToTensor()
        ])

        os.chdir(self.root_dir)
        self.images = [os.path.join(data_dir,img) for img in sorted(os.listdir(data_dir)) ]
        os.chdir(self.proj_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load full image
        os.chdir(self.root_dir)
        image = Image.open(image_path).convert('RGB')
        os.chdir(self.proj_dir)
        
        if self.transform:
            image = self.transform(image)
        
        a,b = image[:,:,0:int(self.image_size)],image[:,:,int(self.image_size):]
        return a, b
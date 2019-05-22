import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import pdb
import numpy as np
from PIL import Image

class BatchGenerator():
    def __init__(self, data_dir, transform=False, split=False):
        self.data_dir = data_dir
        self.class_list = os.listdir(data_dir)
        x = []
        y = []
        for ci, cls in enumerate(self.class_list):

            ddir = os.listdir(os.path.join(data_dir, cls))
            ddir = [os.path.join(cls, d) for d in ddir]
            if split=='train':
                ddir = ddir[:500]
            elif split=='test':
                ddir = ddir[500:]
            x.append(ddir)
            y.append([ci]*len(ddir))
                    
        self.x = np.concatenate(x, axis=0)
        self.y = np.concatenate(y, axis=0)
        self.n = len(self.y)
        self.n_classes = len(self.class_list)
        
        if transform:
            self.trfn = transforms.Compose([
                transforms.RandomCrop(84, 84//8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.485,.455,.406],
                    std=[.229,.224,.225]) ])
        else:
            self.trfn = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.485,.455,.406],
                    std=[.229,.224,.225]) ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.x[idx])
        x = Image.open(data_path)
        x = self.trfn(x)
        return x, self.y[idx]
        

if __name__=='__main__':
    epgen = BatchGenerator('../data/miniImagenet/train', transform=True)
    epgen[0]

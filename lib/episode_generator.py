import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import pdb
import numpy as np
import csv
import pdb
from PIL import Image
import pickle


class CIFAR10():
    def __init__(self, data_dir, transform=False, phase='train'):
        assert phase in ['train', 'test']
        self.data_dir = data_dir

        if phase=='train':
            filenames = [os.path.join(self.data_dir, 
                'data_batch_{}'.format(i+1)) for i in range(5)]
        else:
            filenames = [os.path.join(self.data_dir,
                'test_batch')]
        
        def load_data(path):
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            imgs = np.array(data['data'])
            imgs = np.reshape(imgs, (imgs.shape[0], 3, 32, 32))
            imgs = np.transpose(imgs, (0,2,3,1))
            labels = np.array(data['labels'])
            return imgs, labels
        
        self.x = []
        self.y = []
        for p in filenames:
            x, y = load_data(p)
            self.x.append(x)
            self.y.append(y)

        self.x = np.concatenate(self.x, axis=0)
        # x should be PIL image to augmentation
        self.x = [Image.fromarray(x) for x in self.x]
        self.y = np.concatenate(self.y, axis=0)

        self.n = len(self.y)
        self.n_classes = 10
        self.hw = 32
    

        if transform:
            self.trfn = transforms.Compose([
                transforms.RandomCrop(self.hw, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.4914,.4822,.4465],
                    std=[.2023,.1994,.2010]) ])
        else:
            self.trfn = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.4914,.4822,.4465],
                    std=[.2023,.1994,.2010]) ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.x[idx]
        x = self.trfn(x)
        y = self.y[idx]
        return x, y
        

#class ImageNet():
#    def __init__(self, data_dir, transform=True, phase='train'):
#        assert phase in ['train', 'valid', 'test']
#        self.data_dir = data_dir
#        csv_file_name = os.path.join(data_dir,
#                'ImageSets/CLS-LOC'

class ImagenetTrain(torchvision.datasets.ImageFolder):
    def __init__(self, data_dir):

        jitter_param = 0.4
        self.hw = 224
        self.n_classes = 1000
    
        self.trfn = transforms.Compose([
            transforms.RandomResizedCrop(self.hw),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(jitter_param,
                jitter_param, jitter_param),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.455,.406],
                std=[.229,.224,.225]) ])
        
        data_dir = os.path.join(data_dir, 'train')
        super(ImagenetTrain, self).__init__(data_dir, transform=self.trfn)

class ImagenetVal():
    def __init__(self, data_dir):

        csv_file_name = os.path.join(data_dir, 'valid.csv')
        self.data_dir = os.path.join(data_dir, 'val')
        self.x, self.y = [], []
        with open(csv_file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i==0: #header
                    self.header = row
                else:
                    foldername = row[0].split('/')[0]
                    row[0] = row[0].replace(foldername + '/', '')
                    self.x.append(row[0])
                    self.y.append(int(row[1]))

        self.n = len(self.y)
        self.n_classes = 1000
        self.hw = 224

        self.trfn = transforms.Compose([
            transforms.Resize(self.hw),
            transforms.CenterCrop(self.hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.455,.406],
                std=[.229,.224,.225]) ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        filename = self.x[idx]
        data_path = os.path.join(self.data_dir, filename)
        x = Image.open(data_path)
        x = self.trfn(x)
        y = self.y[idx]
        return (x, y)




class Chexpert():
    def __init__(self, data_dir, transform=False, phase='train'):
        assert phase in ['train', 'valid']
        self.data_dir = data_dir
        
        csv_file_name = os.path.join(data_dir, phase + '.csv')
        #self.feature_start_index = 1 # 1-> includes gender, age, ...
        self.feature_start_index = 5
        self.uncertain_val = 0.2

        self.data = []
        with open(csv_file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0: # headers
                    self.header = row
                else:
                    # lets remove folder name: it already exists
                    foldername = row[0].split('/')[0]
                    row[0] = row[0].replace(foldername + '/', '')
                    self.data.append(row)

        self.n = len(self.data)
        self.n_classes = len(self.header[self.feature_start_index:])
        self.hw = 320
        
        self.x, self.y = [], []
        for d in self.data:
            self.x.append(d[0])
            y = d[self.feature_start_index:]
            y = [int(float(_y)) if _y!='' else self.uncertain_val for _y in y]
            self.y.append(y)
        self.y = np.array(self.y)


        if transform: 
            self.trfn = transforms.Compose([
                transforms.RandomCrop(self.hw, 0),
                transforms.ToTensor()])
        else:
            self.trfn = transforms.Compose([
                transforms.CenterCrop(self.hw),
                transforms.ToTensor()])

        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        filename = self.x[idx]
        data_path = os.path.join(self.data_dir, filename)
        x = Image.open(data_path)
        x = self.trfn(x)
        x = x.repeat(3,1,1) # to match with other image datasets which has 3 channels

        y = self.y[idx]

        return x, y

        

class MiniImagenet(): # only for miniImagenet
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
#    epgen = BatchGenerator('../data/miniImagenet/train', transform=True)
#    epgen[0]

    dataset = Chexpert('../data/CheXpert-v1.0-small', phase='train')
    pdb.set_trace()

#    CIFAR10('../data/cifar-10-batches-py', phase='test')

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:39:48 2023

@author: 33699
"""

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return (self.images[index], self.labels[index])
import pandas
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

def _load_file(f):
    return np.load(r'C:/Users/33699/git/dtu_mlops/data/corruptmnist/' + f)

def _concat_content(content):
    images = [file['images'] for file in content]
    images = torch.from_numpy(np.concatenate(images))
    labels = [file['labels'] for file in content]
    labels = torch.from_numpy(np.concatenate(labels))
    return images, labels

class MyDataset(Dataset):
  def __init__(self, filepaths):
    content = [_load_file(f) for f in filepaths]
    self.imgs, self.labels = _concat_content(content)
  
  def __len__(self):
    return self.imgs.shape[0]

  def __getitem__(self, idx):
    return (self.imgs[idx], self.labels[idx])

def mnist(batch_size = 64, eval_batch_size = 100):
    train_files = ['train_{}.npz'.format(i) for i in range(5)]
    test_file = ['test.npz']

    train_dl = DataLoader(MyDataset(train_files), batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(MyDataset(test_file), batch_size = eval_batch_size, shuffle = True) 
    
    return train_dl, test_dl

def visualize():
    
    train_dl, test_dl = mnist()
    
    f, axarr = plt.subplots(4, 16, figsize=(16, 4))
    
    images, _ = next(iter(train_dl))
    
    for i, ax in enumerate(axarr.flat):
        ax.imshow(images[i])
        ax.axis('off')
        
    plt.show()






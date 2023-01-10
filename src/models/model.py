from torch import nn
import torch
    
class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3), # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3), # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3), # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3), # [N, 8, 20]
            nn.LeakyReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected input to be a 3D tensor.')
        if x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError('Expected each sample to have shape [28, 28]')
        x = x.type(torch.FloatTensor)
        x = x.view(x.shape[0], 1, 28, 28)
        
        return self.classifier(self.backbone(x))     
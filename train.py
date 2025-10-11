import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

# --- VGG16 with custom top ---
class CustomVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.vgg16(weights=None)
        self.features = base.features  # All convolutional layers
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class MILoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        #self.lambs = torch.ones(13, requires_grad=True)
        self.lambs = torch.tensor(lam,requires_grad=True)

    def forward(self, outputs, features, labels):
        total_mi = 0
        for lam, layer in zip(self.lambs, features):
            mi = 0
            B, C, H, W = layer.shape
            feat_flat = layer.view(B, C, -1).mean(dim=2) 
            C2 = int(C / 2)
            
            #y = labels[:B]
            #y = y.repeat(C, 1)

            
            #y1 = labels[:B]
            #y1 = y1.repeat(C2, 1)

            #y2 = labels[B:]
            #y2 = y2.repeat(C2, 1)
            #print(y)
            #print(feat_flat.shape, y.shape)
            #mi += calcMI(feat_flat, y)[0]
            #mi += calcMI(feat_flat[:, :C2], y1)[0]
            #mi += calcMI(feat_flat[:, C2:], y2)[0]
            for c in range(C):
                x = feat_flat[:, c]
                y = labels[:B]
                mi += calcMI(x, y)[0]
                
            total_mi += lam * (mi / C)
        
        return total_mi
        

class ComboLoss(nn.Module):
    def __init__(self, lam, alpha=1):
        super().__init__()
        self.l1 = nn.CrossEntropyLoss()
        self.l2 = MILoss(lam)
        self.alpha = alpha


    def forward(self, outputs, features, labels):
        return self.l1(outputs, labels) + (self.alpha * self.l2(outputs, features, labels))





def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    bins = 32


    # --- Dataset & Loader ---
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    if os.path.exists('./data/imagenette'): 
        train_data = datasets.Imagenette(root='./data/imagenette', split='train', transform=transform, download=True)
        val_data = datasets.Imagenette(root='./data/imagenette', split='val', transform=transform, download=True)
    else:
        train_data = datasets.Imagenette(root='./data/imagenette', split='train', transform=transform, download=False)
        val_data = datasets.Imagenette(root='./data/imagenette', split='val', transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    model = CustomVGG16()
    model = model.cuda()

main()
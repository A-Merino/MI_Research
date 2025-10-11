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
from tqdm import tqdm
from mi_estimators import mi2d_gpu as calcMI
from hooks import ActivationCatcher as AC
from hooks import list_conv_layers as get_conv

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Config ---
data_dir = "../Ryan/datasets/imagenette2"
batch_size = 32
bins = 20

# --- Dataset & Loader ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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

from mi_estimators import mi2d_gpu as calcMI
from hooks import ActivationCatcher as AC
from hooks import list_conv_layers as get_conv

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


model = CustomVGG16()
model = nn.DataParallel(model)  # Multi-GPU if available
model = model.cuda()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, verbose=True)
checkpoint_dir = "models/plain"
# --- Training Loop ---
best_acc = 0
patience = 0
early_stop_patience = 25
num_epochs = 50


#lambdas = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]  # Combo, CEL + MI
#lambdas = [1.0,1.0,1.0,1.0,1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]  # ComboFirst, CEL + MI
#lambdas = [0.1,0.1,0.1,0.1,0.1,0.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0]  # ComboLast, CEL + MI
#lambdas = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]  # ComboSmall, CEL + MI
#lambdas = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]  # ComboMin, CEL - MI


lambdas = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]  #ComboPart, CEL + MI (row by row) 

criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False):
        xb, yb = xb.cuda(), yb.cuda()
        optimizer.zero_grad()
        
        #layers = get_conv(model)
        #acts = None
        #with AC(layers) as ac:
        out = model(xb)
            
         #   acts = ac.get_activations()
        loss = criterion(out, yb)
        #loss = criterion(out, acts, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(out, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        activations = []

    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.cuda(), yb.cuda()
            #layers = get_conv(model)
            #acts = None
            #with AC(layers) as ac:
            out = model(xb)
            
                #acts = ac.get_activations()
            #val_loss += criterion(out,acts, yb).item()
            val_loss += criterion(out, yb).item()
            _, preds = torch.max(out, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    val_acc = 100 * correct / total
    val_loss /= len(val_loader)

    # --- Scheduler and Early Stopping ---
    scheduler.step(val_acc)
    current_lr = scheduler.get_last_lr()[0]

    msg = f"📦 Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {current_lr}"
    
    print(msg)

    filename = f"vgg16_epoch_{epoch:03d}_valacc_{val_acc:.2f}.pt"
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
    print(f"💾 Saved checkpoint: {filename}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
    else:
        patience += 1
        if patience >= early_stop_patience:
            print("⏹️ Early stopping triggered.")
            break


from mi_estimators import mi2d_gpu as calcMI
from hooks import ActivationCatcher as AC
from hooks import list_conv_layers as get_conv

class MILoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        #self.lambs = torch.ones(13, requires_grad=True)
        self.lambs = torch.tensor(lam,requires_grad=True)

    def forward(self, outputs, features, labels):
        total_mi = 0
        for i, lam in enumerate(self.lambs):
            mi = 0
            
            B, C, H, W = features[i].shape
            B1, C1, H1, W1 = features[i+1].shape
            B2, C2, H2, W2 = features[-i].shape
            
            
            feat_flat = features[i].view(B, C, -1).mean(dim=2) 
            feat_flat1 = features[i+1].view(B1, C1, -1).mean(dim=2)
            feat_flat2 = features[-i].view(B2, C2, -1).mean(dim=2)
            feat_flatx = feat_flat[:feat_flat1.shape[0],:feat_flat1.shape[1]]
            feat_flat1 = feat_flat1[:feat_flat.shape[0],:feat_flat.shape[1]]
            mi += calcMI(feat_flatx, feat_flat1)[0]
            feat_flat = feat_flat[:feat_flat2.shape[0],:feat_flat2.shape[1]]
            feat_flat2 = feat_flat2[:feat_flat.shape[0],:feat_flat.shape[1]]
            mi += calcMI(feat_flat, feat_flat2)[0]
            #mi = calcMI(feat_flat, feat_flat1)
            #for c in range(C):
             #   x = feat_flat[:, c]
              #  y = labels[:B]
               # mi += calcMI(x, y)[0]
                
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


def trainModel(lambdas, loss, checkpoint_dir):
    model = CustomVGG16()
    model = nn.DataParallel(model)  # Multi-GPU if available
    model = model.cuda()
    model.load_state_dict(torch.load('./models/plain/vgg16_006_acc_66.59.pt', weights_only=True))
    
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, verbose=True)
    # --- Training Loop ---
    best_acc = 0
    patience = 0
    early_stop_patience = 25
    num_epochs = 25

    # combo
    if loss == 'combo':
        criterion = ComboLoss(lambdas)
    # comp
    elif loss == 'comp':
        criterion = ComboLoss(lambdas, alpha=0.5)
    
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
    
        for xb, yb in tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False):
            xb, yb = xb.cuda(), yb.cuda()
            optimizer.zero_grad()
            
            layers = get_conv(model)
            acts = None
            with AC(layers) as ac:
                out = model(xb)
                
                acts = ac.get_activations()
            
            loss = criterion(out, acts, yb)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            activations = []
    
        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)
    
        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.cuda(), yb.cuda()
                layers = get_conv(model)
                acts = None
                with AC(layers) as ac:
                    out = model(xb)
                
                    acts = ac.get_activations()
         
                val_loss += criterion(out,acts, yb).item()
                _, preds = torch.max(out, 1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
    
        val_acc = 100 * correct / total
        val_loss /= len(val_loader)
    
        # --- Scheduler and Early Stopping ---
        scheduler.step(val_acc)
        current_lr = scheduler.get_last_lr()[0]
    
        msg = f"📦 Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {current_lr}"
        
        print(msg)
    
        filename = f"vgg16_epoch_{epoch:03d}_valacc_{val_acc:.2f}.pt"
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
        print(f"💾 Saved checkpoint: {filename}")
    
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("⏹️ Early stopping triggered.")
                break
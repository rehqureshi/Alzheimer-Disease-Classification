# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:37:25 2021

@author: OKOK PROJECTS
"""

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import gc
import os
import time
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_pytorch import EfficientNet
import cv2
from PIL import Image, ImageFilter
from albumentations.pytorch import ToTensorV2

import albumentations as albu

from albumentations.pytorch import ToTensorV2
from pytorch_ranger import Ranger
# At least fixing some random seeds. 
# It is still impossible to make results 100% reproducible when using GPU
warnings.simplefilter('ignore')
torch.manual_seed(47)
np.random.seed(47)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv('train.csv')
class Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True, transforms = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            data (np.ndarray): resized images data in a shape of (HxWxC)
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            
        """
        self.df = df
        
        self.transforms = transforms
        self.train = train
        
    def __getitem__(self, index):
        name = self.df.iloc[index,1]
        
        if self.train:
          if name[0:4]=='mild':
            path = '/Users/rehanqureshi/Desktop/Major_Project/code/dataset/train/MildDemented/' + str(name)
          elif name[0:4]=='very':
            path='/Users/rehanqureshi/Desktop/Major_Project/code/dataset/train/VeryMildDemented/' + str(name)
          elif name[0:4]=='nonD':
            path='/Users/rehanqureshi/Desktop/Major_Project/code/dataset/train/NonDemented/' + str(name)   
          else:
            path= '/Users/rehanqureshi/Desktop/Major_Project/code/dataset/train/ModerateDemented/' + str(name)  
        else:
          path = 'test/' + str(name)
          
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)

        if self.transforms:
            image = self.transforms(image=np.array(image))
            image=image['image']
            
        if self.train:
            y = self.df.loc[index]['label']
            return image, y
        else:
            return image
    
    def __len__(self):
        return len(self.df)
    
train_transform = albu.Compose([
               albu.HorizontalFlip(),
               albu.VerticalFlip(),
               
               albu.Rotate(limit=30),
               albu.Resize(256,256),
               albu.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
test_transform = albu.Compose([
                albu.Resize(256,256),
                albu.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]) 
# Going to use efficientnet-b0 NN architecture
skf = StratifiedKFold(n_splits=5, random_state=47, shuffle=True)
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffling the train data
train_df

# Number of epochs to run
model_path = 'model.pth'  # Path and filename to save model to
es_patience = 5  # Early Stopping patience - for how many epochs with no improvements to wait

oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions

# We stratify by target value, thus, according to sklearn StratifiedKFold documentation
# We can fill `X` with zeroes of corresponding length to use it as a placeholder
# since we only need `y` to stratify the data
def train_model(model,criterion,optimizer,scheduler,epochs=25):
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['label']), 1):
    
        if fold==1:
            print('=' * 20, 'Fold', fold, '=' * 20)
            best_val = None  # Best validation score within this fold
            patience = es_patience  # Current patience counter
        
    
            train = Dataset(df=train_df.iloc[train_idx].reset_index(drop=True), train=True, transforms=train_transform)
            val = Dataset(df=train_df.iloc[val_idx].reset_index(drop=True),  train=True, transforms=test_transform)  
    
            train_loader = DataLoader(dataset=train, batch_size=128, shuffle=True)   
            val_loader = DataLoader(dataset=val, batch_size=64, shuffle=False)
       #     test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False)
            val_roc=0
            for epoch in range(epochs):
                start_time = time.time()
                correct = 0
                epoch_loss = 0
                scheduler.step(val_roc)
                model.train()
        
                for x, y in train_loader:
                    x = torch.tensor(x, device=device, dtype=torch.float32)
                    y = torch.tensor(y, device=device, dtype=torch.float32)
            
                    optimizer.zero_grad()
            
                    z = model(x)
                    loss = criterion(z, y.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
                    correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
                    epoch_loss += loss.item()
                train_acc = correct / len(train_idx)

                model.eval()  # switch model to the evaluation mode
                val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
                with torch.no_grad():  # Do not calculate gradient since we are only predicting
                    # Predicting on validation set
                    for j, (x_val, y_val) in enumerate(val_loader):
                        x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                        y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                        z_val = model(x_val)
                        val_pred = torch.sigmoid(z_val)
                        val_preds[j*x_val.shape[0]:j*x_val.shape[0] + x_val.shape[0]] = val_pred
                    val_acc = accuracy_score(train_df.iloc[val_idx]['label'].values, torch.round(val_preds.cpu()))
                    val_roc = roc_auc_score(train_df.iloc[val_idx]['label'].values, val_preds.cpu())
            
                    print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
                    epoch + 1, 
                    epoch_loss, 
                    train_acc, 
                    val_acc, 
                    val_roc, 
                    str(datetime.timedelta(seconds=time.time() - start_time))))
            
                    # During the first iteration (first epoch) best validation is set to None
                    if not best_val:
                        best_val = val_roc  # So any validation roc_auc we have is the best one for now
                        torch.save(model, model_path)  # Saving the model
                        continue
                
                    if val_roc >= best_val:
                        best_val = val_roc
                        patience = es_patience  # Resetting patience since we have new best validation accuracy
                        torch.save(model, model_path)  # Saving current best model
                        torch.save(model.state_dict(),'./checkpoint/current_checkpoint.pt')
                    else:
                        patience -= 1
                        if patience == 0:
                            print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                            break
        else :
            break
model = EfficientNet.from_pretrained('efficientnet-b0')
for param in model.parameters():
    param.requires_grad = False

in_features = model._fc.in_features
classifier =nn.Linear(in_features, 1)
model._fc=classifier

if torch.cuda.is_available():
    model = model.cuda()
     
optimizer = Ranger(model.parameters(), lr=0.001,weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=4, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
weight = torch.tensor([32542])/1168
weight=weight.cuda()
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
model=train_model(model,criterion,optimizer,scheduler,epochs=4)
model = EfficientNet.from_pretrained('efficientnet-b0')
for param in model.parameters():
    param.requires_grad = True

in_features = model._fc.in_features
classifier =nn.Linear(in_features, 1)
model._fc=classifier

if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('./checkpoint/current_checkpoint.pt'),strict=True)    
optimizer = Ranger(model.parameters(), lr=0.001,weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=4, verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
weight = torch.tensor([32542])/1168
weight=weight.cuda()
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
model=train_model(model,criterion,optimizer,scheduler,epochs=10)
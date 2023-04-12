# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:25:18 2022

@author: user
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from customdataset import CustomDataset
import numpy as np
import random
import os
from datasets import load_dataset

from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dataloader(batch_size, num_workers):
    
    dataset = load_dataset('glue','rte')
    
    train = dataset['train']
    train = pd.DataFrame(train)
    train_data = train[['sentence1','sentence2','label']]
    
    val = dataset['validation']
    val = pd.DataFrame(val)
    val_data = val[['sentence1','sentence2','label']]
    
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)#, collate_fn=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)#, collate_fn=None)
    
    return train_loader, val_loader
    
class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
def GradualLabelSmoothing(smoothing, epoch, for_num):
    
    loss = LabelSmoothingLoss(smoothing - smoothing*(for_num/(epoch-1)))
    
    return loss

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
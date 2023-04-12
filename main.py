# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:29:20 2022

@author: user
"""

import torch
import torch.nn as nn


from transformers import (AdamW,
                          BertForSequenceClassification,
                          )
from transformers.optimization import get_linear_schedule_with_warmup

import numpy as np
import pandas as pd

from tqdm import tqdm
from custom_utils import *

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pretrain = "bert-base-cased"

EPOCHS = 10
batch_size = 32
warmup_ratio = 0.1
max_grad_norm = 1
max_length = 512
num_workers = 0
smoothing = 0.4

train_loader, val_loader = dataloader(batch_size, num_workers)

total_steps = len(train_loader) * EPOCHS
warmup_step = int(total_steps * warmup_ratio)

model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=2,
                                                      ).to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_step,
                                            num_training_steps=total_steps)

# loss_ls = LabelSmoothingLoss(smoothing) # smoothing = 0.x -> normal label smoothing

PATH = './model/'

for i in range(EPOCHS):
    total_loss = 0.0
    correct_train = 0
    correct_eval = 0
    total_train = 0
    total_eval = 0
    batches = 0
    
    loss_ls = GradualLabelSmoothing(smoothing, EPOCHS, i)

    model.train()
    for batch in tqdm(train_loader):
        batch = tuple(v.to(device) for v in batch)
        input_ids, token_type_ids, attention_masks, labels = batch
        
        out = model(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    labels=labels)

        logits = out[1]
        
        loss = loss_ls(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
          
        total_loss += loss.item()
        
        _, predicted = torch.max(logits, 1)
        correct_train += (predicted == labels).sum()
        total_train += len(labels)

    print("")
    print("epoch {} Train Loss {:.6f} train acc {:.6f}".format(i+1,
                                                       torch.true_divide(total_loss, total_train),
                                                       torch.true_divide(correct_train, total_train)))
    print("")

    torch.save(model.state_dict(), PATH + 'Epoch_{}_loss_{:.4f}.pt'.format(i+1,
                                                                           torch.true_divide(total_loss,
                                                                                             total_train)))
    
    model.eval()
    for batch in tqdm(val_loader):
        batch = tuple(v.to(device) for v in batch)
        input_ids, token_type_ids, attention_masks, labels = batch
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        labels=labels
                        )
            logits = out[1]
            _, predicted = torch.max(logits, 1)
            correct_eval += (predicted == labels).sum()
            total_eval += len(labels)
        
    print("epoch {} Test acc {:.6f}".format(i+1, torch.true_divide(correct_eval, total_eval)))
    print("")
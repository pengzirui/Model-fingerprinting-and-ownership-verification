#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import torch.nn as nn
from torch import optim
import copy
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class Contrastiveloss(nn.Module):
    def __init__(self, tau = 0.1):
        self.tau = tau
        super().__init__()
    
    def similarity(self, x1 , x2):
        return (x1.T @ x2) / (torch.norm(x1,2) * torch.norm(x2,2))
    
    def single_image_loss(self, x1, y1, x, y):
        same = torch.where(y1 == y)[0]
        diff = torch.where(y1 != y)[0]
        top = 0.0
        down = 0.0
        for i in same:
            top += torch.exp(self.similarity(x1, x[i]) / self.tau)
        for i in range(x.shape[0]):
            down += torch.exp(self.similarity(x1, x[i]) / self.tau)
        return -torch.log(top/down) 
    
    def forward(self, x, y):
        loss = 0
        for i in range(x.shape[0]):
            loss += self.single_image_loss(x[i],y[i],x,y)
            
        return 1/(x.shape[0])*loss

    
def train(net,trainloader,testloader,n_epochs,optimizer, fp_v, tau=0.2, device = "cuda:0"):
        accuracy = 0
        net.train()
        for epoch in range(n_epochs):
                running_loss = 0.0
                for i, (inputs, labels) in enumerate(trainloader, 0):
                        # Transfer to GPU
                        inputs, labels = inputs.to(device), labels.to(device)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = net(inputs)
                        contras = Contrastiveloss(tau)
                        
                        loss = contras(net.feature(inputs),labels)            
                        print("loss:",loss)
                        loss.backward()
                        optimizer.step()
                evaluate(net, fp_v, testloader, device)
        print('Finished Training')
        return net

    
def similarity(x1 , x2):
    return (x1 @ x2.T) / (torch.norm(x1,2) * torch.norm(x2,2))

def evaluate(net, fp_v, test_loader, device = "cuda:0"):
    """
    param:
    net: f
    data_loader: data points used as queries
    pert: UAP
    label: y
    b_z: batch_size of return
    nb_class: the number of classification class

    return:
    (xi,yi) as an dataloader
    """
    net = net.to(device)
    sim_indep = 0
    sim_homo = 0
    cnt_indep = 0
    cnt_homo = 0
    
    for inputs, labels in test_loader:
        idx_indep = (labels == 2)
        idx_homo = (labels == 1)
        fp_indep = encoder.feature(inputs[idx_indep,:])
        for cnt in range(fp_indep.shape[0]):
            sim_indep += similarity(fp_v, fp_indep[cnt])
            cnt_indep += fp_indep.shape[0]
        fp_homo = encoder.feature(inputs[idx_homo,:])
        for cnt in range(fp_homo.shape[0]):
            sim_homo += similarity(fp_v, fp_homo[cnt])
            cnt_homo += fp_homo.shape[0]
            
    print("similarity of homologous models:",sim_homo/cnt_homo)
    print("similarity of independent models:",sim_indep/cnt_indep)
    


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
from model_architecture import get_model
from encoder import classifier
from encoder_training import train, evaluate
from utils import *
from encoder_trainingdata_preparation import concanate_fingerprint

def framework_building(fv, v, fp_gene_points, indep_models_path, indep_models_archi, homo_models_path, 
                       homo_models_archi, train_val_split_rate = 0.8, optimizer = optim.Adam(c.parameters(), lr=5e-4),
                       batch_size = 512, device = torch.device('cuda:0')):
    """
    @Param:
        fv: victim model
        v:UAP vector
        fp_gene_points: result of fingerprint_point_selection
        indep_models_path: list of indep models used for training encoder
        indep_models_archi: architecture of indep models
        homo_models_path: list of homologous models used for training encoder
        homo_models_archi:architecture of homologous models
        train_val_split_rate: train and validation dataset splitting rate
        batch_size: bz of training encoder
        device: CPU or GPU
        optimizer: optimizer used for training encoder
    @Return:
        return a trained encoder
    
    """
    #generate fingerprints
    data_set = concanate_fingerprint(fv, fp_gene_points, v, 0).dataset
    fp_v = data_set[0]
    
    for cnt in range(len(indep_models_path)):
        f_indep = get_model(indep_models_archi[cnt])
        f_indep.load_state_dict(torch.load(indep_models_path[cnt]))
        data_set_cur = concanate_fingerprint(f_indep, fp_gene_points,v, 2).dataset
        data_set = torch.utils.data.ConcatDataset([data_set,data_set_cur])
        
    for cnt in range(len(homo_models_path)):
        f_homo = get_model(homo_models_archi[cnt])
        f_homo.load_state_dict(torch.load(homo_models_path[cnt]))
        data_set_cur = concanate_fingerprint(f_homo, fp_gene_points,v, 1).dataset
        data_set = torch.utils.data.ConcatDataset([data_set,data_set_cur])
   
    print("The length of encoder's training data:",len(data_set))
    train_size = int(train_val_split_rate * len(data_set))
    test_size = len(data_set) - train_size 
    train_dataset, val_dataset = torch.utils.data.random_split(data_set, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #train and evaluate encoder
    c = classifier().to(device)
    n_epochs = 10
    train(c,train_loader,val_loader,n_epochs,optimizer,fp_v, 0.2, device)
    return c

def framework_testing(fv, v, fp_gene_points, indep_models_path, indep_models_archi, homo_models_path, homo_models_archi, 
                      net, batch_size, device = torch.device('cuda:0')):
     """
    @Param:
        fv: victim model
        v:UAP vector
        fp_gene_points: result of fingerprint_point_selection
        indep_models_path: list of indep models used for testing encoder
        indep_models_archi: architecture of indep models
        homo_models_path: list of homologous models used for testing encoder
        homo_models_archi:architecture of homologous models
        net: trained encoder
        batch_size: bz of testing encoder
        device: CPU or GPU
    @Return:
        None
    """
    
    data_set = concanate_fingerprint(fv, fp_gene_points, v, 0).dataset
    fp_v = data_set[0]
    for cnt in range(len(indep_models_path)):
        f_indep = get_model(indep_models_archi[cnt])
        f_indep.load_state_dict(torch.load(indep_models_path[cnt]))
        data_set_cur = concanate_fingerprint(f_indep, fp_gene_points,v, 2).dataset
        data_set = torch.utils.data.ConcatDataset([data_set,data_set_cur])
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
        evaluate(net, fp_v, test_loader, device)
    for cnt in range(len(homo_models_path)):
        f_homo = get_model(homo_models_archi[cnt])
        f_homo.load_state_dict(torch.load(homo_models_path[cnt]))
        data_set_cur = concanate_fingerprint(f_homo, fp_gene_points,v, 1).dataset
        data_set = torch.utils.data.ConcatDataset([data_set,data_set_cur])
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
        evaluate(net, fp_v, test_loader, device)


# In[ ]:





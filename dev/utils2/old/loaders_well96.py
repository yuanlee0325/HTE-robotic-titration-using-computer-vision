#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import shutil
from tqdm import tqdm
import random
from numpy import genfromtxt
from torch.utils.data import SubsetRandomSampler


# In[28]:


class WellData(Dataset):
    def __init__(self,
                 hrpath=r'D:\All_files\Blender\Test2\mask_12345w',
                 lrpath=r'D:\All_files\Blender\Test2\image_12345w',
                 transform_hr=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]),
                 transform_lr=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]),
                 #mode='train',
                 #shuffle=True,
                 verbose=False
                ):
        '''
        load microscopy data & transform to lr
        # path : directory
        # id_file: id file 
        # transform_hr: list of transforms for HR/original images
        # transform_lr: list of transforms for low res images
        
        '''
        super(WellData,self).__init__()
        self.hrpath=hrpath
        self.lrpath=lrpath
        
        os.chdir(lrpath)
        ids=glob.glob('*') #s2/s1 files
        self.ids=ids
    
        self.transform_hr=transform_hr
        self.transform_lr=transform_lr
        
        self.verbose=verbose
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        basename=self.ids[idx]
        if self.verbose : print(basename)
        (image_lr,image_hr)=self.get_data(basename)
        
        #print(image.shape)
        #plt.imshow(image)
        #print(type(image))
        
        if self.transform_hr: 
            image_hr=self.transform_hr(image_hr)/255
        if self.transform_lr: 
            image_lr=self.transform_lr(image_lr)/255

        image_hr[image_hr>0]=1

        return (image_lr,image_hr,basename)
    
    def get_data(self, basename=None):
    
        x=Image.open(os.path.join(self.lrpath,basename)).convert('RGB')
        #x/=255
        basename_hr=self._get_y_label(basename)
        y=Image.open(os.path.join(self.hrpath,basename_hr)).convert('L')
        #y/=255

        if self.verbose:
            #print(basename)
            print(basename_hr)
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.imshow(x);plt.axis('off');plt.title('lr: '+basename)
            plt.subplot(1,2,2)
            plt.imshow(y);plt.axis('off');plt.title('hr: '+self._get_y_label(basename))
            plt.show()

        return (x,y)
        
    #@staticmethod
    def _get_y_label(self,x_file):
        return x_file.replace('orig','mask')


# Creating data indices for training and validation splits:
# get data

def train_test_split(dataset,
                     validation_split=0.2,
                     shuffle_dataset=True,
                     batch_size=10,
                     seed=112
                    ):

    
    dataset_size=len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    
    return (train_loader,validation_loader)
    
def train_test_split_infer(dataset,
                     validation_split=0.3,
                     shuffle_dataset=True,
                     batch_size=10
                    ):

    
    dataset_size=len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(112)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    
    np.random.seed()
    d1=np.random.choice(train_indices,batch_size)
    d2=np.random.choice(val_indices,batch_size)
    
    train_sampler = torch.utils.data.Subset(dataset, d1)
    valid_sampler = torch.utils.data.Subset(dataset, d2)

    train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=False)
    
    return (d1,d2,train_loader,validation_loader)


def get_fs(lrpath=r'C:\Users\bdutta\work\Matlab\water_Matlab\data\scat',
           indices=None):

    os.chdir(lrpath)
    ids=glob.glob('*')
    
    lst_f1=[]
    lst_f2=[]
    
    ids=np.array(ids)[indices]
        
    for i in range(len(ids)):
    
        idx=ids[i].split('.')[0].split('_')[-1]
        lst_f1.append('sig_f1_'+idx+'.csv')
        lst_f2.append('sig_f2_'+idx+'.csv')
        
    return(ids,lst_f1,lst_f2)
    


    
class WellDataC2(Dataset):
    def __init__(self,
                 hrpath=r'D:\All_files\Blender\Test2\mask_12345w',
                 lrpath=r'D:\All_files\Blender\Test2\image_12345w',
                 transform_hr=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]),
                 transform_lr=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]),
                 num_classes=2,
                 verbose=False
                ):
        '''
        load microscopy data & transform to lr
        # path : directory
        # id_file: id file 
        # transform_hr: list of transforms for HR/original images
        # transform_lr: list of transforms for low res images
        
        '''
        super(WellDataC2,self).__init__()
        self.hrpath=hrpath
        self.lrpath=lrpath
        
        self.num_classes=num_classes
        
        os.chdir(lrpath)
        ids=glob.glob('*') #s2/s1 files
        self.ids=ids
    
        self.transform_hr=transform_hr
        self.transform_lr=transform_lr
        
        self.verbose=verbose
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        basename=self.ids[idx]
        if self.verbose : print(basename)
        (image_lr,image_hr)=self.get_data(basename)
        
        #print(image.shape)
        #plt.imshow(image)
        
        #print(type(image))
        
        if self.transform_hr: 
            image_hr=self.transform_hr(image_hr)/255
        if self.transform_lr: 
            image_lr=self.transform_lr(image_lr)/255

        image_hr[image_hr>0]=1
        
        mask=torch.zeros((self.num_classes,*image_hr.shape[1:]))
        #print(mask.shape)
        for idx in range(self.num_classes):
            mask[idx]=(image_hr==idx)
        #mask[1]=(image_hr==1)

        return (image_lr,mask,basename)
    
    def get_data(self, basename=None):
    
        x=Image.open(os.path.join(self.lrpath,basename)).convert('RGB')
        #x/=255
        basename_hr=self._get_y_label(basename)
        y=Image.open(os.path.join(self.hrpath,basename_hr)).convert('L')
        #y/=255

        if self.verbose:
            #print(basename)
            print(basename_hr)
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.imshow(x);plt.axis('off');plt.title('lr: '+basename)
            plt.subplot(1,2,2)
            plt.imshow(y);plt.axis('off');plt.title('hr: '+self._get_y_label(basename))
            plt.show()

        return (x,y)
        
    #@staticmethod
    def _get_y_label(self,x_file):
        return x_file.replace('orig','mask')
    
    
    
    






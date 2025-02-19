import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.transforms as T
import os
from PIL import Image
from tqdm.auto import tqdm
from glob import glob


class Well96Dataset(Dataset):
    def __init__(self, 
                 root=r'D:\All_files\Blender\Test2', 
                 #root=r'C:\Users\bdutta\work\pys\AI_algos\Liv\Camera_Test\color_palette_images\subsets',
                 train=True,
                 transforms_image=T.Compose([T.PILToTensor(),T.Resize((256,256),T.InterpolationMode.NEAREST)]),
                 transforms_mask=T.Compose([T.PILToTensor(),T.Resize((256,256),T.InterpolationMode.NEAREST)]),
                 resize=128,
                 verbose=True): # transforms
        
        self.root = root
        self.transforms_image=transforms_image
        self.transforms_mask=transforms_mask
        self.verbose=verbose
        self.resize=resize
        self.err=0
  
        if train:
            os.chdir(os.path.join(root, "img_aug"))
            self.imgs = glob('*.png') # list of images in train folder
            

    def __getitem__(self, idx):
        # print(idx)
        img_path = os.path.join(self.root, "img_aug", self.imgs[idx])
        
        self._get_mask_name(self.imgs[idx])
        mask_path = os.path.join(self.root, "mask2_label", self.masks)
        
        if self.verbose:
            print(self.imgs[idx])
            print(self.masks)

        img = Image.open(img_path).convert("RGB")
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        
        if self.transforms_image : img = self.transforms_image(img)
        if self.transforms_mask : mask = self.transforms_mask(mask)
            
        # convert the PIL Image into a numpy array
        mask = np.array(mask.permute(1,2,0))
        #print(mask.shape)
        mask=mask[:,:,0]
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        #print(obj_ids.shape)
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        #print(masks.shape)
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        
        # print(num_objs)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # Check if area is larger than a threshold
            A = abs((xmax-xmin) * (ymax-ymin)) 
            #print([xmin, ymin, xmax, ymax])
            boxes.append([xmin, ymin, xmax, ymax])

        # print('nr boxes is equal to nr ids:', len(boxes)==len(obj_ids))
        num_objs = len(obj_ids)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] =masks
        target["labels"] = labels # Not sure if this is needed
        
        

        return img.double()/255, target

    def __len__(self):
        return len(self.imgs)
    
    def _get_mask_name(self,fname):
        dummy=fname.split('.')[0].split('_')[:-1]
        self.masks='_'.join(dummy)+'_mask.png'
    
    '''
    def _get_mask_name(self,fname):
        fname=fname.split('.')[0]
        dummy=fname.split('_')
        dummy[dummy.index('orig')]='mask'

        fname='_'.join(dummy)
        self.masks=fname+'.png'
        print(self.masks)
    '''
    
def train_test_split(dataset,
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

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler,collate_fn=lambda x:list(zip(*x)))
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler,collate_fn=lambda x:list(zip(*x)))
    
    return (train_loader,validation_loader)
    
    
    
class WellDset_t2(Dataset):
    def __init__(self, 
                 root=r'D:\All_files\Blender\Test2', 
                 #root=r'C:\Users\bdutta\work\pys\AI_algos\Liv\Camera_Test\color_palette_images\subsets',
                 train=True,
                 transforms_image=T.Compose([T.PILToTensor(),T.Resize((256,256),T.InterpolationMode.NEAREST)]),
                 transforms_mask=T.Compose([T.PILToTensor(),T.Resize((256,256),T.InterpolationMode.NEAREST)]),
                 resize=128,
                 verbose=True): # transforms
        
        self.root = root
        self.transforms_image=transforms_image
        self.transforms_mask=transforms_mask
        self.verbose=verbose
        self.resize=resize
        self.err=0
  
        if train:
            os.chdir(os.path.join(root, "image2"))
            self.imgs = glob('*.png') # list of images in train folder
            

    def __getitem__(self, idx):
        # print(idx)
        img_path = os.path.join(self.root, "image2", self.imgs[idx])
        
        self._get_mask_name(self.imgs[idx])
        mask_path = os.path.join(self.root, "mask2_label", self.masks)
        
        if self.verbose:
            print(self.imgs[idx])
            print(self.masks)

        img = Image.open(img_path).convert("RGB")
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        
        if self.transforms_image : img = self.transforms_image(img)
        if self.transforms_mask : mask = self.transforms_mask(mask)
            
        # convert the PIL Image into a numpy array
        mask = np.array(mask.permute(1,2,0))
        #print(mask.shape)
        mask=mask[:,:,0]
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        #print(obj_ids.shape)
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        #print(masks.shape)
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        
        # print(num_objs)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # Check if area is larger than a threshold
            A = abs((xmax-xmin) * (ymax-ymin)) 
            #print([xmin, ymin, xmax, ymax])
            boxes.append([xmin, ymin, xmax, ymax])

        # print('nr boxes is equal to nr ids:', len(boxes)==len(obj_ids))
        num_objs = len(obj_ids)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] =masks
        target["labels"] = labels # Not sure if this is needed
        
        

        return img.double()/255, target

    def __len__(self):
        return len(self.imgs)
    
    def _get_mask_name(self,fname):
        dummy=fname.split('.')[0].split('_')[:-1]
        self.masks='_'.join(dummy)+'_mask.png'

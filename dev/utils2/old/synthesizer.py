'''
Strategy : 
1. Run perspective & affine first to generate 25 images/ original --> create also mask
2. Run bsc,sepia & jitter blur for each of the above mentioned images --> no mask generation

'''
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import os

from fimage import FImage
from fimage.presets import Preset
from fimage.filters import Contrast, Brightness, Saturation, Sepia


def perspective(orig_img,mask_img,num_transform=10,path=None,verbose=True):
    create_dir(path)
    # orig image is a PIL object
    basename=get_basename(orig_img.filename)
    seed=np.random.randint(1100000)
    
    torch.manual_seed(seed)
    perspective_transformer = T.RandomPerspective(distortion_scale=0.4, p=1.0)

    perspective_imgs = [perspective_transformer(orig_img) for _ in range(num_transform)]
    plot(perspective_imgs,with_orig=False)

    torch.manual_seed(seed)
    perspective_imgs2 = [perspective_transformer(mask_img) for _ in range(num_transform)]
    
    if verbose:
        plt.figure(figsize=(18,6))
        plot(perspective_imgs2,with_orig=False)
        
    image_dir=os.path.join(path,'image') if path else os.getcwd()
    mask_dir= os.path.join(path,'mask') if path else os.getcwd()
    
    for idx,img in enumerate(perspective_imgs):
        img.save(os.path.join(image_dir,basename+'_orig_persp_'+str(idx)+'.png'))
    
    for idx,img in enumerate(perspective_imgs2):
        img.save(os.path.join(mask_dir,basename+'_mask_persp_'+str(idx)+'.png'))


def affine(orig_img,mask_img,num_transform=10,path=None,verbose=True):   
    create_dir(path)
    basename=get_basename(orig_img.filename)
    
    #if path: basename=os.path.join(path,basename)
    # rotation
    seed=np.random.randint(1100000)
    torch.manual_seed(seed)
    
    affine_transfomer = T.RandomAffine(degrees=(5, 30), translate=(0.1, 0.3), scale=(0.25, 0.75))

    affine_imgs = [affine_transfomer(orig_img) for _ in range(num_transform)]

    torch.manual_seed(seed)
    affine_imgs2 = [affine_transfomer(mask_img) for _ in range(num_transform)]
    
    image_dir=os.path.join(path,'image') if path else os.getcwd()
    mask_dir= os.path.join(path,'mask') if path else os.getcwd()
    
    for idx,img in enumerate(affine_imgs):
        img.save(os.path.join(image_dir,basename+'_orig_affine_'+str(idx)+'.png'))
    
    for idx,img in enumerate(affine_imgs2):
        img.save(os.path.join(mask_dir,basename+'_mask_affine_'+str(idx)+'.png'))
        
    if verbose:
        plt.figure(figsize=(18,6))
        plot(affine_imgs,with_orig=False)
        plot(affine_imgs2,with_orig=False)
        
        
def jitter_and_blur(orig_img=None,num_transform=5,path=None, kernel_size=(11, 11), sigma=(5, 15),verbose=True):
    
    basename=get_basename(orig_img.filename)
    
    if path: basename=os.path.join(path,basename)
    
    tr=T.Compose([T.ColorJitter(brightness=.5, hue=.2),T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)])

    tr_imgs = [tr(orig_img) for _ in range(num_transform)]
    
    if verbose : plot(tr_imgs)
        
        
    [img.save(basename+'jtblr'+str(idx)+'.png') for idx,img in enumerate(tr_imgs)]
        

def get_basename(fname):
    return fname.split('.')[0]


def bsc(img_file='col_file2_Color_Color.png',num_transform=5,path=None):
    #img_file : filename
    image = FImage(img_file)
    basename=get_basename(img_file)
    
    if path: basename=os.path.join(path,basename)
    
    contrasts=np.random.choice(np.linspace(5,30,10),num_transform, replace=False).astype('int')
    saturations=np.random.choice(np.linspace(2,20,10),num_transform, replace=False).astype('int')
    brightness=np.random.choice(np.linspace(2,15,10),num_transform, replace=False).astype('int')
    
    for idx,(i1,i2,i3) in enumerate(zip(contrasts,saturations,brightness)):
        
        image.apply(Saturation(i1),
                    Contrast(i2),
                    Brightness(i3))


        # save the image with the applied preset
        image.save(basename+'bsc'+str(idx)+'.png')


def sepia(img_file='col_file2_Color_Color.png',num_transform=5,path=None):
    basename=get_basename(img_file)
    
    if path: basename=os.path.join(path,basename)
    
    choices=np.random.choice(np.linspace(5,70,15),num_transform, replace=False).astype('int')
    
    for idx,choice in enumerate(choices):
        image = FImage(img_file)
        image.apply(Sepia(choice))

        image.save(basename+'sep'+str(idx)+'.png')
  
        
def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False,figsize=(12,6))
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    
    
def create_dir(path=None):
    if not(os.path.exists(os.path.join(path,'mask'))):
        os.mkdir(os.path.join(path,'mask'))
        
    if not(os.path.exists(os.path.join(path,'image'))):
        os.mkdir(os.path.join(path,'image'))
from scipy import ndimage
import cv2
import numpy as np
from models.inference_unet import *
import os
from torchvision import transforms
from typing import NamedTuple, List, Callable, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from utils2.datastruct import DataStruct
from skimage.color import rgb2lab
import xlsxwriter
import string
import pickle

class color_segment(ABC):
    
    def __init__(self,
                 image,
                 mask,
                 squeeze_fac = 0.4, 
                 col_list = [8]*12):
        
        self.image = image
        self.mask = mask
        self.squeeze_fac = squeeze_fac
        self.mask_orig = None
        self.instance_mask = None
        self.instance_mask_sorted = None
        self.bboxes_sorted = None
        self.col_list = col_list
        self.color_list = None
        self.err_list = None
        self.mode = None


    def squeeze_mask(self):
        '''
        reduces mask area by shortening ellipses mmajor and minor axis
        
        '''
        fac = self.squeeze_fac
        image = self.image
        mask = self.mask
        # fit BBoxes
        contours = cv2.findContours(mask,  cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        bboxes = []
        #dummy = np.zeros(image.shape).astype('uint8')
        #result = image.copy()
        result = np.zeros(image.shape).astype('uint8')
        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            if w*h>30 : 
                bboxes.append([x,y,w,h])
        bboxes = np.array(bboxes)
        bboxes[:,0] = bboxes[:,0] + bboxes[:,2]/2
        bboxes[:,1] = bboxes[:,1] + bboxes[:,3]/2
        bboxes[:,2] = (1-fac)*bboxes[:,2]/2
        bboxes[:,3] = (1-fac)*bboxes[:,3]/2
        for el in bboxes:
            cv2.ellipse(result, (el[0],el[1]), (el[2],  el[3]), 0, 0,360, [255,255,255], -1)
        self.mask_orig = self.mask
        self.mask = result[:,:,0]
        

    def get_instance_masks(self) -> np.ndarray:
        img = self.image
        mask = self.mask
        label_im, nb_labels = ndimage.label(mask) 
        instance_mask = np.zeros((*img.shape[:2],nb_labels))
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i+1) 
            separate_mask = np.equal(label_im, mask_compare).astype(int) 
            separate_mask[separate_mask == 1] = 1
            instance_mask[:,:,i] = separate_mask    
        self.instance_mask = instance_mask
        assert np.mod(instance_mask.shape[-1],8) == 0 , 'instance masks are not integer multiple of 8!' 
        self.col_list = self.col_list if self.col_list else [8]*(instance_mask.shape[-1]//8)
        print('col list modified : {}'.format(self.col_list))
    
    
    def sort_instance_masks(self) :
        '''
        takes instance mask and group mask accoriding to col_list. Use bounding box estimates on each masks 
        for sorting.
        args :
         - instance_mask : ndarray representing num wells containing liquids
         - col_list : group columns 

        reuturns :
         - instance_mask_sorted : instance sorted and grouped
         - bboxes_sorted : sorted bounding boxes

        '''
        instance_mask = self.instance_mask
        img = self.image
        col_list = self.col_list
        lst_bbox = [] # bbox for sorting
        for i in range(instance_mask.shape[2]):
            separate_mask = instance_mask[:,:,i]
            lst_bbox.append(np.array(get_bboxes(img,separate_mask.astype('uint8'))).ravel())
        dummy =[]
        dummy_ids =[]
        for idx, i in enumerate(lst_bbox):
            if i.size > 0 :
                if i[2]*i[3]>100 and i.size<5:
                    dummy.append(i)
                    dummy_ids.append(idx)
        lst_bbox = dummy
        instance_mask= instance_mask[:,:,dummy_ids]
        lst_bbox = np.array(lst_bbox)
        idx = [idx for idx, el in enumerate(lst_bbox) if el.any()]
        instance_mask = instance_mask[:,:,idx]
        lst_bbox = np.array([lst_bbox[k].tolist() for k in idx])
        # x-sort
        idx = lst_bbox[:,0].argsort()
        lst_bbox = lst_bbox[idx]
        instance_mask = instance_mask[:,:,idx]
        # ysort provided by user
        bboxes_sorted =[]
        instance_mask_sorted = []
        start = 0
        for num in col_list:
            bboxes_sorted.append(lst_bbox[start:start+num])
            instance_mask_sorted.append(instance_mask[:,:, start:start+num])
            start += num
        # sort y
        #bboxes_sorted2 =#lst_sorted2 =[]
        for i, (el,els) in enumerate(zip(bboxes_sorted,instance_mask_sorted)):
            idx = el[:,1].argsort()
            bboxes_sorted[i] = el[idx]
            instance_mask_sorted[i] = els[:,:,idx]
        #print(bboxes_sorted)
        self.instance_mask_sorted = instance_mask_sorted
        self.bboxes_sorted = bboxes_sorted
     

    def get_colors_from_patches(self, mode = 'rgb', background_rgb = [255,255,255], background_std = np.array([1e-8,1e-8,1e-8]), verbose = False):
        
        img = self.image
        instance_mask_sorted = self.instance_mask_sorted
        color_list =[]
        err_list = []
        self.mode = mode
        errfn = lambda r,e1,b,e2 : np.sqrt((1/r)**2 * e1**2 + (1/b)**2 * e2**2)
        epsi = 1e-12

        '''
        if background_rgb:
            background_std = np.array([1e-8,1e-8,1e-8])
            if not(isinstance(background_rgb,np.ndarray)) : background_rgb = np.array(background_rgb)
            
            print('background set to user-defined values')
            print('background values {}'.format(background_rgb))

        else :
            w = img.shape[0]
            #background_rgb = img[img.shape[0]-10 :img.shape[0],img.shape[0]-10 :img.shape[0],:].mean(axis = (0,1))
            #background_std = img[img.shape[0]-10 :img.shape[0],img.shape[0]-10 :img.shape[0],:].std(axis = (0,1))

            background_rgb = img[int(w//2-5):int(w//2+5),int(w-10):w,:].mean(axis = (0,1))
            background_std = img[int(w//2-5):int(w//2+5),int(w-10):w,:].std(axis = (0,1))
            background_rgb = (background_rgb * 255).astype('uint8')
            background_std = (background_std * 255).astype('uint8')
            
            print('background obtained from middle-east of resized image')
            print('background values {}'.format(background_rgb))
        '''        
        #print(background_rgb)
        img = (img * 255).astype('uint8')
        # lab processing
        if mode == 'lab':
            # lab processing
            if verbose : print('running lab mode')
            for i in instance_mask_sorted:
                mask = i
                dummy =[]
                dummy_err = []
                im1 = np.zeros(img.shape)
                for j in range(mask.shape[2]):
                    im1[:,:,0]= mask[:,:,j] * img[:,:,0]
                    im1[:,:,1]= mask[:,:,j] * img[:,:,1]
                    im1[:,:,2]= mask[:,:,j] * img[:,:,2]
                    lab = rgb2lab(im1[mask[:,:,j]>0].astype('uint8')).mean(axis = 0)
                    err  = rgb2lab(im1[mask[:,:,j]>0].astype('uint8')).std(axis = 0)
                    dummy.append(lab.tolist())
                    dummy_err.append(err.tolist())
                color_list.append(dummy)
                err_list.append(dummy_err)
        else : # rgb-resolved processing
            if verbose : print('running rgb-resolved mode')
            for i in instance_mask_sorted:
                mask = i
                dummy =[]
                dummy_err = []
                im1 = np.zeros(img.shape)
                for j in range(mask.shape[2]):
                    im1[:,:,0]= mask[:,:,j] * img[:,:,0]
                    im1[:,:,1]= mask[:,:,j] * img[:,:,1]
                    im1[:,:,2]= mask[:,:,j] * img[:,:,2]
                    rgb = im1[mask[:,:,j]>0].mean(axis = 0)
                    err = im1[mask[:,:,j]>0].std(axis = 0)
                    dummy.append(np.log(background_rgb/(rgb+epsi)).tolist())
                    dummy_err.append(errfn(rgb,err,background_rgb,background_std).tolist())
                color_list.append(dummy)
                err_list.append(dummy_err)  
        self.color_list = np.array(color_list)
        self.err_list = np.array(err_list)


    @abstractmethod
    def analyze_wells_from_patches(self):
        pass
    

class wellsegment(color_segment):
    def __init__(self,
                 path = r'C:\work\biplab\OpenTron\TestSMB\unilever_test\04082023',
                 file_initial = 'unitest_',
                 use_file_idx = 1,
                 squeeze_fac = 0.4, 
                 col_list = None
                 ):
        '''
        This function performs segmentataion of wells & extract colors from patches
        '''
        
        ##########################################
        # sort pngs or jpgs
        self.path = path
        self.file_list = wellsegment.get_files(path, initials=file_initial)
        if len(self.file_list) == 0 : raise ValueError('file not found')
        model_path = os.path.join('..', 'weights')
        #model_path = r'C:\work\biplab\OpenTron\dev\weights'
        model_weights = 'unet_params_104_lr_1e-05.pt'
        # model init
        self.model = get_trained_unet(model_name='unet_vgg11_upsampling',
                       path=model_path,
                       params=model_weights)
        self.file = self.file_list[use_file_idx] # take first file
        self.col_list = col_list
        (self.image,self.mask) = get_inference(self.model,
                                               image_path = os.path.join(self.path,self.file),
                                               transform_hr = transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Resize((512,512), antialias=True)]))
          
        super().__init__(self.image , self.mask , squeeze_fac = squeeze_fac, col_list = col_list)
        self.squeeze_mask()
        self.get_instance_masks()
        self.sort_instance_masks()
        self.out = None


    def extract(self, 
                analysis_mode = 'patch', 
                color_mode ='rgb',
                background_rgb = [255,255,255],
                frame_interval = 30,
                verbose = True):
        img = self.image
        mask = self.mask
        self.analysis_mode = analysis_mode
        if background_rgb:
            background_std = np.array([1e-8,1e-8,1e-8])
            if not(isinstance(background_rgb,np.ndarray)) : background_rgb = np.array(background_rgb)
            
            print('background set to user-defined values')
            print('background values {}'.format(background_rgb))
        else :
            w = img.shape[0]
            background_rgb = img[int(w//2-5):int(w//2+5),int(w-10):w,:].mean(axis = (0,1))
            background_std = img[int(w//2-5):int(w//2+5),int(w-10):w,:].std(axis = (0,1))
            background_rgb = (background_rgb * 255).astype('uint8')
            background_std = (background_std * 255).astype('uint8')
            print('background obtained from middle-east of resized image')
            print('background values {}'.format(background_rgb))
        self.background_rgb = background_rgb
        self.background_std = background_std
        self.frame_interval = frame_interval
        self.verbose = verbose
        self.color_mode = color_mode
        # for h2o2, 'patch' mode was selected   
        if analysis_mode == 'patch':
            out = self.analyze_wells_from_patches(self.file_list,
                                             path = self.path,
                                             crop = 0.2,
                                             instance_mask_sorted = self.instance_mask_sorted,
                                             mode  = self.color_mode,
                                             frame_interval= self.frame_interval,
                                             verbose = True)
        # kinetics plot
        if self.verbose : plot_kinetics(os.path.join(self.path,'results'), out, channel = 'g')
        
    
    def get_background(self):
        image = self.image #(512,512)
        w = image.shape[0]
        print('background obtained from middle east of resized image')
        print('background values')
        self.background_rgb = image[int(w//2-5):int(w//2+5),int(w-10):w,1].mean()
        

    def analyze_wells_from_patches(self,
                                   file_list : Union[np.ndarray, List],
                                   path : str,
                                   crop : float,
                                   instance_mask_sorted : List, 
                                   mode : str = 'rgb-resolved',
                                   channel = 'g',
                                   frame_interval = 30, # seconds
                                   verbose : bool = False):
        transform_hr = transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))])
        print('extracting  signals from image files & masks')
        for idx , file in tqdm(enumerate(file_list)):
            img=Image.open(os.path.join(path,file)) #consisitent with torch's bilin interpolation
            if crop : img = wellsegment.crop_img(img,crop)
            img=transform_hr(img).permute(1,2,0).numpy()
            self.image = img
            # single color
            self.get_colors_from_patches(mode = self.mode, 
                                         background_rgb = self.background_rgb, 
                                         background_std = self.background_std, 
                                         verbose = False)
            colors, errs = self.color_list, self.err_list
            #print(len(colors))
            if idx == 0 : 
                dat = {k : [] for k in range(len(colors))}
                daterr = {k : [] for k in range(len(colors))}
            #print(dummy)
            for idx2, color in enumerate(colors):
                dat[idx2].append(np.array(color))
                daterr[idx2].append(np.array(errs))
        out = DataStruct(lab = (dat if mode == 'lab' else None),
                         rgb = (dat if mode == 'rgb' or mode =='rgb-resolved' else None),
                         laberr = (daterr if mode == 'lab' else None),
                         rgberr = (daterr if mode == 'rgb' or mode =='rgb-resolved' else None),
                         mask = instance_mask_sorted,
                         t = np.arange(0,len(file_list),1)*frame_interval
                        )
        self.out = out
        path = os.path.join(path,'results')
        if not(os.path.exists(path)) :
            os.mkdir(path) 
        if verbose: 
            visualize_patches(path,img,instance_mask_sorted)
        save_data_json(path, self.out, channel)
        save_rawkinetics_xlsx(path = path, file = 'data.json',frame_interval = frame_interval)
        save_pkl(path, out)
        return out
    
    
    @staticmethod
    def get_files(path : str, initials : str = None) -> List:
        '''
        return : function returns sorted file list
        args :
         - path : full path
         - initials : file name initials, e.g. for filename unitest_123454.png, use initials 'u'/'uni','unitest', etc.
        '''

        if initials: 
            if not(isinstance(initials, str)):
                raise TypeError('initials must be a string')
        os.chdir(path)
        ext = initials + '*.png' if initials else '*.png'
        file_list = np.array(glob(ext))
        #len(lst)

        # sort list
        sort_idx = np.array([int(file.split('.')[0].split('_')[-1]) for file in file_list]).astype('int')
        file_list = file_list[sort_idx.argsort()]

        return file_list
    

    @staticmethod
    def crop_img(img,fac =0.1):
        (w,h) = img.size
        left = int(fac * w)
        right = int(w -fac*w)
        top = int(fac*h)
        bottom = int(h -fac*h)
        im2 = img.crop((left,top,right,bottom))
        return im2
    
        
def visualize_patches(path, img,instance_mask_sorted):
    z = np.zeros((512,512))
    for instance in instance_mask_sorted:
        z += np.sum(instance,axis = 2)
    plt.subplot(1,2,1);plt.imshow(img, cmap = 'gray');plt.axis('off')
    plt.subplot(1,2,2);plt.imshow(z, cmap = 'gray');plt.axis('off')
    plt.savefig(os.path.join(path,'segmentation_'+'.png'),bbox_inches='tight')
    

def plot_kinetics(path, out, channel = 'g'):
    dic = {'r':0, 'g': 1, 'b': 2}
    for idx, data in enumerate(out.rgb.values()):
        plt.figure()
        for k in range(len(data[0])) :
            plt.plot(out.t/60, [x[k,dic[channel]] for x in data])
            #print(len(data))
            #break
            plt.xlabel('t[min]')
            plt.ylabel('rgb-resolved signal')
        plt.savefig(os.path.join(path,'kinetics_col_'+str(idx)+'.png'),bbox_inches='tight')


def save_rawkinetics_xlsx(path = r'C:\work\biplab\OpenTron\TestSMB\unilever_test\04082023',
                          file = 'data.json',
                          frame_interval = 0.5, # in minute
                          ):
    f = open(os.path.join(path,'data.json'))
    data = json.load(f)
    f.close()

    workbook = xlsxwriter.Workbook(os.path.join(path,'rawdata.xlsx'))
    row = 1
    for count,(item,well_column) in enumerate(data.items()):
        print(item)
        if count == 0 : x = np.arange(0,len(well_column[0]),1)*frame_interval
        sheet_data = workbook.add_worksheet(item)
        for idx, well in enumerate(well_column):
            sheet_data.write(string.ascii_letters[idx+1].upper()+'1','wellrow_'+str(idx+1))
            sheet_data.write_column(row, idx+1, well)
            
        sheet_data.write('A1','t[s]')
        sheet_data.write_column(row, 0 , x.tolist())

    workbook.close()
    
    
def save_pkl(path,out):
    with open(os.path.join(path,'out.pkl'), 'wb') as file:
        pickle.dump(out, file)        

    
def save_data_json(path : str, out : DataStruct, channel : str = 'g'):
    # save data as json
    dic = {'r':0, 'g': 1, 'b': 2}
    datadic = {}
    for key, data in out.rgb.items(): # num columns
        dummy = []
        for k in range(len(data[0])) : # num wells
            dummy.append([x[k,dic[channel]] for x in data])
        datadic['col_'+ str(key)] = dummy    
    # save file     
    with open(os.path.join(path, 'data.json'), 'w') as outfile:
        json.dump(datadic, outfile)
        

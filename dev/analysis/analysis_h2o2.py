from scipy import ndimage
import cv2
import os
import numpy as np
from typing import NamedTuple, List, Callable, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from glob import glob
import matplotlib.pyplot as plt
import string
import pandas as pd
import pickle 
from skimage.color import rgb2lab
from torchvision import transforms
from multipledispatch import dispatch
from models.inference_unet import *
from analysis.analysis_base import *


class wellsegment_h2o2(color_segment):
    def __init__(self, 
                 image_path, 
                 image_file,
                 mask_dict = None,
                 squeeze_fac = 0.4, 
                 col_list = [8]*12,
                 path =r'D:\All_files\pys\AI_algos\WhiteWellPlate_Segmentation',
                 ):
                 
        model_path = os.path.abspath('weights')   
        model_weights = 'unet_params_104_lr_1e-05_h2o2.pt'
        self.image = None
        self.mask = None
        self.col_list = col_list
        self.mask_dict = mask_dict
        self.squeeze_fac = squeeze_fac
        if mask_dict:
            self.preprocess_image(image_path, image_file, 0.2)
            self.mask_dict = mask_dict
        else :
            gen = get_trained_unet(model_name='unet_vgg11_upsampling',
                                   path = model_path,
                                   params = model_weights)
            (image,mask) = get_inference(gen,
                                         image_path= os.path.join(image_path, image_file),
                                         transform_hr=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Resize((512,512), antialias=True)]))
            print(image.shape)
            print(mask.shape)
            del gen
            self.mask = mask
            self.image = image
            super().__init__(self.image,self.mask,self.squeeze_fac,self.col_list)
        self.wells = [i + str(k) for k in range(1,13) for i in string.ascii_uppercase[:8]]
        self.reservoir_list = None
        self.path = path if path else None #: path = r'D:\All_files\pys\AI_algos\WhiteWellPlate_Segmentation'
                    
        
    def extract(self):
        self.squeeze_mask()
        self.get_instance_masks()
        self.sort_instance_masks() 
        # asign mask with wells
        mask_sorted = self.instance_mask_sorted
        dummy = []
        for k in range(len(mask_sorted)):
            for j in range(mask_sorted[k].shape[-1]):
                dummy.append(mask_sorted[k][:,:,j])
        dummy = np.array(dummy)
        self.instance_mask_sorted = {key : val for key, val in zip(self.wells,dummy)}
        fname = os.path.join(self.path,'mask.pkl') if self.path else 'mask.pkl'
        with open(fname, 'wb') as f:  
            pickle.dump(self.instance_mask_sorted, f)
    

    def analyze_wells_from_bboxes(self):
        print('analyze wells from bboxes')
        

    @classmethod
    def load_pkl(cls, image_path, image_file, pkl_file):
        with open(pkl_file, 'rb') as f:  
            mask_dict = pickle.load(f) 
        print(mask_dict.keys())
        return wellsegment_h2o2(image_path, image_file, mask_dict = mask_dict)
        
        
    @staticmethod
    def visualize_patches(mask_dict, lst =  ['E12','F12','G12','H12']):
        x = 0
        for k in lst:
            x +=mask_dict[k]
        plt.imshow(x)
        
        
    @staticmethod
    def visualize(image, mask):
        plt.figure(figsize=(8,6))
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(mask, cmap = 'gray')
        
    # save analysis to smb
    def save_to_smb_(self):
        path_smb = os.path.abspath(r'../DriveSMB')
        pd.DataFrame({'reservoir_list': self.reservoir_list}).to_csv(os.path.join(path_smb, 'reservoir_list.csv'))


    def analyze_prestimation(self, tcdict, params, boundary, larger_than_boundary):
        well_list = np.array(([el[1] for el in tcdict])).T   
        labs = rgb2lab(self.image)
        reservoir_list = np.empty(well_list.shape[0], dtype=object)
        # estimate colors
        for idx, cols in enumerate(well_list): #[[E1,F1,G1,H1],[E2,F2,G2,H2]]
            for well in cols:
                lab_well = labs.copy()
                mask = self.mask_dict[well]
                lab_well_a = lab_well[:,:,1] * mask 
                lab_well_b = lab_well[:,:,2] * mask 
                # Assuming lab_well_a and lab_well_b are NumPy arrays:
                hues = np.degrees(np.arctan2(lab_well_b, lab_well_a))
                hues = np.where(hues < 0, 360 + hues, hues)
                hue_well = hues.copy()
                hue_well = hue_well * mask
                if params == 'a*':
                    val = lab_well_a[lab_well_a > 0]
                elif params == 'hue':
                    val = hue_well[hue_well > 0]
                else: 
                    print('unknown params')
                if len(val) > 0 :
                    if larger_than_boundary:
                        if val.mean() < boundary: # the first data point higher than the boundary
                            print(val.mean())
                            reservoir_list[idx] = well
                            break
                    else:
                        if val.mean() > boundary: # the first data point lower than the boundary
                            print(val.mean())
                            reservoir_list[idx] = well
                            break
        if (reservoir_list == None).all() : 
            raise ValueError('could not find a well meeting the hue condition')
        self.reservoir_list = reservoir_list


    @dispatch(str, str, float)    
    def preprocess_image(self, path, file, crop):
        img=Image.open(os.path.join(path,file)) #consisitent with torch's bilin interpolation
        if crop : img = wellsegment_h2o2.crop_img(img,crop)
        self.image = wellsegment_h2o2.t_hr(img)


    @staticmethod
    def preprocess_image_v2(file,crop):
        img=Image.open(file) #consisitent with torch's bilin interpolation
        if crop : img = wellsegment_h2o2.crop_img(img,crop)
        return wellsegment_h2o2.t_hr(img)
        

    @staticmethod
    def crop_img(img,fac =0.1):
        (w,h) = img.size
        left = int(fac * w)
        right = int(w -fac*w)
        top = int(fac*h)
        bottom = int(h -fac*h)
        im2 = img.crop((left,top,right,bottom))
        return im2
    

    @staticmethod
    def t_hr(img):
        transform_hr = transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512), antialias=True)])
        return transform_hr(img).permute(1,2,0).numpy()
        

    def analyze_wells_from_patches(self):
        print('analyze wells from patches')

    
    def analyze_titration(path_cam, params, vol_step = 5):
        '''
        analyze titration result
        '''
        if os.path.exists(os.path.join(path_cam, 'mask','mask.pkl')) :
            pkl_file = os.path.join(path_cam, 'mask','mask.pkl')
            with open(pkl_file, 'rb') as f:  
                mask_dict = pickle.load(f)
        else:
            raise FileNotFoundError('pickle file not found!')
        if os.path.exists(os.path.join(path_cam, 'titration')) :
            path_titrate = os.path.join(path_cam, 'titration')
        else:
            raise FileNotFoundError('titration folder does not exist!')
        for sub_folder in os.listdir(path_titrate):
            path = os.path.join(path_titrate, sub_folder)
            df = pd.read_csv(os.path.join(path, 'msi2pi_h2o2.csv'))
            # sorting files
            files_uns = glob(os.path.join(path_titrate,sub_folder,'*_titration_*.jpg'))
            dummy = [file.split('\\')[-1] for file in files_uns]
            dummy = np.array([int(file.split('_')[3]) for file in dummy])
            idx = dummy.argsort()
            files = [files_uns[k] for k in idx]
            # Save for each folder
            dummy = []
            a_star_values = []  
            b_star_values = [] 
            l_values = [] 
            for file in files:
                print(file.split('\\')[-1])
                image = wellsegment_h2o2.preprocess_image_v2(file, .2)
                labs = rgb2lab(image)
                dummy_l = []
                dummy_a = []
                dummy_b= [] 
                for well in df.loc[:, 'cols'].tolist():  # Iterate over wells
                    lab_well = labs.copy()
                    mask = mask_dict[well]
                    lab_well_l = (lab_well[:, :, 0] * mask).flatten()
                    lab_well_a = (lab_well[:, :, 1] * mask).flatten()
                    lab_well_b = (lab_well[:, :, 2] * mask).flatten()
                    # Append results
                    dummy_l.append(lab_well_l.sum()/(lab_well_a.shape[0] - (lab_well_a==0).sum()))
                    dummy_a.append(lab_well_a.sum()/(lab_well_a.shape[0] - (lab_well_a==0).sum())) 
                    dummy_b.append(lab_well_b.sum()/(lab_well_b.shape[0] - (lab_well_b==0).sum())) 
                l_values.append(dummy_l)
                a_star_values.append(dummy_a)
                b_star_values.append(dummy_b)
            # Convert to NumPy arrays
            l_values = np.array(l_values)
            a_star_values = np.array(a_star_values)
            b_star_values = np.array(b_star_values)
            # Create DataFrame
            out_l = pd.DataFrame({f'{well}_l': l_values[:, idx] for idx, well in enumerate(df.loc[:, 'cols'].tolist())})
            out_a = pd.DataFrame({f'{well}_a*': a_star_values[:, idx] for idx, well in enumerate(df.loc[:, 'cols'].tolist())})
            out_b = pd.DataFrame({f'{well}_b*': b_star_values[:, idx] for idx, well in enumerate(df.loc[:, 'cols'].tolist())})
            # Add mean and std columns
            out = pd.DataFrame({'vol': np.arange(0, a_star_values.shape[0]) * vol_step})
            out['mean_l'] = [out_l.iloc[k, :].mean() for k in range(out_l.shape[0])]
            out['std_l'] = [out_l.iloc[k, :].std() for k in range(out_l.shape[0])]
            out['mean_a*'] = [out_a.iloc[k, :].mean() for k in range(out_a.shape[0])]
            out['std_a*'] = [out_a.iloc[k, :].std() for k in range(out_a.shape[0])]
            out['mean_b*'] = [out_b.iloc[k, :].mean() for k in range(out_b.shape[0])]
            out['std_b*'] = [out_b.iloc[k, :].std() for k in range(out_b.shape[0])]
            # Combine all DataFrames
            final_out = pd.concat([out, out_l, out_a, out_b], axis=1)
            # Save to CSV
            final_out.to_csv(os.path.join(path, 'titration_result.csv'), index=False)
            if params == 'hue':
                # Define a function to calculate the hue value
                def calculate_mean_hue(a_star, b_star):
                    hue_angle_rad = np.arctan2(b_star, a_star)  # Calculate hue in radians
                    hue_angle_deg = np.degrees(hue_angle_rad) % 360  # Convert to degrees and ensure [0, 360]
                    if hue_angle_deg < 10: # in case that purple colour is too close to the zero point and facilitate fitting analysis
                        return 360-hue_angle_deg
                    return hue_angle_deg 
                def calculate_std_hue(mean_a, mean_b, std_a, std_b):
                    hue_rad = np.arctan2(mean_b, mean_a) % 360
                    denom = mean_a**2 + mean_b**2 + 1e-10
                    d_hue_da = - mean_b/denom
                    d_hue_db = mean_a/denom
                    std_hue_rad = np.sqrt((d_hue_da**2 * std_a**2) + (d_hue_db**2 * std_b**2))
                    std_hue_deg = np.degrees(std_hue_rad)
                    return std_hue_deg   
                # Calculate the mean hue value using mean_a* and mean_b*
                file_path = os.path.join(path, 'titration_result.csv')
                data = pd.read_csv(file_path)
                data['mean_hue'] = data.apply(lambda row: calculate_mean_hue(row['mean_a*'], row['mean_b*']), axis=1)
                data['std_hue'] = data.apply(lambda row: calculate_std_hue(row['mean_a*'], row['mean_b*'], row['std_a*'], row['std_b*']), axis=1)
                # Save the updated DataFrame to a new CSV file
                output_file_path = file_path
                data.to_csv(output_file_path, index=False)
            else:
                pass

         
def plot_titration(path_cam, params, vol_step = 5):
    if os.path.exists(os.path.join(path_cam, 'titration')) :
        path_titrate = os.path.join(path_cam, 'titration')
    else:
        raise FileNotFoundError('titration folder does not exist!')
    vol = vol_step
    for idx, sub_folder in enumerate(os.listdir(path_titrate)):
        print(sub_folder)
        df = pd.read_csv(os.path.join(path_titrate, sub_folder,'titration_result.csv'))
        vol_list = np.arange(0,df.shape[0],1)*vol if vol else np.arange(0,df.shape[0],1)
        plt.figure(figsize = (6,4))
        if params == 'a*':
            y = df['mean_a*'].to_numpy()
            y_error = df['std_a*'].to_numpy()
            plt.ylabel('a*')
        else:
            y = df['mean_hue'].to_numpy()
            y_error = df['std_hue'].to_numpy()
            plt.ylabel('hue')
        plt.plot(vol_list, y, 'o', color = 'red')
        plt.errorbar(vol_list,y, yerr = y_error, fmt = 'o', ecolor = 'blue', elinewidth = 2, capsize = 5, mfc = 'red', mec ='red')
        plt.xlabel('vol [$\mu$l]')
        plt.title('samp :' + str(idx+1))
        plt.savefig(os.path.join(path_titrate, sub_folder,'titr_plot_err.jpg'))

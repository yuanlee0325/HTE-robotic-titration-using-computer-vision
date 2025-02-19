import os
from utils.realsense import start_realsense
import pandas as pd
from watchfiles import watch
import numpy as np
import shutil
from glob import glob
from time import time
import json
import matplotlib.pyplot as plt
from analysis.analysis_h2o2 import wellsegment_h2o2

class protocol_master_h2o2():

    __path_smb =  os.path.abspath(r'../DriveSMB')
    __smbfile = 'msi2pi_h2o2.csv'

    def __init__(self, 
                 path_cam = r'C:\work\yuan\protcol_h2o2_test\titration' # this path will be overwrite from the H2O2_master_dev2.ipynb
                ):
        self.path = path_cam
        self.cls = None # segmentation maps

    # Initialize the camera  
    def set_realsense(self, frame_interval = 1, stop = 0.001, sensitivity = 150):
        self.frame_interval = frame_interval
        self.stop = stop        
        self.sensitivity = sensitivity
    

    def run(self):
        #################################################################################################################################
        path_smb = os.path.abspath(r'../DriveSMB')
        path_cam = self.path 
        try :
            # create sub folders
            titration_base_path = os.path.join(path_cam,'titration')
            mask_path = os.path.join(path_cam,'mask')
            prest_path = os.path.join(path_cam,'preestimation')
            if not(os.path.exists(titration_base_path)) : os.mkdir(titration_base_path)
            if not(os.path.exists(mask_path)) : os.mkdir(mask_path)
            if not(os.path.exists(prest_path)) : os.mkdir(prest_path)
            for changes in watch(path_smb):
                print(changes)
                out = get_changes()
                print('printing out...................................')
                print(out['file'])
                print(out['mode'])
                print(out['reservoir_list_exist'])
                print('printing out done...................................')
                
                ############################################################################################################################################
                # blancplate imaging
                if out['file'] and out['mode'] == 'init' :
                    print('imaging blancplate & generating instance mask')
                    t1 = time()
                    start_realsense(fname=out['save_filename'],
                                    folder=mask_path,
                                    frame_interval=self.frame_interval,#seconds
                                    stop=self.stop,#hours
                                    take_image=True,
                                    sensitivity = self.sensitivity)
                    cleanup(mask_path,out['save_filename'])
                    # save mask to mask folder
                    # get mask from first image prior to liquid transfer
                    cls = wellsegment_h2o2(mask_path, out['save_filename'] + '3.jpg' ,None, squeeze_fac= 0.4, col_list = [8]*12, path = mask_path)
                    cls.extract() # save pickle file of sorted mask
                    wellsegment_h2o2.visualize(cls.image, cls.mask)
                    t2=time()
                    del cls
                    print('elapsed time on blancplate imaging : {}'.format(t2-t1))
                    
                ###############################################################################################################################################
                # pre-estimation
                if out['file'] and out['mode'] == 'pre-estimation' and not(out['reservoir_list_exist']):
                    print('performing pre-estimation')
                    t1 = time()
                    start_realsense(fname=out['save_filename'],
                                    folder=prest_path,
                                    frame_interval=self.frame_interval,#seconds
                                    stop=self.stop,#hours
                                    take_image=True,
                                    sensitivity = self.sensitivity)
                    cleanup(prest_path,out['save_filename'])
                    # save analysis to smb
                    with open(os.path.join(path_smb,'ot2_transfer_config_h2o2.json'),'r') as f:
                        print('open file')
                        data = json.load(f)
                    tcdict = data['transfer_3']['locs']
                    params = data['transfer_3']['params']
                    boundary = data['transfer_3']['boundary']
                    larger_than_boundary = data['transfer_3']['larger_than_boundary']
                    cls2 = wellsegment_h2o2.load_pkl(prest_path,'preestimation_3.jpg',os.path.join(mask_path, 'mask.pkl'))
                    cls2.analyze_prestimation(tcdict, params, boundary, larger_than_boundary)
                    cls2.save_to_smb_()
                    self.cls = cls2 
                    t2=time()
                    print('elapsed time on pre-estimation : {}'.format(t2-t1))

                ###############################################################################################################################################
                # titration
                if out['file'] and out['mode'] == 'titration' :
                    print('acquiring images during titration')
                    titration_path = os.path.join(titration_base_path, out['save_location'])
                    if not(os.path.exists(titration_path)) : os.mkdir(titration_path)
                    shutil.copyfile(os.path.join(path_smb,'msi2pi_h2o2.csv'),os.path.join(titration_path, 'msi2pi_h2o2.csv'))
                    start_realsense(fname=out['save_filename'],
                                    folder=titration_path,
                                    frame_interval=self.frame_interval,#seconds
                                    stop=self.stop,#hours
                                    take_image=True,
                                    sensitivity = self.sensitivity)
                    
                    cleanup(titration_path,out['save_filename'])
                # stop
                if out['file'] and out['mode'] == 'exit':
                    print('exiting')
                    break
        except KeyboardInterrupt:
            print('interrupted')
            delete_files(path_smb)
            if os.path.exists(os.path.join(path_smb,'msi2pi_h2o2.csv')) : 
                os.remove(os.path.join(path_smb,'msi2pi_h2o2.csv'))
        except Exception as e:
            print(e)   
        delete_files(path_smb)       


def get_changes():
    path_smb = os.path.abspath(r'../DriveSMB')
    titration_count = False
    reservoir_list_exist = False
    fileExist = False
    if os.path.exists(os.path.join(path_smb,'msi2pi_h2o2.csv')) :
        fileExist = True
        data = pd.read_csv(os.path.join(path_smb,'msi2pi_h2o2.csv'))
        titration_count = None
        save_location = None
        save_filename = None
        wells = None
        copyfile = False
        if data.loc[0,'mode'] == 'init' : 
            save_filename = 'blancplate_'

        if data.loc[0,'mode'] == 'pre-estimation' : 
            save_filename = 'preestimation_'

        if data.loc[0,'mode'] == 'titration':
            titration_count = data.loc[0,'titration_count']
            save_filename = 'samp_' + str(data.loc[0,'id'].astype('int'))+ '_titration_' + str(data.loc[0,'titration_count'].astype('int'))+ '_'
            save_location = 'samp_' + str(data.loc[0,'id'].astype('int'))
            wells = data.loc[:,'cols'].to_list()
            copyfile = True
        if os.path.exists(os.path.join(path_smb, 'reservoir_list.csv')):
            reservoir_list_exist = True
    out = {'file': fileExist, 
           'mode' : data.loc[0,'mode'],
           'titration_count' : titration_count, 
           'save_filename' : save_filename,
           'save_location' : save_location,
           'wells' : wells,
           'copyfile' : copyfile,
           'reservoir_list_exist' : reservoir_list_exist}
    return out


def cleanup(path, pattern):
    lst = glob(os.path.join(path,pattern+'*.jpg'))
    [os.remove(file) for idx,file in enumerate(lst[:-1])]


def delete_files(path_smb):
    if os.path.exists(os.path.join(path_smb,'msi2pi_h2o2.csv')) : os.remove(os.path.join(path_smb,'msi2pi_h2o2.csv'))
    if os.path.exists(os.path.join(path_smb,'reservoir_list.csv')) : os.remove(os.path.join(path_smb,'reservoir_list.csv'))
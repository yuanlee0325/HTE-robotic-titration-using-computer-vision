import json
import os 
import numpy as np
from multipledispatch import dispatch
import string
from abc import ABC, abstractmethod
from setting.settings_base import system_config, transfer_config, utility


class system_config_h2o2(system_config):

    # default config including tiprack_1, tiprack_2, plate_1, camera, reservoir_1, reservoir_2, pipette_1, pipette_2
    __a_dict = {} 

    __a_dict['tiprack_1'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['tiprack_1']['type'] = 'opentrons_96_tiprack_20ul'
    __a_dict['tiprack_1']['pos'] = 10
    __a_dict['tiprack_1']['offset']['x'] = -0.7
    __a_dict['tiprack_1']['offset']['y'] = 0.9
    __a_dict['tiprack_1']['offset']['z'] = 0

    __a_dict['tiprack_2'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['tiprack_2']['type'] = 'opentrons_96_tiprack_20ul'
    __a_dict['tiprack_2']['pos'] = 11
    __a_dict['tiprack_2']['offset']['x'] = -0.5
    __a_dict['tiprack_2']['offset']['y'] = 0.7
    __a_dict['tiprack_2']['offset']['z'] = 0

    __a_dict['tiprack_3'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['tiprack_3']['type'] = 'opentrons_96_tiprack_300ul'
    __a_dict['tiprack_3']['pos'] = 1
    __a_dict['tiprack_3']['offset']['x'] = -0.7
    __a_dict['tiprack_3']['offset']['y'] = 0.9
    __a_dict['tiprack_3']['offset']['z'] = 0
    
    __a_dict['tiprack_4'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['tiprack_4']['type'] = 'opentrons_96_tiprack_300ul'
    __a_dict['tiprack_4']['pos'] = 8
    __a_dict['tiprack_4']['offset']['x'] = -0.7
    __a_dict['tiprack_4']['offset']['y'] = 0.9
    __a_dict['tiprack_4']['offset']['z'] = 0

    __a_dict['camera'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['camera']['type'] = 'opentrons_96_tiprack_20ul'
    __a_dict['camera']['pos'] = 3
    __a_dict['camera']['tip'] = 'D3'
    __a_dict['camera']['offset']['x'] = 0.2
    __a_dict['camera']['offset']['y'] = 0.3
    __a_dict['camera']['offset']['z'] = 0

    __a_dict['plate_1'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['plate_1']['type'] = 'corning_96_wellplate_360ul_flat'
    __a_dict['plate_1']['pos'] = 5
    __a_dict['plate_1']['offset']['x'] = 0.7
    __a_dict['plate_1']['offset']['y'] = 0.6
    __a_dict['plate_1']['offset']['z'] = 0.9

    __a_dict['reservoir_1'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['reservoir_1']['type'] = 'azenta_12_reservoir_2100ul'
    __a_dict['reservoir_1']['pos'] = 9
    __a_dict['reservoir_1']['offset']['x'] = 0
    __a_dict['reservoir_1']['offset']['y'] = 0
    __a_dict['reservoir_1']['offset']['z'] = 0

    __a_dict['reservoir_2'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['reservoir_2']['type'] = 'azenta_12_reservoir_2100ul'
    __a_dict['reservoir_2']['pos'] = 6
    __a_dict['reservoir_2']['offset']['x'] = 0
    __a_dict['reservoir_2']['offset']['y'] = 0
    __a_dict['reservoir_2']['offset']['z'] = 0

    __a_dict['reservoir_3'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['reservoir_3']['type'] = 'azenta_12_reservoir_2100ul'
    __a_dict['reservoir_3']['pos'] = 7
    __a_dict['reservoir_3']['offset']['x'] = 0
    __a_dict['reservoir_3']['offset']['y'] = 0
    __a_dict['reservoir_3']['offset']['z'] = 0
    
    # use the 12 column reservoir to replace 2 column reservoir, position could be simplified as A3 & A9
    __a_dict['reservoir_4'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['reservoir_4']['type'] = 'azenta_12_reservoir_2100ul'  
    __a_dict['reservoir_4']['pos'] = 4
    __a_dict['reservoir_4']['offset']['x'] = 0
    __a_dict['reservoir_4']['offset']['y'] = 0
    __a_dict['reservoir_4']['offset']['z'] = 0

    __a_dict['pipette_1'] = {'type':{},'mount' : {}}
    __a_dict['pipette_1']['type'] = 'p300_multi_gen2'
    __a_dict['pipette_1']['mount']='left'

    __a_dict['pipette_2'] = {'type':{},'mount' : {}}
    __a_dict['pipette_2']['type'] = 'p20_single_gen2'
    __a_dict['pipette_2']['mount']='right'
    
    __file = 'ot2_system_config_h2o2.json' # filename (verified)
    __path = os.path.abspath(r'../DriveSMB')
       
    def __init__(self, 
                 a_dict = None, 
                 path = os.path.abspath(r'../DriveSMB'),
                 file = None):
        if a_dict:
            self.dict = a_dict
            print(self.dict)
        else:
            self.dict = system_config_h2o2.__a_dict
            dummy = self.dict
            self.dict.update(dummy) # update configurations dynamically
        self.file = file if file else system_config_h2o2.__file
        self.path = path if path else path


    # used to define a string representation of an instance of the system_config class and it returns a formatted string that displays key components of the systems
    def __repr__(self):
        return 'class system_config.'+ \
                '\ntiprack 1 : {}'.format(self.dict['tiprack_1']) +'\n'+ \
                '*'*100+\
                '\ntiprack 2 : {}'.format(self.dict['tiprack_2']) +'\n'+\
                '*'*100+\
                '\ntiprack 3 : {}'.format(self.dict['tiprack_3']) +'\n'+\
                '*'*100+\
                '\ntiprack 4 : {}'.format(self.dict['tiprack_4']) +'\n'+\
                '*'*100+\
                '\nplate 1 : {}'.format(self.dict['plate_1']) +'\n'+\
                '*'*100+\
                '\nreservoir 1 : {}'.format(self.dict['reservoir_1']) +'\n'+\
                '*'*100+\
                '\nreservoir 2 : {}'.format(self.dict['reservoir_2']) +'\n'+\
                '*'*100+\
                '\nreservoir 3 : {}'.format(self.dict['reservoir_3']) +'\n'+\
                '*'*100+\
                '\nreservoir 4 : {}'.format(self.dict['reservoir_4']) +'\n'+\
                '*'*100+\
                '\ncamera : {}'.format(self.dict['camera']) +'\n'+\
                '*'*100+\
                '\npipette 1 : {}'.format(self.dict['pipette_1']) +'\n'+\
                '*'*100+\
                '\npipette 2 : {}'.format(self.dict['pipette_2']) +'\n'
    
    
    @dispatch()
    def save_json(self):
        with open(os.path.join(self.path, self.file), 'w') as outfile:
            json.dump(self.dict, outfile)
    

    @dispatch(str)
    def save_json(self,flag = 'smb'):
        if flag == 'smb' : 
            with open(os.path.join(self.path, self.file), 'w') as outfile:
                json.dump(self.dict, outfile)
            
            
    @dispatch(str,str)        
    def save_json(self, smb_path, smb_file): 
        with open(os.path.join(smb_path, smb_file), 'w') as outfile:
            json.dump(self.dict, outfile)
        
        
    @classmethod
    def load_from_json(cls, path, file):
        with open(os.path.join(path,file), 'r') as outfile:
            a_dict = json.load(outfile)
        return cls(a_dict, None, None)


class transfer_config_h2o2(transfer_config):
    # default config for each transfer, it contains keys(vol, locs, tiprack, speed, mix, blowout) for transfer 1-3, and extra keys(titration, num_wells) for transfer_4
    __a_dict = {}
      
    # All these parameters will be overwrite in the following setting_h2o2.py and Protocol_Setup_H2O2.ipynb

    # Transfer 1
    # left multichannel p300_mult buffer or indicaotr transfer
    __a_dict['transfer_1'] = {'vol':{},'locs':{},'tiprack':{},'speed':{},'mix':{}, 'blowout':{}}
    __a_dict['transfer_1']['tiprack'] = ['A1', 'A2'] 
    __a_dict['transfer_1']['vol'] = {'A1':100, 'A2':50} # volume aspirate uL
    __a_dict['transfer_1']['locs'] = [['A1','A1:A3'],
                                      ['A2', 'A1:A3']] # transfer bufferin A1 & indicator in A2 to all cols A1, A2, A3
    __a_dict['transfer_1']['mix'] = {'rep': 2, 'vol': 100} # mix 2 times volume 100 uL
    __a_dict['transfer_1']['speed'] ={'aspirate' : 1, 'dispense' : 1} # aspirate & dispense speed 1~5
    __a_dict['transfer_1']['blowout'] = False # blowout after dispense liquid
    
    # Transfer 2
    # left multichannel p300_mult sample transfer
    __a_dict['transfer_2'] = {'vol':{},'locs':{},'tiprack':{},'speed':{},'mix':{}, 'blowout':{}}
    __a_dict['transfer_2']['tiprack'] = ['A2','A3','A4'] 
    __a_dict['transfer_2']['vol'] = 20 
    __a_dict['transfer_2']['locs'] = [['A3','A4'],['A4','A6'],['A8','A7']] 
    __a_dict['transfer_2']['mix'] = {'rep': 2, 'vol': 100} 
    __a_dict['transfer_2']['speed'] ={'aspirate' : 1, 'dispense' : 1}
    __a_dict['transfer_2']['blowout'] = False
    
    # Transfer 3
    # right singlechannel p20_single pre-estimation - transfer titrant
    __a_dict['transfer_3'] = {'vol':{},'locs':{},'tiprack':{},'mix':{}, 'blowout':{}}
    __a_dict['transfer_3']['tiprack'] = ['A2','A3'] 
    __a_dict['transfer_3']['vol'] = 40 
    __a_dict['transfer_3']['locs'] = [['A2','A6'],['A1','A7']] 
    __a_dict['transfer_3']['mix'] = {'rep': 4, 'vol': 100} 
    __a_dict['transfer_3']['speed'] ={'aspirate' : 1, 'dispense' : 1}
    __a_dict['transfer_3']['blowout'] = True

    # Transfer 4 : Titartion
    # right singlechannel p20_single titration - transfer titrant
    __a_dict['transfer_4'] = {'vol':{},'locs':{},'tiprack':{},'mix':{}, 'blowout':{}}
    __a_dict['transfer_4']['tiprack'] = ['A1:H12', 'A1:H12'] 
    __a_dict['transfer_4']['vol'] = 5
    __a_dict['transfer_4']['locs'] = [['A2','A4','A6'],['A1:2:A5']] # ['A1':2:'A5'] means A1, A3, and A5
    __a_dict['transfer_4']['mix'] = {'rep': 4, 'vol': 100} 
    __a_dict['transfer_4']['speed'] ={'aspirate' : 1, 'dispense' : 1}
    __a_dict['transfer_4']['blowout'] = False
    __a_dict['transfer_4']['titration'] = [12]*3 # total titration number
    __a_dict['transfer_4']['num_wells'] = 4  # titration wells number/sample
    __smb_path = os.path.abspath(r'../DriveSMB')
    __smb_file = 'ot2_transfer_config_h2o2.json'

    
    def __init__(self,
                 a_dict = None,
                 path = None, 
                 file = 'ot2_transfer_config_h2o2.json'):
        self.path = path if path else transfer_config_h2o2.__smb_path
        self.file = file if file else transfer_config_h2o2.__smb_file
        if a_dict:
            self.dict = a_dict
        else:
            self.dict = transfer_config_h2o2.__a_dict 
            dummy = self.dict
            self.dict.update(dummy) # update configurations dynamically
      
        
    @dispatch()
    def save_json(self):
        '''
        This function save a json file
        To be used after instantiation and property setting
        implementation of abstract function and overloaded.
        
        '''
        self.__set_reservoirs()
        with open(os.path.join(self.__smb_path, self.__smb_file), 'w') as outfile:
            json.dump(self.dict, outfile)
            
            
    @dispatch(str,str)
    def save_json(self, smb_path, smb_file):
        '''
        This function save a json file
        To be used after instantiation and property setting
        implementation of abstract function and overloaded.
        
        '''
        self.__set_reservoirs()
    
        if smb_file and smb_path:
            with open(os.path.join(smb_path, smb_file), 'w') as outfile:
                json.dump(self.dict, outfile)
        else : 
            raise ValueError('path or file not found')
            
            
    def __set_reservoirs(self):
        if utility.hasColon(self.dict['transfer_1']['tiprack']) :
            self.dict['transfer_1']['tiprack'] = utility.col_broadcast(self.dict['transfer_1']['tiprack'][0]) 
        if utility.hasColon(self.dict['transfer_2']['tiprack']) :
            self.dict['transfer_2']['tiprack'] = utility.col_broadcast(self.dict['transfer_2']['tiprack'][0])
        if utility.hasColon(self.dict['transfer_3']['tiprack']) :
            dummy = self.dict['transfer_3']['tiprack'][0] if isinstance(self.dict['transfer_3']['tiprack'], list) else self.dict['transfer_3']['tiprack']
            print(dummy)
            self.dict['transfer_3']['tiprack'] = utility.get_wells(dummy)
        dummy = []
        for el in self.dict['transfer_4']['tiprack']:
            if utility.hasColon(el) :
                dummy.extend(utility.get_wells(el))
        self.dict['transfer_4']['tiprack'] = dummy   
        self.dict['transfer_1']['locs'] = transfer_config.locs_broadcast(self.dict['transfer_1']['locs'], utility.get_wells_rowwise)
        self.dict['transfer_2']['locs'] = utility.repmat(transfer_config.locs_broadcast(self.dict['transfer_2']['locs'], utility.col_broadcast))
        self.dict['transfer_3']['locs'] = transfer_config.locs_broadcast(self.dict['transfer_3']['locs'], utility.get_wells_rowwise)
        self.dict['transfer_4']['locs'] = utility.transfer4_broadcast(self.dict['transfer_4']['locs'].copy(), self.dict['transfer_4']['num_wells'])
        
        
        
    def __repr__(self):
        return  'class transfer_config.'+ \
                '\ntransfer 1: '+ \
                '\nvol : {}'.format(self.dict['transfer_1']['vol']) +'\n'+ \
                '\nlocs : {}'.format(self.dict['transfer_1']['locs']) +'\n'+ \
                '\ntiprack : {}'.format(self.dict['transfer_1']['tiprack']) +'\n'+ \
                '\nmix : {}'.format(self.dict['transfer_1']['mix']) +'\n'+ \
                '\nblowout : {}'.format(self.dict['transfer_1']['blowout']) +'\n'+ \
                '\nspeed : {}'.format(self.dict['transfer_1']['speed']) +'\n'+ \
                '*'*100+\
                '\ntransfer 2: '+ \
                '\nvol : {}'.format(self.dict['transfer_2']['vol']) +'\n'+ \
                '\nlocs : {}'.format(self.dict['transfer_2']['locs']) +'\n'+ \
                '\ntiprack : {}'.format(self.dict['transfer_2']['tiprack']) +'\n'+ \
                '\nmix : {}'.format(self.dict['transfer_2']['mix']) +'\n'+ \
                '\nblowout : {}'.format(self.dict['transfer_2']['blowout']) +'\n'+ \
                '\nspeed : {}'.format(self.dict['transfer_2']['speed']) +'\n'+ \
                '*'*100+\
                '\ntransfer 3: '+ \
                '\nvol : {}'.format(self.dict['transfer_3']['vol']) +'\n'+ \
                '\nlocs : {}'.format(self.dict['transfer_3']['locs']) +'\n'+ \
                '\ntiprack : {}'.format(self.dict['transfer_3']['tiprack']) +'\n'+ \
                '\nmix : {}'.format(self.dict['transfer_3']['mix']) +'\n'+ \
                '\nblowout : {}'.format(self.dict['transfer_3']['blowout']) +'\n'+\
                '\nspeed : {}'.format(self.dict['transfer_3']['speed']) +'\n'+ \
                '*'*100+\
                '\ntransfer 4: '+ \
                '\nvol : {}'.format(self.dict['transfer_4']['vol']) +'\n'+ \
                '\nlocs : {}'.format(self.dict['transfer_4']['locs']) +'\n'+ \
                '\ntiprack : {}'.format(self.dict['transfer_4']['tiprack']) +'\n'+ \
                '\nmix : {}'.format(self.dict['transfer_4']['mix']) +'\n'+ \
                '\nblowout : {}'.format(self.dict['transfer_4']['blowout']) +'\n'+\
                '\nspeed : {}'.format(self.dict['transfer_4']['speed']) +'\n'+ \
                '\nnum_wells : {}'.format(self.dict['transfer_4']['num_wells']) +'\n'+ \
                '\ntitration_points : {}'.format(self.dict['transfer_4']['titration']) +'\n'+ \
                '*'*100
        
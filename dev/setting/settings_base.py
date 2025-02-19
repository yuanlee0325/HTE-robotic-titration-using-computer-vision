import json
import os 
import numpy as np
from multipledispatch import dispatch
import string
from abc import ABC, abstractmethod

class system_config():
    
    # default config, especially including offset information to avoid collision at the initial stage
    __a_dict = {}
    __a_dict['tiprack_1'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['tiprack_1']['type'] = 'opentrons_96_tiprack_300ul'
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

    __a_dict['plate_1'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['plate_1']['type'] = 'corning_96_wellplate_360ul_flat'
    __a_dict['plate_1']['pos'] = 5
    __a_dict['plate_1']['offset']['x'] = 0.7
    __a_dict['plate_1']['offset']['y'] = 0.6
    __a_dict['plate_1']['offset']['z'] = 0.9

    __a_dict['camera'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['camera']['type'] = 'opentrons_96_tiprack_20ul'#'opentrons_96_tiprack_300ul'
    __a_dict['camera']['pos'] = 3
    __a_dict['camera']['tip'] = 'D3'
    __a_dict['camera']['offset']['x'] = 0.2
    __a_dict['camera']['offset']['y'] = 0.3
    __a_dict['camera']['offset']['z'] = 0

    __a_dict['reservoir_1'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['reservoir_1']['type'] = 'azenta_12_reservoir_2100ul'
    __a_dict['reservoir_1']['pos'] = 7
    __a_dict['reservoir_1']['offset']['x'] = 0
    __a_dict['reservoir_1']['offset']['y'] = 0
    __a_dict['reservoir_1']['offset']['z'] = 0


    __a_dict['reservoir_2'] = {'type':{},'pos' : {},'offset':{}}
    __a_dict['reservoir_2']['type'] = 'azenta_12_reservoir_2100ul'
    __a_dict['reservoir_2']['pos'] = 8
    __a_dict['reservoir_2']['offset']['x'] = 0
    __a_dict['reservoir_2']['offset']['y'] = 0
    __a_dict['reservoir_2']['offset']['z'] = 0

    __a_dict['pipette_1'] = {'type':{},'mount' : {}}
    __a_dict['pipette_1']['type'] = 'p300_multi_gen2'
    __a_dict['pipette_1']['mount']='left'

    __a_dict['pipette_2'] = {'type':{},'mount' : {}}
    __a_dict['pipette_2']['type'] = 'p20_single_gen2'
    __a_dict['pipette_2']['mount']='right'
    
          
    def __init__(self, 
                 a_dict = None,
                 path = os.path.join('..', '..', 'DriveSMB'),
                 #path = r'C:\Users\scrc112\Desktop\work\biplab\OpenTron\DriveSMB', 
                 file = 'ot2_system_config_h2o2.json' ):
        
        self.dict = (a_dict if a_dict else system_config.__a_dict)
        self.path = (path if path else system_config.__path)
        self.file = (file if file else system_config.__file)
        

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
        
        
    def __repr__(self):
        return 'class system_config.'+ \
                '\ntiprack 1 : {}'.format(self.dict['tiprack_1']) +'\n'+ \
                '*'*100+\
                '\ntiprack 2 : {}'.format(self.dict['tiprack_2']) +'\n'+\
                '*'*100+\
                '\nplate 1 : {}'.format(self.dict['plate_1']) +'\n'+\
                '*'*100+\
                '\nreservoir 1 : {}'.format(self.dict['reservoir_1']) +'\n'+\
                '*'*100+\
                '\nreservoir 2 : {}'.format(self.dict['reservoir_2']) +'\n'+\
                '*'*100+\
                '\ncamera : {}'.format(self.dict['camera']) +'\n'+\
                '*'*100+\
                '\npipette 1 : {}'.format(self.dict['pipette_1']) +'\n'+\
                '*'*100+\
                '\npipette 2 : {}'.format(self.dict['pipette_2']) +'\n'
        

class utility():
    
    """
    Utility class for handling well position extraction and transformations in a plate layout.

    This class provides static methods to:
    - Extract well positions row-wise or column-wise from given range strings.
    - Broadcast well positions for multi-channel pipettes.
    - Check and validate well indexing.
    - Repeat and replicate list structures where needed.
    """
    @staticmethod
    def get_wells_rowwise(st):
        """
        Extract well positions row-wise from a given range string and return a list of wells in row order.
        If a step size is provided (e.g., 'A1:C3:2'), it selects every nth well.
        """
        # traverse row wise    
        if st.count(':') == 0 : return st
        rows = [el for el in string.ascii_uppercase[:8]]
        cols = np.arange(1,12+1,1).tolist()
        wells = [el1+str(el2) for el1 in rows for el2 in cols]
        skip = 0
        if st.count(':') > 1 :
            st = st.split(':')
            skip = int(st[1])
            st = [st[0],st[-1]]
            utility.check_elements_sing(wells, st)
        else:
            st = st.split(':')
            utility.check_elements_sing(wells, st)
            
        out = wells[wells.index(st[0]) : wells.index(st[1])+1]
        if skip:
            skip = np.arange(0,len(out),skip)
            out = [out[el] for el in skip] 
        if not(out) : raise ValueError('check row/col indexing')
        return out


    @staticmethod    
    def get_wells(st):
        """
        Extract well positions from a range string (e.g., 'A1:C3') and return a list of well positions.
        If a step size is provided (e.g., 'A1:C3:2'), it selects every nth well.
        """
        if st.count(':') == 0 : return st
        rows = [el for el in string.ascii_uppercase[:8]]
        cols = np.arange(1,12+1,1).tolist()
        wells = [el1+str(el2) for el2 in cols for el1 in rows]
        skip = 0
        if st.count(':') > 1 : 
            st = st.split(':')
            skip = int(st[1])
            st = [st[0],st[-1]]
            utility.check_elements_sing(wells, st)
        else :
            st = st.split(':')
            utility.check_elements_sing(wells, st)
        out = wells[wells.index(st[0]) : wells.index(st[1])+1]
        if skip:
            skip = np.arange(0,len(out),skip)
            out = [out[el] for el in skip]   
        if not(out) : raise ValueError('check row/col indexing')   
        return out
        
 
    @staticmethod    
    def needRepmat(lst):
        """
        Identify which rows in a nested list need to be repeated for column-wise transfers.
        """
        track = False
        if not(isinstance(lst[0], list)) : 
            lst = [lst]
            track = True
        track_rows = []
        for idx, el in enumerate(lst):
            if (not(isinstance(el[0], list)) and isinstance(el[1], list)):
                track_rows.append(True)
            else:
                track_rows.append(False)
        return track_rows
        

    @staticmethod
    def repmat(lst):
        """
        Repeat elements in lists where necessary to ensure correct well alignment.
        """
        track_rows = utility.needRepmat(lst)
        track = False
        if not(isinstance(lst[0],list)) : 
            lst = [lst]
            track = True
        for idx1, el in enumerate(lst):
            if track_rows[idx1]:
                for idx2, el1 in enumerate(el):
                    if not(isinstance(el1, list)):
                        lst[idx1][idx2] = [el1]*len(el[-1])      
        if track : lst = lst[0]
        return lst
        

    @staticmethod
    def hasColon(arr):
        """
        Check whether a given input contains a colon (':').
        Used for identifying well range specifications.
        """
        st = ':'
        if isinstance(arr, list) :
            out = (True if len(arr) == 1 and (st in arr[0]) else False)
        else:
            out = (True if (st in arr) else False)
        return out
        
        
    @staticmethod    
    def col_broadcast(st = 'A1:A5'):
        """
        Expand a column range string (e.g., 'A1:A5') into individual well positions.
        """
        if st.count(':') == 1 :
            lst = st.split(':')
            utility.check_elements_mult(lst)
            out = [lst[0][0]+str(el) for el in np.arange(int(lst[0][1:]),int(lst[1][1:])+1,1)]
        elif st.count(':') == 2 :
            lst = st.split(':')
            utility.check_elements_mult([lst[0],lst[-1]])
            out = [lst[0][0]+str(el) for el in np.arange(int(lst[0][1:]),int(lst[2][1:])+1,int(lst[1]))]
        else:
            out = st
        return out
        

    @staticmethod
    def check_elements_mult(st):
        """
        Validate that a given well range is correctly formatted and within plate bounds.
        """
        end = 12 + 1
        base_str = [string.ascii_uppercase[0] + str(el) for el in range(1,end)]
        if (st[0] in base_str) and (st[1] in base_str):
            out = (True if base_str.index(st[0]) < base_str.index(st[1]) else False)
            if not(out) : raise ValueError('inverse indexing is not allowed')
        else :
            raise ValueError('overflow : keys out of range!')
            
            
    @staticmethod
    def check_elements_sing(wells, st):
        """
        Validate that the provided well range is in the correct order.
        """
        if (st[0] in wells) and (st[1] in wells):
            out = (True if wells.index(st[0]) < wells.index(st[1]) else False)
            if not(out) : raise ValueError('inverse indexing is not allowed')
        else :
            raise ValueError('overflow : keys out of range!')
            
        
    @staticmethod
    def get_elements(st,num = 4):
        """
        Get a list of well elements based on a specific number of rows.
        """
        st1 = st[1:]
        return [el + st1 for el in string.ascii_uppercase[:num]]
        

    @staticmethod
    def transfer4_broadcast(lst,num_wells):
        """
        Broadcast a list for transfer_4 titration handling.
        """
        lst[1] = utility.col_broadcast(lst[1][0])
        dummy = []
        for el1, el2 in zip(lst[0],lst[1]):
            dummy.append([el1, utility.get_elements(el2,num_wells)])
        return dummy


class transfer_config(ABC):
    """
    Class for liquid transfer configurations.
    """
    __smb_path = os.path.join('..', '..', 'DriveSMB')
    #__smb_path = r'C:\Users\scrc112\Desktop\work\biplab\OpenTron\DriveSMB'
    __smb_file = 'ot2_transfer_config_h2o2.json'

    # Liquid transfers dictionary
    __a_dict = {}
    
    # Define default transfer configurations
    __a_dict['transfer_1'] = {'vol': {}, 'locs': {}, 'tiprack': {}, 'speed': {}, 'mix': {}, 'blowout': {}}
    __a_dict['transfer_2'] = {'vol': {}, 'locs': {}, 'tiprack': {}, 'speed': {}, 'mix': {}, 'blowout': {}}
    __a_dict['transfer_3'] = {'vol': {}, 'locs': {}, 'tiprack': {}, 'speed': {}, 'mix': {}, 'blowout': {}}
    __a_dict['transfer_4'] = {'vol': {}, 'locs': {}, 'tiprack': {}, 'speed': {}, 'mix': {}, 'blowout': {}, 'titration': {}, 'num_wells': 0}

    def __init__(self, a_dict=None, path=None, file=None):
        self.dict = a_dict if a_dict else self.__a_dict
        self.path = path if path else self.__smb_path
        self.file = file if file else self.__smb_file
    

    def string_broadcast(self, st1):
        lst = ['A','B','C','D','E','F','G','H']
        dummy = st1.split(':')
        if dummy[1][1] != dummy[0][1]:
            raise Exception("Column indices do not match!")
        dummy = [dummy[0][0], dummy[1][0], dummy[1][1]]
        idx1 = lst.index(dummy[0])
        idx2 = lst.index(dummy[1]) + 1
        return [lst[i] + dummy[2] for i in range(idx1, idx2)]
    

    def string_array_broadcast(self, st):
        dummy_list = []
        for el in st:
            dummy = [[i, j] for i, j in zip(self.string_broadcast(el[0]), self.string_broadcast(el[1]))]
            dummy_list.append(dummy)
        self.dict['transfer_2']['locs'] = dummy_list
    

    @staticmethod
    def locs_broadcast(lst, fun):
        dummy = lst.copy()
        for idx1, transfer in enumerate(dummy):
            if isinstance(transfer, list):
                for idx2, el1 in enumerate(transfer):
                    if isinstance(el1, list):
                        for idx3, el2 in enumerate(el1):
                            dummy[idx1][idx2][idx3] = fun(el2)
                    else:
                        dummy[idx1][idx2] = fun(el1)
            else:
                dummy[idx1] = fun(transfer)
        return dummy
    

    @dispatch()
    def save_json(self):
        with open(os.path.join(self.path, self.file), 'w') as outfile:
            json.dump(self.dict, outfile)
    

    @dispatch(str, str)
    def save_json(self, path, file):
        if path and file:
            with open(os.path.join(path, file), 'w') as outfile:
                json.dump(self.dict, outfile)
        else:
            raise ValueError('Path or file missing')
    

    @dispatch(str)
    def save_json(self, flag):
        if flag in ('save_to_smb', 'smb'):
            with open(os.path.join(self.__smb_path, self.__smb_file), 'w') as outfile:
                json.dump(self.dict, outfile)
        else:
            raise ValueError('Invalid saving flag')
   
   
    @dispatch(str, str)
    def save_json(self, path, file):
        """Save JSON to a custom path and file."""
        self.__set_reservoirs()
        if path and file:
            with open(os.path.join(path, file), 'w') as outfile:
                json.dump(self.dict, outfile)
        else:
            raise ValueError('Path or file missing')

                    
    @dispatch(str)
    def save_json(self, flag):
        """Save JSON to SMB if flag is correct."""
        if flag in ('save_to_smb', 'smb'):
            with open(os.path.join(self.__smb_path, self.__smb_file), 'w') as outfile:
                json.dump(self.dict, outfile)
        else:
            raise ValueError('Invalid saving flag')


    def __set_reservoirs(self):
        """Set up tiprack."""
        if utility.hasColon(self.dict['transfer_1']['tiprack']) :
            self.dict['transfer_1']['tiprack'] = utility.col_broadcast(self.dict['transfer_1']['tiprack'][0]) 
        if utility.hasColon(self.dict['transfer_2']['tiprack']) :
            self.dict['transfer_2']['tiprack'] = utility.col_broadcast(self.dict['transfer_2']['tiprack'][0])
        if utility.hasColon(self.dict['transfer_3']['tiprack']) :
            self.dict['transfer_3']['tiprack'] = utility.col_broadcast(self.dict['transfer_3']['tiprack'][0])


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
                '*'*100


def safety_test(sc, tc):
    """A validation function that ensures that the liquid transfer parameters in tc (transfer configuration) 
    are compatible and safe based on the system configuration in sc (system configuration).
    """
    vol_transfer = tc.dict
    data = sc.dict
    
    # Ensure correct reservoir type
    assert data['reservoir_2']['type'] == 'azenta_12_reservoir_2100ul', 'reservoir2 type mismatch!'
    
    # Volume check
    vol1 = vol_transfer['transfer_1']['vol']
    vol2 = vol_transfer['transfer_2']['vol']
    vol3 = vol_transfer['transfer_3']['vol']
    
    if not isinstance(vol1, list):
        raise TypeError('transfer_1 vol must be a list')
    if isinstance(vol2, list):
        raise TypeError('transfer_2 vol must be a single value') 
    if isinstance(vol3, list):
        raise TypeError('transfer_3 vol must be a single value')
    
    for el in vol1:
        if (el + vol2 + vol3) > 300:
            raise ValueError('Total volume exceeds 300 µL')       
    
    print('Safety test: passed')

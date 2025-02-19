from utils.otsmb_aic import *
from opentrons import protocol_api
import opentrons.execute
import pandas as pd
import numpy as np
import time
import json
import os
from smb.smb_structs import OperationFailure
import sys
import time 


metadata = {
    'apiLevel': '2.13',
    'protocolName': 'H2O2 Titration',
    'description': '''This protocol is used for high-throughput H2O2 determination
                      
                   ''',
    'author': 'Biplab Dutta'
    }
    
def run(protocol: protocol_api.ProtocolContext):
    
    try : 
        
        ## parsing json
        print('parsing json')
        path = '/var/lib/jupyter/notebooks/config_files'
        
        with open(os.path.join(path,'ot2_system_config_h2o2.json')) as f:
            data = json.load(f)
        
        with open(os.path.join(path, 'ot2_transfer_config_h2o2.json')) as f:
            vol_transfer = json.load(f)

        connection = protocol.cnc     
        
    
        ##################################################### system config #################################################
        # Load a tiprack for 300uL tips
        tiprack1 = protocol.load_labware(data['tiprack_1']['type'], data['tiprack_1']['pos'])
        tiprack1.set_offset(x = data['tiprack_1']['offset']['x'] ,
                            y = data['tiprack_1']['offset']['y'], 
                            z = data['tiprack_1']['offset']['z'])
        
        tiprack2 = protocol.load_labware(data['tiprack_2']['type'], data['tiprack_2']['pos'])
        tiprack2.set_offset(x = data['tiprack_2']['offset']['x'] ,
                            y = data['tiprack_2']['offset']['y'], 
                            z = data['tiprack_2']['offset']['z'])
                            
        tiprack3 = protocol.load_labware(data['tiprack_3']['type'], data['tiprack_3']['pos'])
        tiprack3.set_offset(x = data['tiprack_3']['offset']['x'] ,
                            y = data['tiprack_3']['offset']['y'], 
                            z = data['tiprack_3']['offset']['z'])
        
        tiprack4 = protocol.load_labware(data['tiprack_4']['type'], data['tiprack_4']['pos'])
        tiprack4.set_offset(x = data['tiprack_4']['offset']['x'] ,
                            y = data['tiprack_4']['offset']['y'], 
                            z = data['tiprack_4']['offset']['z'])
        
        camera_rack = protocol.load_labware(data['camera']['type'], data['camera']['pos'])
        camera_rack.set_offset(x = data['camera']['offset']['x'],
                               y = data['camera']['offset']['y'], 
                               z = data['camera']['offset']['z'])

        # Load 12 column reservoir
        reservoir_1 = protocol.load_labware(data['reservoir_1']['type'], data['reservoir_1']['pos'])

        # Load 12 column reservoir
        reservoir_2 = protocol.load_labware(data['reservoir_2']['type'], data['reservoir_2']['pos'])
        
        # Load 12 column reservoir
        reservoir_3 = protocol.load_labware(data['reservoir_3']['type'], data['reservoir_3']['pos'])
        
        # load 2 colimn reservoir
        reservoir_h2o = protocol.load_labware('azenta_12_reservoir_2100ul', 4)
        
        # Load 96 well plate
        plate = protocol.load_labware(data['plate_1']['type'], data['plate_1']['pos'])
        plate.set_offset(x = data['plate_1']['offset']['x'] ,
                         y = data['plate_1']['offset']['y'], 
                         z = data['plate_1']['offset']['z'])

        ## pipettes
        # Load a P300 Multi GEN2 on the left mount
        left = protocol.load_instrument(data['pipette_1']['type'], data['pipette_1']['mount'],  
                                        tip_racks=[tiprack3,tiprack4])
                                        #tip_racks=[tiprack1,tiprack2,tiprack3,camera_rack])
        # Load a P300 Single GEN2 on the right mount
        right = protocol.load_instrument(data['pipette_2']['type'], data['pipette_2']['mount'], 
                                         tip_racks=[tiprack1,tiprack2,camera_rack])
        
        
        ######################################### vol transfer ##################################################
        
        ## Transfer 1 : H2SO4, Multi-Channel Pipette, Tiprack : 12 col : A12
        
        print('H2SO4 Transfer:\n')
        count = 0
        vol = vol_transfer['transfer_1']['vol']
        left.pick_up_tip(tiprack4[vol_transfer['transfer_1']['tiprack']])
        print('pick up tips {} from deck {}'.format(vol_transfer['transfer_1']['tiprack'],data['tiprack_4']['pos']))
        for el0, el1 in zip(vol_transfer['transfer_1']['locs'][0], 
                            vol_transfer['transfer_1']['locs'][1]):
            
            
            print('transferring {} micl from deck {} {} to deck {} {}'.format(vol, 
                                                                             data['reservoir_2']['pos'],
                                                                             el0,
                                                                             data['plate_1']['pos'],
                                                                             el1))
            
            # pipette channels are over the rest of the wells of column 1
            left.aspirate(vol, reservoir_2[el0], rate = vol_transfer['transfer_1']['speed']['aspirate'])
            left.dispense(vol, plate[el1], rate = vol_transfer['transfer_1']['speed']['dispense'])
            
            if vol_transfer['transfer_1']['mix'] : 
                print('mixing {} times vol {} ul'.format(vol_transfer['transfer_1']['mix']['rep'], 
                                                         vol_transfer['transfer_1']['mix']['vol']))
                left.mix(vol_transfer['transfer_1']['mix']['rep'], vol_transfer['transfer_1']['mix']['vol'])
            
            if vol_transfer['transfer_1']['blowout'] :
                print('blowout')
                left.blow_out()
            
            run_cnc(connection, 'dummy_1122.txt')
            print('*'*10)
        
        left.drop_tip() 
        print('H2SO4 transfer complete\n')
        print('#'*100)
        
         
        ## Transfer 2 : H2O2, Multi-Channel Pipette, Tiprack : 1
        
        print('H2O2 Transfer:\n')
        count = 0
        vol = vol_transfer['transfer_2']['vol']

        for tip, el0, el1 in zip(vol_transfer['transfer_2']['tiprack'],
                                 vol_transfer['transfer_2']['locs'][0], 
                                 vol_transfer['transfer_2']['locs'][1]):
            
            print('pick up tips {} from deck {}'.format(tip,data['tiprack_3']['pos']))
            left.pick_up_tip(tiprack3[tip])

            print('transferring {} micl from deck {} {} to deck {} {}'.format(vol, 
                                                                             data['reservoir_1']['pos'],
                                                                             el0,
                                                                             data['plate_1']['pos'],
                                                                             el1))
            
            # pipette channels are over the rest of the wells of column 1
            left.aspirate(vol, reservoir_1[el0], rate = vol_transfer['transfer_2']['speed']['aspirate'])
            left.dispense(vol, plate[el1], rate = vol_transfer['transfer_2']['speed']['dispense'])
            
            if vol_transfer['transfer_2']['mix'] : 
                print('mixing {} times vol {} ul'.format(vol_transfer['transfer_2']['mix']['rep'], 
                                                         vol_transfer['transfer_2']['mix']['vol']))
                left.mix(vol_transfer['transfer_2']['mix']['rep'], vol_transfer['transfer_2']['mix']['vol'])
            
            if vol_transfer['transfer_2']['blowout'] : 
                print('blowout')
                left.blow_out()
            
            run_cnc(connection, 'dummy_1122.txt')
        
            left.drop_tip() 
            count +=1
            print('*'*10)
                
        print('H2O2 transfer complete\n')
        print('#'*100)
         
        ## capture image for mask generation
        ## init   
        
        print('Blanc plate imaging:\n')
        samp_dict = {'id' : 0, 'mode' : 'init'}
        trigger_cam(right,camera_rack, data['camera']['tip'], connection, delay = 20, samp_dict = samp_dict) 
      
        
        ## Transfer 3 : Pre-Estimation using KMnO4, Single-Channel Pipette, Tiprack : 11, col : A11
        
        print('pre-estimation:\n')
        vol = vol_transfer['transfer_3']['vol']
     
        for tip, transfer_el in zip(vol_transfer['transfer_3']['tiprack'], vol_transfer['transfer_3']['locs']):
            right.pick_up_tip(tiprack2[tip])
            vol = vol_transfer['transfer_3']['vol'] #* 2 #len(transfer_el[1]) 
            
            el0 = transfer_el[0] 
            print('pick up tips from deck {} {}'.format(data['tiprack_2']['pos'], tip))
            
            #print('aspirating {} from reservoir deck {} {}.............................'.format(vol,
            #                                                                                    data['reservoir_3']['pos'],
            #                                                                                    el0
            #                                                                                    ))
            
            for el1 in transfer_el[1] :
                
                print('aspirating from {} {} dispensing {} uL to plate deck {} {}'.format(data['reservoir_2']['pos'],
                                                                                          el0,  
                                                                                          vol,
                                                                                          data['plate_1']['pos'],
                                                                                          el1
                                                                                          ))
                
                right.aspirate(vol, reservoir_3[el0], rate = vol_transfer['transfer_3']['speed']['aspirate'])
                right.dispense(vol, plate[el1], rate = vol_transfer['transfer_3']['speed']['dispense'])
                
                if vol_transfer['transfer_3']['mix']['rep'] : right.mix(vol_transfer['transfer_3']['mix']['rep'], 
                                                                        vol_transfer['transfer_3']['mix']['vol'])
                if vol_transfer['transfer_3']['blowout'] : right.blow_out()
                
                print('aspirating from {} {} dispensing {} uL to plate deck {} {}'.format(data['reservoir_2']['pos'],
                                                                                          el0,  
                                                                                          vol,
                                                                                          data['plate_1']['pos'],
                                                                                          el1
                                                                                          ))
                
                right.aspirate(vol, reservoir_3[el0], rate = vol_transfer['transfer_3']['speed']['aspirate'])
                right.dispense(vol, plate[el1], rate = vol_transfer['transfer_3']['speed']['dispense'])
                
                if vol_transfer['transfer_3']['mix']['rep'] : right.mix(vol_transfer['transfer_3']['mix']['rep'], 
                                                                        vol_transfer['transfer_3']['mix']['vol'])
                if vol_transfer['transfer_3']['blowout']: right.blow_out()
                
                
                                                                                          
                run_cnc(connection, 'dummy_1122.txt')
            
                print('*'*10)
            
            right.drop_tip()
            
        # mixing with multichannel pipette
        vol2 = 200
        mix_rep = 5
        left.pick_up_tip(tiprack4['A2'])
        aspirate_speed = 1
        dispense_speed = 1

        mix_col = ['A'+(el[1:]) for el in vol_transfer['transfer_3']['locs'][0][1]] #['A6','A7','A8','A9']
        print(mix_col)

        for well in mix_col :
            print('mixing with multichan')
            left.mix(mix_rep, vol2, plate[well])

            # first run aspirate water from A3 (deck 4) & dispense on A9 (deck 4)
            left.aspirate(vol2, reservoir_h2o['A3'], rate = aspirate_speed)
            left.dispense(vol2, reservoir_h2o['A9'], rate = dispense_speed)
            print('rinsing protocol1 : aspirating & dispensing deck 4, A3 to A9')
            left.blow_out()

            # mix at A3
            left.mix(3, vol2, reservoir_h2o['A3'])
            print('rinsing protocol2: aspirating & dispensing deck 4, A3')
            left.aspirate(vol2, reservoir_h2o['A3'], rate = aspirate_speed)
            left.dispense(vol2, reservoir_h2o['A3'], rate = dispense_speed)
            left.blow_out()

        left.drop_tip()
        
          
        samp_dict = {'id' : 0, 'mode' : 'pre-estimation'}
     
        trigger_cam(right,camera_rack, data['camera']['tip'], connection, delay = 20, samp_dict = samp_dict)     
        
        # receive input on reservoir
        reservoir_list = get_reservoir_list(connection)
        print('reservoir list {}'.format(reservoir_list))    
        # update transfer4 locs
        vol_transfer['transfer_4']['locs'] = update_transfer4_locs(vol_transfer['transfer_4']['locs'], reservoir_list)
        

        print('pre-estimation complete\n')
        print('#'*100)
       
      
        ## Transfer 4: Titration
        
        print('Titration :\n')
        
        count = 0
        vol = vol_transfer['transfer_4']['vol']
        
        # multichan pick up tips 
        left.pick_up_tip(tiprack4['A3'])
        
        for samp_id, transfer_el in enumerate(vol_transfer['transfer_4']['locs']):
            titration_count = 0
            # run camera here
            samp_dict = {'id' : samp_id + 1, 
                         'mode' : 'titration', 
                         'titration_count' : 0, 
                         'status' : 'run', 
                         'cols' : transfer_el[1]}
                         
            # zero-time : no KMnO4
            trigger_cam(right,
                        camera_rack,  
                        data['camera']['tip'], 
                        connection, delay = 20, 
                        samp_dict = samp_dict)
                
            
            for k in range(vol_transfer['transfer_4']['titration'][titration_count]):
                
                print('*'*100)
                print('sample {} titration {}'.format(samp_id+1,k+1))
                
                tip = vol_transfer['transfer_4']['tiprack'][count]
                if count < 96 :
                    right.pick_up_tip(tiprack1[tip])
                    print('pick up tips {} {}'.format(data['tiprack_1']['pos'],tip))
                else:
                    right.pick_up_tip(tiprack2[tip])
                    print('pick up tips {} {}'.format(data['tiprack_2']['pos'],tip))
                    
                
                right.aspirate(vol * len(transfer_el[1]), 
                               reservoir_3[transfer_el[0]],
                               rate = vol_transfer['transfer_4']['speed']['aspirate'])
                               
                run_cnc(connection, 'dummy_1122.txt')
                
                for el in transfer_el[1]:
                    

                    print('transferring {} micl from deck {} {} to deck {} {}'.format(vol, 
                                                                                     data['reservoir_3']['pos'],
                                                                                     transfer_el[0],
                                                                                     data['plate_1']['pos'],
                                                                                     el))
                    
                    
                    right.dispense(vol, 
                                   plate[el], 
                                   rate = vol_transfer['transfer_4']['speed']['dispense'])
                                                                                     
                    
                    run_cnc(connection, 'dummy_1122.txt')
                    
                for el in transfer_el[1]:
                    
                    if vol_transfer['transfer_4']['mix']['rep'] :
                        print('mixing {} times {} uL in plate deck {} {}'.format(vol_transfer['transfer_4']['mix']['rep'], 
                                                                         vol_transfer['transfer_4']['mix']['vol'],
                                                                         data['reservoir_3']['pos'],
                                                                         plate[el]
                                                                         ))
                        
                        right.mix(vol_transfer['transfer_4']['mix']['rep'], vol_transfer['transfer_4']['mix']['vol'], plate[el])
                        
                        
                        
                    if vol_transfer['transfer_4']['blowout'] : 
                        print('blowout')
                        right.blow_out() 
                        
                    run_cnc(connection, 'dummy_1122.txt')    
                    
                right.drop_tip() 
                
                
                ## mix it with multichan pipette 
                vol2 = 200
                mix_rep = 5
                
                aspirate_speed = 1
                dispense_speed = 1
                
                well = transfer_el[1][0] #only 1 col
                print(well)
                
                print('mixing with multichan')
                left.mix(mix_rep, vol2, plate[well])
                
                # 2-channel reservoir
                # first run aspirate water from A3 (deck 4) & dispense on A9 (deck 4)
                left.aspirate(vol2, reservoir_h2o['A3'], rate = aspirate_speed)
                left.dispense(vol2, reservoir_h2o['A9'], rate = dispense_speed)
                print('rinsing protocol1 : aspirating & dispensing deck 4, A3 to A9')
                left.blow_out()

                # mix at A3
                left.mix(mix_rep, vol2, reservoir_h2o['A3'])
                print('rinsing protocol2: aspirating & dispensing deck 4, A3')
                left.blow_out()
                
                # run camera here
                samp_dict = {'id' : samp_id + 1, 
                             'mode' : 'titration', 
                             'titration_count' : k+1, 
                             'status' : 'run', 
                             'cols' : transfer_el[1]}
                
                trigger_cam(right,
                            camera_rack,  
                            data['camera']['tip'], 
                            connection, delay = 20, 
                            samp_dict = samp_dict)
        
                
                count+=1
                titration_count +=1
        
        left.drop_tip() # drop left tips @@16122023
            
        print('titration complete\n')
        print('#'*100)
          
        ##########################################################################################################################
        # send stop signal
        samp_dict = {'id' : 0, 'mode' : 'exit'}
        trigger_cam(right,camera_rack, data['camera']['tip'], connection, delay = 1, samp_dict = samp_dict)
        
        
    except KeyboardInterrupt:
        print('interrupted')
        home_robot_remove_tip(protocol)
    
def home_robot():
    protocol = opentrons.execute.get_protocol_api('2.13')
    protocol.home()
    print('robot homed!')
    
    return protocol

def home_robot_remove_tip(protocol):
    
    if len(protocol.loaded_instruments):
            for key, pipette in protocol.loaded_instruments.items():
                if pipette.has_tip : 
                    print('tipattached')
                    pipette.drop_tip()
                #pipette.home_plunger()
    protocol.home()
    
    del protocol

def run_protocol(cnc, 
                fname = 'msi2pi.csv',
                remote_path = '/msi2pi.csv', 
                local_path = '/var/lib/jupyter/notebooks/msi2pi.csv'):
    
    # home robot and get protocol context
    protocol=home_robot()


    # parse file and append df to protcol context
    df=pd.read_csv('msi2pi.csv'); print(df.status); protocol.df = df

    # run protocol
    print('running protocol..............')
    run(protocol)

    # overwrite file and store to SMB server
    df.status = 'finished'
    df.to_csv(fname)
    print('file overwritten..............')
    time.sleep(4)
    smb_store_file(cnc,remote_path = remote_path, local_path=local_path)

    # delete protocol
    del protocol
    
    print('end protocol')
    
    
def run_cnc(connection, fname) :
    SERVICE_NAME = "DriveSMB"

    file_attr=None

    try:
        file_attr=connection.getAttributes(SERVICE_NAME,fname)
        if file_attr.short_name : print('SMB pinging succesful\n')
    except OperationFailure:
        print("Oops!  file does not exist")
        
def trigger_cam(right,camera_rack, tip, cnc, delay = 20, samp_dict = None):
    local_path = '/var/lib/jupyter/notebooks/h2o2'
    fname = 'msi2pi_h2o2.csv'
    # for camera view    
    right.pick_up_tip(camera_rack[tip])
   
    print('running trigger cam mode : {}'.format(samp_dict['mode']))
    
    if samp_dict['mode'] == 'titration' :
        print('trigger sent to camera')
       
        df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in samp_dict.items()]))
        df.replace(np.nan, 0, inplace=True)
        df.to_csv(os.path.join(local_path,fname),index=False)
        
    
    elif samp_dict['mode'] == 'pre-estimation' :
        # init camera
        print('trigger sent to camera')
        df=pd.DataFrame({'mode' :['pre-estimation'],'status' : ['run']}); 
        df.to_csv(os.path.join(local_path,fname),index=False); 
        
        
    elif samp_dict['mode'] == 'init':
        print('trigger sent to camera')
        df=pd.DataFrame({'mode' :['init'],'status' : ['run']}); 
        df.to_csv(os.path.join(local_path,fname),index=False); 
        
    elif samp_dict['mode'] == 'exit':
        print('termination trigger sent to camera')
        df=pd.DataFrame({'mode' :['exit'],'status' : ['stop']}); 
        df.to_csv(os.path.join(local_path,fname),index=False); 
        
        
    smb_store_file(cnc,
                   remote_path = fname, 
                   local_path = os.path.join(local_path,fname))


    print('sleep for few seconds!')
    time.sleep(delay)

    right.return_tip()
    
    
def get_reservoir_list(cnc):
    local_path = '/var/lib/jupyter/notebooks/config_files'
    reservoir_list_file = 'reservoir_list.csv'
    time.sleep(10)
    run_cnc(cnc, reservoir_list_file)
    
    smb_retrieve_file(cnc,
                      remote_path = reservoir_list_file, 
                      local_path = os.path.join(local_path,reservoir_list_file))
    
    
    df = pd.read_csv(os.path.join(local_path,reservoir_list_file))
    
    return df.loc[:,'reservoir_list'].tolist()



def update_transfer4_locs(transfer4_locs, reservoir_list):
    reservoir_dict = {'E' : 'A1', 'F' : 'A2', 'G' : 'A3', 'H' : 'A4'}

    try :
        reservoir_list = [reservoir_dict[el[0]] for el in reservoir_list] 

    except KeyError:
        print('E,F,G or H wells should be used for pre-estimation')
    
    for el in range(len(transfer4_locs)):
        transfer4_locs[el][0] = reservoir_list[el]
    
    return transfer4_locs


def run_opentron_h2o2(fname='msi2pi_h2o2.csv',
                      remote_system_config_file = 'ot2_system_config_h2o2.json',
                      remote_transfer_config_file  = 'ot2_transfer_config_h2o2.json',
                      local_path = '/var/lib/jupyter/notebooks/config_files',
                     ):

    # connect to SMB
    cnc=smb_connect()
    count=0

    try:  
        # system config
        smb_retrieve_file(cnc,
                          remote_path = remote_system_config_file, 
                          local_path = os.path.join(local_path,remote_system_config_file))

        # transfer_config
        smb_retrieve_file(cnc,
                          remote_path = remote_transfer_config_file, 
                          local_path = os.path.join(local_path, remote_transfer_config_file))

        protocol = home_robot()
        protocol.cnc = cnc
        run(protocol)
        del protocol

    except (AssertionError, KeyboardInterrupt, AttributeError, NameError) as err:
        print(err)
        print(err.args)
        del protocol
        #home_robot_remove_tip(protocol)
        cnc.close()
        sys.exit('Exiting!')

    cnc.close()

from opentrons import protocol_api

metadata = {
    'apiLevel': '2.13',
    'protocolName': 'Pre-estimation Test',
    'description': '''testing preestimation.''',
    'author': 'BD'
    }


def run(protocol: protocol_api.ProtocolContext):


    # params
    mixing_multichan = 5 # repeat mixing by multichan pipette
    mixing_singlechan = 2 # repeat mixing by single chan pipette
    vol = 20 # volume aspirated/dispense by single chan pipette
    vol2 = 200 # vol for mixing by multichan pipette
    aspirate_speed = 1 # aspirate speed
    dispense_speed = 1 # dispense speed

    tiprack2 = protocol.load_labware('opentrons_96_tiprack_20ul', 11) #tiprack 2
    tiprack4 = protocol.load_labware('opentrons_96_tiprack_300ul', 8) #tiprack 4
    
    reservoir_3 = protocol.load_labware('azenta_12_reservoir_2100ul', 7) # KMnO4
    reservoir_2 = protocol.load_labware('azenta_12_reservoir_2100ul', 6) # h2so4
    
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', 5) # plate
    
    left = protocol.load_instrument('p300_multi_gen2', 'left', tip_racks=[tiprack2, tiprack4])
    right = protocol.load_instrument('p20_single_gen2', 'right', tip_racks=[tiprack2, tiprack4])
    
    
    # tip locations
    tiplocations = ['E11','F11','G11','H11']
    
    # transfers
    #locs = [['A1', ['E6', 'E7', 'E8', 'E9']], 
    #        ['A2', ['F6', 'F7', 'F8', 'F9']], 
    #        ['A3', ['G6', 'G7', 'G8', 'G9']], 
    #        ['A4', ['H6', 'H7', 'H8', 'H9']]]
            
    locs = [['A1', ['E6', 'E7', 'E8']], 
            ['A2', ['F6', 'F7', 'F8']], 
            ['A3', ['G6', 'G7', 'G8']], 
            ['A4', ['H6', 'H7', 'H8']]]
    
    
    for tip, transfer_el in zip(tiplocations, locs):
    
        right.pick_up_tip(tiprack2[tip])
        
        el0 = transfer_el[0] 
        
        # 40 micl transfer
        for el1 in transfer_el[1]:
        
            right.aspirate(vol, reservoir_3[el0], rate = aspirate_speed)
            right.dispense(vol, plate[el1], rate = dispense_speed)
            right.mix(mixing_singlechan, vol)
            right.blow_out()
                
            right.aspirate(vol, reservoir_3[el0], rate = aspirate_speed)
            right.dispense(vol, plate[el1], rate = dispense_speed)
            right.mix(mixing_singlechan, vol)
            right.blow_out()
            
        right.drop_tip()


    
    # mixing with multichannel pipette
    
    left.pick_up_tip(tiprack4['A2'])
    
    #mix_col = ['A6','A7','A8','A9']
    mix_col = ['A6','A7','A8']
    
    for well in mix_col :
        left.mix(mixing_multichan, vol2, plate[well])
        
        # first run aspirate water from A2 (deck 6) & dispense on A12 (deck 6)
        left.aspirate(vol2, reservoir_2['A2'], rate = 2)#aspirate_speed)
        left.dispense(vol2, reservoir_2['A12'], rate = dispense_speed)
        
        # mix at A2
        left.mix(3, vol2, reservoir_2['A2'])
        left.blow_out()
        #left.aspirate(vol2, reservoir_2['A2'], rate = 2)#aspirate_speed)
        #left.dispense(vol2, reservoir_2['A2'], rate = dispense_speed)
        
    
    left.drop_tip()   
    
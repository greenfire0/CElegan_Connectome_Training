# GoPiGo Connectome
# Written by Timothy Busbice, Gabriel Garrett, Geoffrey Churchill (c) 2014, in Python 2.7
# The GoPiGo Connectome uses a postSynaptic dictionary based on the C Elegans Connectome Model
# This application can be ran on the Raspberry Pi GoPiGo robot with a Sonar that represents Nose Touch when activated
# To run standalone without a GoPiGo robot, simply comment out the sections with Start and End comments 

#TIME STATE EXPERIMENTAL OPTIMIZATION
#The previous version had a logic error whereby if more than one neuron fired into the same neuron in the next time state,
# it would overwrite the contribution from the previous neuron. Thus, only one neuron could fire into the same neuron at any given time state.
# This version also explicitly lists all left and right muscles, so that during the muscle checks for the motor control function, instead of 
# iterating through each neuron, we now iterate only through the relevant muscle neurons.

## Start Comment
#from gopigo import *
## End Comment
import time
import networkx as nx
import matplotlib.pyplot as plt
import copy
import numpy as np
# The postSynaptic dictionary contains the accumulated weighted values as the
# connectome is executed

global thisState
global nextState
thisState = 0 
nextState = 1

# The Threshold is the maximum sccumulated value that must be exceeded before
# the Neurite will fire
threshold = 30


# Used to remove from Axon firing since muscles cannot fire.
muscles = ['MVU', 'MVL', 'MDL', 'MVR', 'MDR']

muscleList = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']

mLeft = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']
mRight = ['MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']
# Used to accumulate muscle weighted values in body muscles 07-23 = worm locomotion
musDleft = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23']
musVleft = ['MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']
musDright = ['MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23']
musVright = ['MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']
all_neuron_names = [
    'ADAL', 'ADAR', 'ADEL', 'ADER', 'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AFDL', 'AFDR',
    'AIAL', 'AIAR', 'AIBL', 'AIBR', 'AIML', 'AIMR', 'AINL', 'AINR', 'AIYL', 'AIYR',
    'AIZL', 'AIZR', 'ALA', 'ALML', 'ALMR', 'ALNL', 'ALNR', 'AQR', 'AS1', 'AS10', 'AS11',
    'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'ASEL', 'ASER', 'ASGL', 'ASGR',
    'ASHL', 'ASHR', 'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR', 'AUAL', 'AUAR', 'AVAL',
    'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER', 'AVFL', 'AVFR', 'AVG', 'AVHL',
    'AVHR', 'AVJL', 'AVJR', 'AVKL', 'AVKR', 'AVL', 'AVM', 'AWAL', 'AWAR', 'AWBL', 'AWBR',
    'AWCL', 'AWCR', 'BAGL', 'BAGR', 'BDUL', 'BDUR', 'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR', 'DA1',
    'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9', 'DB1', 'DB2', 'DB3', 'DB4', 'DB5',
    'DB6', 'DB7', 'DD1', 'DD2', 'DD3', 'DD4', 'DD5', 'DD6', 'DVA', 'DVB', 'DVC', 'FLPL', 'FLPR',
    'HSNL', 'HSNR', 'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6', 'IL1DL', 'IL1DR', 'IL1L',
    'IL1R', 'IL1VL', 'IL1VR', 'IL2L', 'IL2R', 'IL2DL', 'IL2DR', 'IL2VL', 'IL2VR', 'LUAL', 'LUAR',
    'M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5',  'MCL', 'MCR', 'MDL01', 'MDL02', 'MDL03',
    'MDL04', 'MDL05', 'MDL06', 'MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14',
    'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MDL24', 'MDR01',
    'MDR02', 'MDR03', 'MDR04', 'MDR05', 'MDR06', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12',
    'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDR21', 'MDR22', 'MDR23',
    'MDR24', 'MI', 'MVL01', 'MVL02', 'MVL03', 'MVL04', 'MVL05', 'MVL06', 'MVL07', 'MVL08', 'MVL09',
    'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20',
    'MVL21', 'MVL22', 'MVL23', 'MVR01', 'MVR02', 'MVR03', 'MVR04', 'MVR05', 'MVR06', 'MVR07', 'MVR08',
    'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19',
    'MVR20', 'MVR21', 'MVR22', 'MVR23', 'MVR24', 'MVULVA', 'NSML', 'NSMR', 'OLLL', 'OLLR', 'OLQDL',
    'OLQDR', 'OLQVL', 'OLQVR', 'PDA', 'PDB', 'PDEL', 'PDER', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'PHCL',
    'PHCR', 'PLML', 'PLMR', 'PLNL', 'PLNR', 'PQR', 'PVCL', 'PVCR', 'PVDL', 'PVDR', 'PVM', 'PVNL', 'PVNR',
    'PVPL', 'PVPR', 'PVQL', 'PVQR', 'PVR', 'PVT', 'PVWL', 'PVWR', 'RIAL', 'RIAR', 'RIBL', 'RIBR', 'RICL',
    'RICR', 'RID', 'RIFL', 'RIFR', 'RIGL', 'RIGR', 'RIH', 'RIML', 'RIMR', 'RIPL', 'RIPR', 'RIR', 'RIS',
    'RIVL', 'RIVR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMED', 'RMEL', 'RMER', 'RMEV',
    'RMFL', 'RMFR', 'RMGL', 'RMGR', 'RMHL', 'RMHR', 'SAADL', 'SAADR', 'SAAVL', 'SAAVR', 'SABD', 'SABVL',
    'SABVR', 'SDQL', 'SDQR', 'SIADL', 'SIADR', 'SIAVL', 'SIAVR', 'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR',
    'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'URADL', 'URADR', 'URAVL',
    'URAVR', 'URBL', 'URBR', 'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL', 'URYVR', 'VA1', 'VA10', 'VA11',
    'VA12', 'VA2', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7', 'VA8', 'VA9', 'VB1', 'VB10', 'VB11', 'VB2', 'VB3',
    'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VC1', 'VC2', 'VC3', 'VC4', 'VC5', 'VC6', 'VD1', 'VD10',
    'VD11', 'VD12', 'VD13', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9'
]

# This is the full C Elegans Connectome as expresed in the form of the Presynatptic
# neurite and the postSynaptic neurites
# postSynaptic['ADAR'][nextState] = (2 + postSynaptic['ADAR'][thisState])
# arr=postSynaptic['AIBR'] potential optimization

from weight_dict import dict
    # Add other neuron combinations as needed
combined_weights= dict


class wormConnectone:
    def __init__(self, weight_m):
        self.weight_matrix = weight_m
        self.fire=0
        self.spike_count=2
        self.postSynaptic = {}
        self.createpostSynaptic()
        



    def apply(self,neuron_name):
            
            if neuron_name != "BIAS":
                for neuron, weight in combined_weights[neuron_name].items():
                        #print(neuron, weight,neuron_name)
                        self.postSynaptic[neuron][nextState] += weight
            else:
                for neuron, weight in zip(all_neuron_names, self.weight_matrix):
                        self.postSynaptic[neuron][nextState] += weight

    def motorcontrol(self):
        self.accumright =0
        self.accumleft =0
        # accumulate left and right muscles and the accumulated values are
        # used to move the left and right motors of the robot
        for muscle in muscleList:
            if muscle in mLeft:
                self.accumleft += self.postSynaptic[muscle][nextState]
                # postSynaptic[muscle][thisState] = 0
                self.postSynaptic[muscle][nextState] = 0
            elif muscle in mRight:
                self.accumright += self.postSynaptic[muscle][nextState]
                self.postSynaptic[muscle][nextState] = 0

        new_speed = abs(self.accumleft) + abs(self.accumright)
        return [self.accumleft,self.accumright,min(max(new_speed,75),150)
]


    def dendriteAccumulate(self, dneuron):
        self.apply(dneuron)


    def fireNeuron(self,fneuron):
        global fire
        # The threshold has been exceeded and we fire the neurite
        if fneuron != "MVULVA":
            self.apply(fneuron)
            # postSynaptic[fneuron][nextState] = 0
            # postSynaptic[fneuron][thisState] = 0
            self.fire+=1
            self.postSynaptic[fneuron][nextState] = 0
        


    def runconnectome(self):
        # Each time a set of neuron is stimulated, this method will execute
        # The weighted values are accumulated in the postSynaptic array
        # Once the accumulation is read, we see what neurons are greater
        # than the threshold and fire the neuron or muscle that has exceeded
        # the threshold 
        global thisState
        global nextState
        
        for ps in self.postSynaptic:
            if ps[:3] not in muscles and abs(self.postSynaptic[ps][thisState]) > threshold:
                self.fireNeuron(ps)
                # print(ps)
                # print(ps)
                # postSynaptic[ps][nextState] = 0
        movement = self.motorcontrol()
        for ps in self.postSynaptic:
            # if postSynaptic[ps][thisState] != 0:
            # print(ps)
            # print("Before Clone: ", postSynaptic[ps][thisState])
            self.postSynaptic[ps][thisState] = copy.deepcopy(self.postSynaptic[ps][nextState])  # fired neurons keep getting reset to previous weight
            # print("After Clone: ", postSynaptic[ps][thisState])
        thisState, nextState = nextState, thisState
        
        return movement



    def move(self, dist,sees_food,interval):
                            if dist > 0 and dist < 100:
                                    self.dendriteAccumulate("FLPR")
                                    self.dendriteAccumulate("FLPL")
                                    self.dendriteAccumulate("ASHL")
                                    self.dendriteAccumulate("ASHR")
                                    self.dendriteAccumulate("IL1VL")
                                    self.dendriteAccumulate("IL1VR")
                                    self.dendriteAccumulate("OLQDL")
                                    self.dendriteAccumulate("OLQDR")
                                    self.dendriteAccumulate("OLQVR")
                                    self.dendriteAccumulate("OLQVL")
                                    return (self.runconnectome())
                                    
                            else:
                                    if interval == -10:
                                          for _ in range (self.spike_count): ##cheaty work around?
                                            self.dendriteAccumulate("BIAS")
                                            self.runconnectome()
                            
                                          
                                    elif sees_food:
                                            self.dendriteAccumulate("ADFL")
                                            self.dendriteAccumulate("ADFR")
                                            self.dendriteAccumulate("ASGR")
                                            self.dendriteAccumulate("ASGL")
                                            self.dendriteAccumulate("ASIL")
                                            self.dendriteAccumulate("ASIR")
                                            self.dendriteAccumulate("ASJR")
                                            self.dendriteAccumulate("ASJL")
                                            return self.runconnectome()
                                            #tfood += 0.5
                                            #if tfood > 20:
                                            #        tfood = 0

                                    else: return self.runconnectome()
    def createpostSynaptic(self):
        # The postSynaptic dictionary maintains the accumulated values for
        # each neuron and muscle. The Accumulated values are initialized to Zero
        for muscle in muscleList:
            self.postSynaptic[muscle] = [0,0]
        for neuron in all_neuron_names:
            self.postSynaptic[neuron] = [0,0]
        self.postSynaptic['BIAS']=[0,0]
        
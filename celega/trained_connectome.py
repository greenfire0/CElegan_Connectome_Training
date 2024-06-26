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

# The postSynaptic dictionary contains the accumulated weighted values as the
# connectome is executed
postSynaptic = {}

global thisState
global nextState
thisState = 0 
nextState = 1

# The Threshold is the maximum sccumulated value that must be exceeded before
# the Neurite will fire
threshold = 30

# Accumulators are used to decide the value to send to the Left and Right motors
# of the GoPiGo robot
accumleft = 0
accumright = 0

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

# This is the full C Elegans Connectome as expresed in the form of the Presynatptic
# neurite and the postSynaptic neurites
# postSynaptic['ADAR'][nextState] = (2 + postSynaptic['ADAR'][thisState])
# arr=postSynaptic['AIBR'] potential optimization

from weight_dict import dict
    # Add other neuron combinations as needed

combined_weights= dict

def apply(neuron_name):
        for neuron, weight in combined_weights[neuron_name].items():
                #print(neuron, weight,neuron_name)
                postSynaptic[neuron][nextState] += weight

def ADAL():
    apply('ADAL')

def ADAR():
    apply('ADAR')

def ADEL():
    apply('ADEL')

def ADER():
    apply('ADER')

def ADFL():
    apply('ADFL')

def ADFR():
    apply('ADFR')

def ADLL():
    apply('ADLL')

def ADLR():
    apply('ADLR')

def AFDL():
    apply('AFDL')

def AFDR():
    apply('AFDR')

def AIAL():
    apply('AIAL')

def AIAR():
    apply('AIAR')

def AIBL():
    apply('AIBL')

def AIBR():
    apply('AIBR')

def AIML():
    apply('AIML')

def AIMR():
    apply('AIMR')

def AINL():
    apply('AINL')

def AINR():
    apply('AINR')

def AIYL():
    apply('AIYL')

def AIYR():
    apply('AIYR')

def AIZL():
    apply('AIZL')

def AIZR():
    apply('AIZR')

def ALA():
    apply('ALA')

def ALML():
    apply('ALML')

def ALMR():
    apply('ALMR')

def ALNL():
    apply('ALNL')

def ALNR():
    apply('ALNR')

def AQR():
    apply('AQR')

def AS1():
    apply('AS1')

def AS2():
    apply('AS2')

def AS3():
    apply('AS3')

def AS4():
    apply('AS4')

def AS5():
    apply('AS5')

def AS6():
    apply('AS6')

def AS7():
    apply('AS7')

def AS8():
    apply('AS8')

def AS9():
    apply('AS9')

def AS10():
    apply('AS10')

def AS11():
    apply('AS11')

def ASEL():
    apply('ASEL')

def ASER():
    apply('ASER')

def ASGL():
    apply('ASGL')

def ASGR():
    apply('ASGR')

def ASHL():
    apply('ASHL')

def ASHR():
    apply('ASHR')

def ASIL():
    apply('ASIL')

def ASIR():
    apply('ASIR')

def ASJL():
    apply('ASJL')

def ASJR():
    apply('ASJR')

def ASKL():
    apply('ASKL')

def ASKR():
    apply('ASKR')

def AUAL():
    apply('AUAL')

def AUAR():
    apply('AUAR')

def AVAL():
    apply('AVAL')

def AVAR():
    apply('AVAR')

def AVBL():
    apply('AVBL')

def AVBR():
    apply('AVBR')

def AVDL():
    apply('AVDL')

def AVDR():
    apply('AVDR')

def AVEL():
    apply('AVEL')

def AVER():
    apply('AVER')

def AVFL():
    apply('AVFL')

def AVFR():
    apply('AVFR')

def AVG():
    apply('AVG')

def AVHL():
    apply('AVHL')

def AVHR():
    apply('AVHR')

def AVJL():
    apply('AVJL')

def AVJR():
    apply('AVJR')

def AVKL():
    apply('AVKL')

def AVKR():
    apply('AVKR')

def AVL():
    apply('AVL')

def AVM():
    apply('AVM')

def AWAL():
    apply('AWAL')

def AWAR():
    apply('AWAR')

def AWBL():
    apply('AWBL')

def AWBR():
    apply('AWBR')

def AWCL():
    apply('AWCL')

def AWCR():
    apply('AWCR')

def BAGL():
    apply('BAGL')

def BAGR():
    apply('BAGR')

def BDUL():
    apply('BDUL')

def BDUR():
    apply('BDUR')

def CEPDL():
    apply('CEPDL')

def CEPDR():
    apply('CEPDR')

def CEPVL():
    apply('CEPVL')

def CEPVR():
    apply('CEPVR')

def DA1():
    apply('DA1')

def DA2():
    apply('DA2')

def DA3():
    apply('DA3')

def DA4():
    apply('DA4')

def DA5():
    apply('DA5')

def DA6():
    apply('DA6')

def DA7():
    apply('DA7')

def DA8():
    apply('DA8')

def DA9():
    apply('DA9')

def DB1():
    apply('DB1')

def DB2():
    apply('DB2')

def DB3():
    apply('DB3')

def DB4():
    apply('DB4')

def DB5():
    apply('DB5')

def DB6():
    apply('DB6')

def DB7():
    apply('DB7')

def DD1():
    apply('DD1')

def DD2():
    apply('DD2')

def DD3():
    apply('DD3')

def DD4():
    apply('DD4')

def DD5():
    apply('DD5')

def DD6():
    apply('DD6')

def DVA():
    apply('DVA')

def DVB():
    apply('DVB')

def DVC():
    apply('DVC')

def FLPL():
    apply('FLPL')

def FLPR():
    apply('FLPR')

def HSNL():
    apply('HSNL')

def HSNR():
    apply('HSNR')

def I1L():
    apply('I1L')

def I1R():
    apply('I1R')

def I2L():
    apply('I2L')

def I2R():
    apply('I2R')

def I3():
    apply('I3')

def I4():
    apply('I4')

def I5():
    apply('I5')

def I6():
    apply('I6')

def IL1DL():
    apply('IL1DL')

def IL1DR():
    apply('IL1DR')

def IL1L():
    apply('IL1L')

def IL1R():
    apply('IL1R')

def IL1VL():
    apply('IL1VL')

def IL1VR():
    apply('IL1VR')

def IL2DL():
    apply('IL2DL')

def IL2DR():
    apply('IL2DR')

def IL2L():
    apply('IL2L')

def IL2R():
    apply('IL2R')

def IL2VL():
    apply('IL2VL')

def IL2VR():
    apply('IL2VR')

def LUAL():
    apply('LUAL')

def LUAR():
    apply('LUAR')

def M1():
    apply('M1')

def M2L():
    apply('M2L')

def M2R():
    apply('M2R')

def M3L():
    apply('M3L')

def M3R():
    apply('M3R')

def M4():
    apply('M4')

def M5():
    apply('M5')

def MCL():
    apply('MCL')

def MCR():
    apply('MCR')

def MI():
    apply('MI')

def NSML():
    apply('NSML')

def NSMR():
    apply('NSMR')

def OLLL():
    apply('OLLL')

def OLLR():
    apply('OLLR')

def OLQDL():
    apply('OLQDL')

def OLQDR():
    apply('OLQDR')

def OLQVL():
    apply('OLQVL')

def OLQVR():
    apply('OLQVR')

def PDA():
    apply('PDA')

def PDB():
    apply('PDB')

def PDEL():
    apply('PDEL')

def PDER():
    apply('PDER')

def PHAL():
    apply('PHAL')

def PHAR():
    apply('PHAR')

def PHBL():
    apply('PHBL')

def PHBR():
    apply('PHBR')

def PHCL():
    apply('PHCL')

def PHCR():
    apply('PHCR')

def PLML():
    apply('PLML')

def PLMR():
    apply('PLMR')

def PLNL():
    apply('PLNL')

def PLNR():
    apply('PLNR')

def PQR():
    apply('PQR')

def PVCL():
    apply('PVCL')

def PVCR():
    apply('PVCR')

def PVDL():
    apply('PVDL')

def PVDR():
    apply('PVDR')

def PVM():
    apply('PVM')

def PVNL():
    apply('PVNL')

def PVNR():
    apply('PVNR')

def PVPL():
    apply('PVPL')

def PVPR():
    apply('PVPR')

def PVQL():
    apply('PVQL')

def PVQR():
    apply('PVQR')

def PVR():
    apply('PVR')

def PVT():
    apply('PVT')

def PVWL():
    apply('PVWL')

def PVWR():
    apply('PVWR')

def RIAL():
    apply('RIAL')

def RIAR():
    apply('RIAR')

def RIBL():
    apply('RIBL')

def RIBR():
    apply('RIBR')

def RICL():
    apply('RICL')

def RICR():
    apply('RICR')

def RID():
    apply('RID')

def RIFL():
    apply('RIFL')

def RIFR():
    apply('RIFR')

def RIGL():
    apply('RIGL')

def RIGR():
    apply('RIGR')

def RIH():
    apply('RIH')

def RIML():
    apply('RIML')

def RIMR():
    apply('RIMR')

def RIPL():
    apply('RIPL')

def RIPR():
    apply('RIPR')

def RIR():
    apply('RIR')

def RIS():
    apply('RIS')

def RIVL():
    apply('RIVL')

def RIVR():
    apply('RIVR')

def RMDDL():
    apply('RMDDL')

def RMDDR():
    apply('RMDDR')

def RMDL():
    apply('RMDL')

def RMDR():
    apply('RMDR')

def RMDVL():
    apply('RMDVL')

def RMDVR():
    apply('RMDVR')

def RMED():
    apply('RMED')

def RMEL():
    apply('RMEL')

def RMER():
    apply('RMER')

def RMEV():
    apply('RMEV')

def RMFL():
    apply('RMFL')

def RMFR():
    apply('RMFR')

def RMGL():
    apply('RMGL')

def RMGR():
    apply('RMGR')

def RMHL():
    apply('RMHL')

def RMHR():
    apply('RMHR')

def SAADL():
    apply('SAADL')

def SAADR():
    apply('SAADR')

def SAAVL():
    apply('SAAVL')

def SAAVR():
    apply('SAAVR')

def SABD():
    apply('SABD')

def SABVL():
    apply('SABVL')

def SABVR():
    apply('SABVR')

def SDQL():
    apply('SDQL')

def SDQR():
    apply('SDQR')

def SIADL():
    apply('SIADL')

def SIADR():
    apply('SIADR')

def SIAVL():
    apply('SIAVL')

def SIAVR():
    apply('SIAVR')

def SIBDL():
    apply('SIBDL')

def SIBDR():
    apply('SIBDR')

def SIBVL():
    apply('SIBVL')

def SIBVR():
    apply('SIBVR')

def SMBDL():
    apply('SMBDL')

def SMBDR():
    apply('SMBDR')

def SMBVL():
    apply('SMBVL')

def SMBVR():
    apply('SMBVR')

def SMDDL():
    apply('SMDDL')

def SMDDR():
    apply('SMDDR')

def SMDVL():
    apply('SMDVL')

def SMDVR():
    apply('SMDVR')

def URADL():
    apply('URADL')

def URADR():
    apply('URADR')

def URAVL():
    apply('URAVL')

def URAVR():
    apply('URAVR')

def URBL():
    apply('URBL')

def URBR():
    apply('URBR')

def URXL():
    apply('URXL')

def URXR():
    apply('URXR')

def URYDL():
    apply('URYDL')

def URYDR():
    apply('URYDR')

def URYVL():
    apply('URYVL')

def URYVR():
    apply('URYVR')

def VA1():
    apply('VA1')

def VA2():
    apply('VA2')

def VA3():
    apply('VA3')

def VA4():
    apply('VA4')

def VA5():
    apply('VA5')

def VA6():
    apply('VA6')

def VA7():
    apply('VA7')

def VA8():
    apply('VA8')

def VA9():
    apply('VA9')

def VA10():
    apply('VA10')

def VA11():
    apply('VA11')

def VA12():
    apply('VA12')

def VB1():
    apply('VB1')

def VB2():
    apply('VB2')

def VB3():
    apply('VB3')

def VB4():
    apply('VB4')

def VB5():
    apply('VB5')

def VB6():
    apply('VB6')

def VB7():
    apply('VB7')

def VB8():
    apply('VB8')

def VB9():
    apply('VB9')

def VB10():
    apply('VB10')

def VB11():
    apply('VB11')

def VC1():
    apply('VC1')

def VC2():
    apply('VC2')

def VC3():
    apply('VC3')

def VC4():
    apply('VC4')

def VC5():
    apply('VC5')

def VC6():
    apply('VC6')

def VD1():
    apply('VD1')

def VD2():
    apply('VD2')

def VD3():
    apply('VD3')

def VD4():
    apply('VD4')

def VD5():
    apply('VD5')

def VD6():
    apply('VD6')

def VD7():
    apply('VD7')

def VD8():
    apply('VD8')

def VD9():
    apply('VD9')

def VD10():
    apply('VD10')

def VD11():
    apply('VD11')

def VD12():
    apply('VD12')

def VD13():
    apply('VD13')
        
        
def createpostSynaptic():
        # The postSynaptic dictionary maintains the accumulated values for
        # each neuron and muscle. The Accumulated values are initialized to Zero
        postSynaptic['ADAL'] = [0,0]
        postSynaptic['ADAR'] = [0,0]
        postSynaptic['ADEL'] = [0,0]
        postSynaptic['ADER'] = [0,0]
        postSynaptic['ADFL'] = [0,0]
        postSynaptic['ADFR'] = [0,0]
        postSynaptic['ADLL'] = [0,0]
        postSynaptic['ADLR'] = [0,0]
        postSynaptic['AFDL'] = [0,0]
        postSynaptic['AFDR'] = [0,0]
        postSynaptic['AIAL'] = [0,0]
        postSynaptic['AIAR'] = [0,0]
        postSynaptic['AIBL'] = [0,0]
        postSynaptic['AIBR'] = [0,0]
        postSynaptic['AIML'] = [0,0]
        postSynaptic['AIMR'] = [0,0]
        postSynaptic['AINL'] = [0,0]
        postSynaptic['AINR'] = [0,0]
        postSynaptic['AIYL'] = [0,0]
        postSynaptic['AIYR'] = [0,0]
        postSynaptic['AIZL'] = [0,0]
        postSynaptic['AIZR'] = [0,0]
        postSynaptic['ALA'] = [0,0]
        postSynaptic['ALML'] = [0,0]
        postSynaptic['ALMR'] = [0,0]
        postSynaptic['ALNL'] = [0,0]
        postSynaptic['ALNR'] = [0,0]
        postSynaptic['AQR'] = [0,0]
        postSynaptic['AS1'] = [0,0]
        postSynaptic['AS10'] = [0,0]
        postSynaptic['AS11'] = [0,0]
        postSynaptic['AS2'] = [0,0]
        postSynaptic['AS3'] = [0,0]
        postSynaptic['AS4'] = [0,0]
        postSynaptic['AS5'] = [0,0]
        postSynaptic['AS6'] = [0,0]
        postSynaptic['AS7'] = [0,0]
        postSynaptic['AS8'] = [0,0]
        postSynaptic['AS9'] = [0,0]
        postSynaptic['ASEL'] = [0,0]
        postSynaptic['ASER'] = [0,0]
        postSynaptic['ASGL'] = [0,0]
        postSynaptic['ASGR'] = [0,0]
        postSynaptic['ASHL'] = [0,0]
        postSynaptic['ASHR'] = [0,0]
        postSynaptic['ASIL'] = [0,0]
        postSynaptic['ASIR'] = [0,0]
        postSynaptic['ASJL'] = [0,0]
        postSynaptic['ASJR'] = [0,0]
        postSynaptic['ASKL'] = [0,0]
        postSynaptic['ASKR'] = [0,0]
        postSynaptic['AUAL'] = [0,0]
        postSynaptic['AUAR'] = [0,0]
        postSynaptic['AVAL'] = [0,0]
        postSynaptic['AVAR'] = [0,0]
        postSynaptic['AVBL'] = [0,0]
        postSynaptic['AVBR'] = [0,0]
        postSynaptic['AVDL'] = [0,0]
        postSynaptic['AVDR'] = [0,0]
        postSynaptic['AVEL'] = [0,0]
        postSynaptic['AVER'] = [0,0]
        postSynaptic['AVFL'] = [0,0]
        postSynaptic['AVFR'] = [0,0]
        postSynaptic['AVG'] = [0,0]
        postSynaptic['AVHL'] = [0,0]
        postSynaptic['AVHR'] = [0,0]
        postSynaptic['AVJL'] = [0,0]
        postSynaptic['AVJR'] = [0,0]
        postSynaptic['AVKL'] = [0,0]
        postSynaptic['AVKR'] = [0,0]
        postSynaptic['AVL'] = [0,0]
        postSynaptic['AVM'] = [0,0]
        postSynaptic['AWAL'] = [0,0]
        postSynaptic['AWAR'] = [0,0]
        postSynaptic['AWBL'] = [0,0]
        postSynaptic['AWBR'] = [0,0]
        postSynaptic['AWCL'] = [0,0]
        postSynaptic['AWCR'] = [0,0]
        postSynaptic['BAGL'] = [0,0]
        postSynaptic['BAGR'] = [0,0]
        postSynaptic['BDUL'] = [0,0]
        postSynaptic['BDUR'] = [0,0]
        postSynaptic['CEPDL'] = [0,0]
        postSynaptic['CEPDR'] = [0,0]
        postSynaptic['CEPVL'] = [0,0]
        postSynaptic['CEPVR'] = [0,0]
        postSynaptic['DA1'] = [0,0]
        postSynaptic['DA2'] = [0,0]
        postSynaptic['DA3'] = [0,0]
        postSynaptic['DA4'] = [0,0]
        postSynaptic['DA5'] = [0,0]
        postSynaptic['DA6'] = [0,0]
        postSynaptic['DA7'] = [0,0]
        postSynaptic['DA8'] = [0,0]
        postSynaptic['DA9'] = [0,0]
        postSynaptic['DB1'] = [0,0]
        postSynaptic['DB2'] = [0,0]
        postSynaptic['DB3'] = [0,0]
        postSynaptic['DB4'] = [0,0]
        postSynaptic['DB5'] = [0,0]
        postSynaptic['DB6'] = [0,0]
        postSynaptic['DB7'] = [0,0]
        postSynaptic['DD1'] = [0,0]
        postSynaptic['DD2'] = [0,0]
        postSynaptic['DD3'] = [0,0]
        postSynaptic['DD4'] = [0,0]
        postSynaptic['DD5'] = [0,0]
        postSynaptic['DD6'] = [0,0]
        postSynaptic['DVA'] = [0,0]
        postSynaptic['DVB'] = [0,0]
        postSynaptic['DVC'] = [0,0]
        postSynaptic['FLPL'] = [0,0]
        postSynaptic['FLPR'] = [0,0]
        postSynaptic['HSNL'] = [0,0]
        postSynaptic['HSNR'] = [0,0]
        postSynaptic['I1L'] = [0,0]
        postSynaptic['I1R'] = [0,0]
        postSynaptic['I2L'] = [0,0]
        postSynaptic['I2R'] = [0,0]
        postSynaptic['I3'] = [0,0]
        postSynaptic['I4'] = [0,0]
        postSynaptic['I5'] = [0,0]
        postSynaptic['I6'] = [0,0]
        postSynaptic['IL1DL'] = [0,0]
        postSynaptic['IL1DR'] = [0,0]
        postSynaptic['IL1L'] = [0,0]
        postSynaptic['IL1R'] = [0,0]
        postSynaptic['IL1VL'] = [0,0]
        postSynaptic['IL1VR'] = [0,0]
        postSynaptic['IL2L'] = [0,0]
        postSynaptic['IL2R'] = [0,0]
        postSynaptic['IL2DL'] = [0,0]
        postSynaptic['IL2DR'] = [0,0]
        postSynaptic['IL2VL'] = [0,0]
        postSynaptic['IL2VR'] = [0,0]
        postSynaptic['LUAL'] = [0,0]
        postSynaptic['LUAR'] = [0,0]
        postSynaptic['M1'] = [0,0]
        postSynaptic['M2L'] = [0,0]
        postSynaptic['M2R'] = [0,0]
        postSynaptic['M3L'] = [0,0]
        postSynaptic['M3R'] = [0,0]
        postSynaptic['M4'] = [0,0]
        postSynaptic['M5'] = [0,0]
        postSynaptic['MANAL'] = [0,0]
        postSynaptic['MCL'] = [0,0]
        postSynaptic['MCR'] = [0,0]
        postSynaptic['MDL01'] = [0,0]
        postSynaptic['MDL02'] = [0,0]
        postSynaptic['MDL03'] = [0,0]
        postSynaptic['MDL04'] = [0,0]
        postSynaptic['MDL05'] = [0,0]
        postSynaptic['MDL06'] = [0,0]
        postSynaptic['MDL07'] = [0,0]
        postSynaptic['MDL08'] = [0,0]
        postSynaptic['MDL09'] = [0,0]
        postSynaptic['MDL10'] = [0,0]
        postSynaptic['MDL11'] = [0,0]
        postSynaptic['MDL12'] = [0,0]
        postSynaptic['MDL13'] = [0,0]
        postSynaptic['MDL14'] = [0,0]
        postSynaptic['MDL15'] = [0,0]
        postSynaptic['MDL16'] = [0,0]
        postSynaptic['MDL17'] = [0,0]
        postSynaptic['MDL18'] = [0,0]
        postSynaptic['MDL19'] = [0,0]
        postSynaptic['MDL20'] = [0,0]
        postSynaptic['MDL21'] = [0,0]
        postSynaptic['MDL22'] = [0,0]
        postSynaptic['MDL23'] = [0,0]
        postSynaptic['MDL24'] = [0,0]
        postSynaptic['MDR01'] = [0,0]
        postSynaptic['MDR02'] = [0,0]
        postSynaptic['MDR03'] = [0,0]
        postSynaptic['MDR04'] = [0,0]
        postSynaptic['MDR05'] = [0,0]
        postSynaptic['MDR06'] = [0,0]
        postSynaptic['MDR07'] = [0,0]
        postSynaptic['MDR08'] = [0,0]
        postSynaptic['MDR09'] = [0,0]
        postSynaptic['MDR10'] = [0,0]
        postSynaptic['MDR11'] = [0,0]
        postSynaptic['MDR12'] = [0,0]
        postSynaptic['MDR13'] = [0,0]
        postSynaptic['MDR14'] = [0,0]
        postSynaptic['MDR15'] = [0,0]
        postSynaptic['MDR16'] = [0,0]
        postSynaptic['MDR17'] = [0,0]
        postSynaptic['MDR18'] = [0,0]
        postSynaptic['MDR19'] = [0,0]
        postSynaptic['MDR20'] = [0,0]
        postSynaptic['MDR21'] = [0,0]
        postSynaptic['MDR22'] = [0,0]
        postSynaptic['MDR23'] = [0,0]
        postSynaptic['MDR24'] = [0,0]
        postSynaptic['MI'] = [0,0]
        postSynaptic['MVL01'] = [0,0]
        postSynaptic['MVL02'] = [0,0]
        postSynaptic['MVL03'] = [0,0]
        postSynaptic['MVL04'] = [0,0]
        postSynaptic['MVL05'] = [0,0]
        postSynaptic['MVL06'] = [0,0]
        postSynaptic['MVL07'] = [0,0]
        postSynaptic['MVL08'] = [0,0]
        postSynaptic['MVL09'] = [0,0]
        postSynaptic['MVL10'] = [0,0]
        postSynaptic['MVL11'] = [0,0]
        postSynaptic['MVL12'] = [0,0]
        postSynaptic['MVL13'] = [0,0]
        postSynaptic['MVL14'] = [0,0]
        postSynaptic['MVL15'] = [0,0]
        postSynaptic['MVL16'] = [0,0]
        postSynaptic['MVL17'] = [0,0]
        postSynaptic['MVL18'] = [0,0]
        postSynaptic['MVL19'] = [0,0]
        postSynaptic['MVL20'] = [0,0]
        postSynaptic['MVL21'] = [0,0]
        postSynaptic['MVL22'] = [0,0]
        postSynaptic['MVL23'] = [0,0]
        postSynaptic['MVR01'] = [0,0]
        postSynaptic['MVR02'] = [0,0]
        postSynaptic['MVR03'] = [0,0]
        postSynaptic['MVR04'] = [0,0]
        postSynaptic['MVR05'] = [0,0]
        postSynaptic['MVR06'] = [0,0]
        postSynaptic['MVR07'] = [0,0]
        postSynaptic['MVR08'] = [0,0]
        postSynaptic['MVR09'] = [0,0]
        postSynaptic['MVR10'] = [0,0]
        postSynaptic['MVR11'] = [0,0]
        postSynaptic['MVR12'] = [0,0]
        postSynaptic['MVR13'] = [0,0]
        postSynaptic['MVR14'] = [0,0]
        postSynaptic['MVR15'] = [0,0]
        postSynaptic['MVR16'] = [0,0]
        postSynaptic['MVR17'] = [0,0]
        postSynaptic['MVR18'] = [0,0]
        postSynaptic['MVR19'] = [0,0]
        postSynaptic['MVR20'] = [0,0]
        postSynaptic['MVR21'] = [0,0]
        postSynaptic['MVR22'] = [0,0]
        postSynaptic['MVR23'] = [0,0]
        postSynaptic['MVR24'] = [0,0]
        postSynaptic['MVULVA'] = [0,0]
        postSynaptic['NSML'] = [0,0]
        postSynaptic['NSMR'] = [0,0]
        postSynaptic['OLLL'] = [0,0]
        postSynaptic['OLLR'] = [0,0]
        postSynaptic['OLQDL'] = [0,0]
        postSynaptic['OLQDR'] = [0,0]
        postSynaptic['OLQVL'] = [0,0]
        postSynaptic['OLQVR'] = [0,0]
        postSynaptic['PDA'] = [0,0]
        postSynaptic['PDB'] = [0,0]
        postSynaptic['PDEL'] = [0,0]
        postSynaptic['PDER'] = [0,0]
        postSynaptic['PHAL'] = [0,0]
        postSynaptic['PHAR'] = [0,0]
        postSynaptic['PHBL'] = [0,0]
        postSynaptic['PHBR'] = [0,0]
        postSynaptic['PHCL'] = [0,0]
        postSynaptic['PHCR'] = [0,0]
        postSynaptic['PLML'] = [0,0]
        postSynaptic['PLMR'] = [0,0]
        postSynaptic['PLNL'] = [0,0]
        postSynaptic['PLNR'] = [0,0]
        postSynaptic['PQR'] = [0,0]
        postSynaptic['PVCL'] = [0,0]
        postSynaptic['PVCR'] = [0,0]
        postSynaptic['PVDL'] = [0,0]
        postSynaptic['PVDR'] = [0,0]
        postSynaptic['PVM'] = [0,0]
        postSynaptic['PVNL'] = [0,0]
        postSynaptic['PVNR'] = [0,0]
        postSynaptic['PVPL'] = [0,0]
        postSynaptic['PVPR'] = [0,0]
        postSynaptic['PVQL'] = [0,0]
        postSynaptic['PVQR'] = [0,0]
        postSynaptic['PVR'] = [0,0]
        postSynaptic['PVT'] = [0,0]
        postSynaptic['PVWL'] = [0,0]
        postSynaptic['PVWR'] = [0,0]
        postSynaptic['RIAL'] = [0,0]
        postSynaptic['RIAR'] = [0,0]
        postSynaptic['RIBL'] = [0,0]
        postSynaptic['RIBR'] = [0,0]
        postSynaptic['RICL'] = [0,0]
        postSynaptic['RICR'] = [0,0]
        postSynaptic['RID'] = [0,0]
        postSynaptic['RIFL'] = [0,0]
        postSynaptic['RIFR'] = [0,0]
        postSynaptic['RIGL'] = [0,0]
        postSynaptic['RIGR'] = [0,0]
        postSynaptic['RIH'] = [0,0]
        postSynaptic['RIML'] = [0,0]
        postSynaptic['RIMR'] = [0,0]
        postSynaptic['RIPL'] = [0,0]
        postSynaptic['RIPR'] = [0,0]
        postSynaptic['RIR'] = [0,0]
        postSynaptic['RIS'] = [0,0]
        postSynaptic['RIVL'] = [0,0]
        postSynaptic['RIVR'] = [0,0]
        postSynaptic['RMDDL'] = [0,0]
        postSynaptic['RMDDR'] = [0,0]
        postSynaptic['RMDL'] = [0,0]
        postSynaptic['RMDR'] = [0,0]
        postSynaptic['RMDVL'] = [0,0]
        postSynaptic['RMDVR'] = [0,0]
        postSynaptic['RMED'] = [0,0]
        postSynaptic['RMEL'] = [0,0]
        postSynaptic['RMER'] = [0,0]
        postSynaptic['RMEV'] = [0,0]
        postSynaptic['RMFL'] = [0,0]
        postSynaptic['RMFR'] = [0,0]
        postSynaptic['RMGL'] = [0,0]
        postSynaptic['RMGR'] = [0,0]
        postSynaptic['RMHL'] = [0,0]
        postSynaptic['RMHR'] = [0,0]
        postSynaptic['SAADL'] = [0,0]
        postSynaptic['SAADR'] = [0,0]
        postSynaptic['SAAVL'] = [0,0]
        postSynaptic['SAAVR'] = [0,0]
        postSynaptic['SABD'] = [0,0]
        postSynaptic['SABVL'] = [0,0]
        postSynaptic['SABVR'] = [0,0]
        postSynaptic['SDQL'] = [0,0]
        postSynaptic['SDQR'] = [0,0]
        postSynaptic['SIADL'] = [0,0]
        postSynaptic['SIADR'] = [0,0]
        postSynaptic['SIAVL'] = [0,0]
        postSynaptic['SIAVR'] = [0,0]
        postSynaptic['SIBDL'] = [0,0]
        postSynaptic['SIBDR'] = [0,0]
        postSynaptic['SIBVL'] = [0,0]
        postSynaptic['SIBVR'] = [0,0]
        postSynaptic['SMBDL'] = [0,0]
        postSynaptic['SMBDR'] = [0,0]
        postSynaptic['SMBVL'] = [0,0]
        postSynaptic['SMBVR'] = [0,0]
        postSynaptic['SMDDL'] = [0,0]
        postSynaptic['SMDDR'] = [0,0]
        postSynaptic['SMDVL'] = [0,0]
        postSynaptic['SMDVR'] = [0,0]
        postSynaptic['URADL'] = [0,0]
        postSynaptic['URADR'] = [0,0]
        postSynaptic['URAVL'] = [0,0]
        postSynaptic['URAVR'] = [0,0]
        postSynaptic['URBL'] = [0,0]
        postSynaptic['URBR'] = [0,0]
        postSynaptic['URXL'] = [0,0]
        postSynaptic['URXR'] = [0,0]
        postSynaptic['URYDL'] = [0,0]
        postSynaptic['URYDR'] = [0,0]
        postSynaptic['URYVL'] = [0,0]
        postSynaptic['URYVR'] = [0,0]
        postSynaptic['VA1'] = [0,0]
        postSynaptic['VA10'] = [0,0]
        postSynaptic['VA11'] = [0,0]
        postSynaptic['VA12'] = [0,0]
        postSynaptic['VA2'] = [0,0]
        postSynaptic['VA3'] = [0,0]
        postSynaptic['VA4'] = [0,0]
        postSynaptic['VA5'] = [0,0]
        postSynaptic['VA6'] = [0,0]
        postSynaptic['VA7'] = [0,0]
        postSynaptic['VA8'] = [0,0]
        postSynaptic['VA9'] = [0,0]
        postSynaptic['VB1'] = [0,0]
        postSynaptic['VB10'] = [0,0]
        postSynaptic['VB11'] = [0,0]
        postSynaptic['VB2'] = [0,0]
        postSynaptic['VB3'] = [0,0]
        postSynaptic['VB4'] = [0,0]
        postSynaptic['VB5'] = [0,0]
        postSynaptic['VB6'] = [0,0]
        postSynaptic['VB7'] = [0,0]
        postSynaptic['VB8'] = [0,0]
        postSynaptic['VB9'] = [0,0]
        postSynaptic['VC1'] = [0,0]
        postSynaptic['VC2'] = [0,0]
        postSynaptic['VC3'] = [0,0]
        postSynaptic['VC4'] = [0,0]
        postSynaptic['VC5'] = [0,0]
        postSynaptic['VC6'] = [0,0]
        postSynaptic['VD1'] = [0,0]
        postSynaptic['VD10'] = [0,0]
        postSynaptic['VD11'] = [0,0]
        postSynaptic['VD12'] = [0,0]
        postSynaptic['VD13'] = [0,0]
        postSynaptic['VD2'] = [0,0]
        postSynaptic['VD3'] = [0,0]
        postSynaptic['VD4'] = [0,0]
        postSynaptic['VD5'] = [0,0]
        postSynaptic['VD6'] = [0,0]
        postSynaptic['VD7'] = [0,0]
        postSynaptic['VD8'] = [0,0]
        postSynaptic['VD9'] = [0,0]

#global postSynapticNext = copy.deepcopy(postSynaptic)


def motorcontrol():
    global accumright
    global accumleft
    accumleft = 0
    accumright = 0
    # accumulate left and right muscles and the accumulated values are
    # used to move the left and right motors of the robot
    for muscle in muscleList:
        if muscle in mLeft:
            accumleft += postSynaptic[muscle][nextState]
            # accumleft = accumleft + postSynaptic[muscle][thisState] #what???  For some reason, thisState weight is always 0.
            # postSynaptic[muscle][thisState] = 0
            #print(muscle, "Before", postSynaptic[muscle][thisState], accumleft)  # Both states have to be set to 0 once the muscle is fired, or
            postSynaptic[muscle][nextState] = 0
            #print(muscle, "After", postSynaptic[muscle][thisState], accumleft)  # it will keep returning beyond the threshold within one iteration.
        elif muscle in mRight:
            accumright += postSynaptic[muscle][nextState]
            # accumleft = accumright + postSynaptic[muscle][thisState] #what???
            # postSynaptic[muscle][thisState] = 0
            postSynaptic[muscle][nextState] = 0

    # We turn the wheels according to the motor weight accumulation
    new_speed = abs(accumleft) + abs(accumright)
    if new_speed > 150:
        new_speed = 150
    elif new_speed < 75:
        new_speed = 75
    return [accumleft,accumright,new_speed]
    
    ## Start Commented section
    # set_speed(new_speed)
    # if accumleft == 0 and accumright == 0:
    #         stop()
    # elif accumright <= 0 and accumleft < 0:
    #         set_speed(150)
    #         turnratio = float(accumright) / float(accumleft)
    #         # print "Turn Ratio: ", turnratio
    #         if turnratio <= 0.6:
    #                  left_rot()
    #                  time.sleep(0.8)
    #         elif turnratio >= 2:
    #                  right_rot()
    #                  time.sleep(0.8)
    #         bwd()
    #         time.sleep(0.5)
    # elif accumright <= 0 and accumleft >= 0:
    #         right_rot()
    #         time.sleep(.8)
    # elif accumright >= 0 and accumleft <= 0:
    #         left_rot()
    #         time.sleep(.8)
    # elif accumright >= 0 and accumleft > 0:
    #         turnratio = float(accumright) / float(accumleft)
    #         # print "Turn Ratio: ", turnratio
    #         if turnratio <= 0.6:
    #                  left_rot()
    #                  time.sleep(0.8)
    #         elif turnratio >= 2:
    #                  right_rot()
    #                  time.sleep(0.8)
    #         fwd()
    #         time.sleep(0.5)
    # else:
    #         stop()
    ## End Commented section
    accumleft = 0
    accumright = 0
    time.sleep(0.5)


def dendriteAccumulate(dneuron):
    f = eval(dneuron)
    f()


def fireNeuron(fneuron):
    # The threshold has been exceeded and we fire the neurite
    if fneuron != "MVULVA":
        f = eval(fneuron)
        f()
        # postSynaptic[fneuron][nextState] = 0
        # postSynaptic[fneuron][thisState] = 0
        postSynaptic[fneuron][nextState] = 0


def runconnectome():
    # Each time a set of neuron is stimulated, this method will execute
    # The weighted values are accumulated in the postSynaptic array
    # Once the accumulation is read, we see what neurons are greater
    # than the threshold and fire the neuron or muscle that has exceeded
    # the threshold 
    global thisState
    global nextState

    for ps in postSynaptic:
        if ps[:3] not in muscles and abs(postSynaptic[ps][thisState]) > threshold:
            fireNeuron(ps)
            # print(ps)
            # print(ps)
            # postSynaptic[ps][nextState] = 0
    movement = motorcontrol()
    for ps in postSynaptic:
        # if postSynaptic[ps][thisState] != 0:
        # print(ps)
        # print("Before Clone: ", postSynaptic[ps][thisState])
        postSynaptic[ps][thisState] = copy.deepcopy(postSynaptic[ps][nextState])  # fired neurons keep getting reset to previous weight
        # print("After Clone: ", postSynaptic[ps][thisState])
    thisState, nextState = nextState, thisState
    
    return movement






def move(dist,sees_food):
                        if dist > 0 and dist < 100:
                                dendriteAccumulate("FLPR")
                                dendriteAccumulate("FLPL")
                                dendriteAccumulate("ASHL")
                                dendriteAccumulate("ASHR")
                                dendriteAccumulate("IL1VL")
                                dendriteAccumulate("IL1VR")
                                dendriteAccumulate("OLQDL")
                                dendriteAccumulate("OLQDR")
                                dendriteAccumulate("OLQVR")
                                dendriteAccumulate("OLQVL")
                                return (runconnectome())
                                
                        else:
                                if sees_food:
                                        dendriteAccumulate("ADFL")
                                        dendriteAccumulate("ADFR")
                                        dendriteAccumulate("ASGR")
                                        dendriteAccumulate("ASGL")
                                        dendriteAccumulate("ASIL")
                                        dendriteAccumulate("ASIR")
                                        dendriteAccumulate("ASJR")
                                        dendriteAccumulate("ASJL")
                                        return runconnectome()
                                        #tfood += 0.5
                                        #if tfood > 20:
                                        #        tfood = 0
                                else: return runconnectome()

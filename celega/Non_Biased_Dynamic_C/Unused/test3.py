all_neuron_names = [
    'ADAL', 'ADAR', 'ADEL', 'ADER', 'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AFDL', 'AFDR', 'AIAL', 'AIAR', 'AIBL', 'AIBR', 'AIML', 'AIMR',
    'AINL', 'AINR', 'AIYL', 'AIYR', 'AIZL', 'AIZR', 'ALA', 'ALML', 'ALMR', 'ALNL', 'ALNR', 'AQR', 'AS1', 'AS2', 'AS3', 'AS4', 'AS5',
    'AS6', 'AS7', 'AS8', 'AS9', 'AS10', 'AS11', 'ASEL', 'ASER', 'ASGL', 'ASGR', 'ASHL', 'ASHR', 'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL',
    'ASKR', 'AUAL', 'AUAR', 'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER', 'AVFL', 'AVFR', 'AVG', 'AVHL', 'AVHR', 'AVJL',
    'AVJR', 'AVKL', 'AVKR', 'AVL', 'AVM', 'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR', 'BAGL', 'BAGR', 'BDUL', 'BDUR', 'CEPDL', 'CEPDR',
    'CEPVL', 'CEPVR', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9', 'DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB6', 'DB7', 'DD1',
    'DD2', 'DD3', 'DD4', 'DD5', 'DD6', 'DVA', 'DVB', 'DVC', 'FLPL', 'FLPR', 'HSNL', 'HSNR', 'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6',
    'IL1DL', 'IL1DR', 'IL1L', 'IL1R', 'IL1VL', 'IL1VR', 'IL2L', 'IL2R', 'IL2DL', 'IL2DR', 'IL2VL', 'IL2VR', 'LUAL', 'LUAR', 'M1', 'M2L', 'M2R',
    'M3L', 'M3R', 'M4', 'M5', 'MCL', 'MCR', 'MDL01', 'MDL02', 'MDL03', 'MDL04', 'MDL05', 'MDL06', 'MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11',
    'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MDL24', 'MDR01', 'MDR02', 'MDR03',
    'MDR04', 'MDR05', 'MDR06', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19',
    'MDR20', 'MDR21', 'MDR22', 'MDR23', 'MDR24', 'MI', 'MVL01', 'MVL02', 'MVL03', 'MVL04', 'MVL05', 'MVL06', 'MVL07', 'MVL08', 'MVL09', 'MVL10',
    'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23', 'MVR01', 'MVR02', 'MVR03',
    'MVR04', 'MVR05', 'MVR06', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19',
    'MVR20', 'MVR21', 'MVR22', 'MVR23', 'MVR24', 'MVULVA', 'NSML', 'NSMR', 'OLLL', 'OLLR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'PDA', 'PDB', 'PDEL',
    'PDER', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'PHCL', 'PHCR', 'PLML', 'PLMR', 'PLNL', 'PLNR', 'PQR', 'PVCL', 'PVCR', 'PVDL', 'PVDR', 'PVM', 'PVNL',
    'PVNR', 'PVPL', 'PVPR', 'PVQL', 'PVQR', 'PVR', 'PVT', 'PVWL', 'PVWR', 'RIAL', 'RIAR', 'RIBL', 'RIBR', 'RICL', 'RICR', 'RID', 'RIFL', 'RIFR',
    'RIGL', 'RIGR', 'RIH', 'RIML', 'RIMR', 'RIPL', 'RIPR', 'RIR', 'RIS', 'RIVL', 'RIVR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMED',
    'RMEL', 'RMER', 'RMEV', 'RMFL', 'RMFR', 'RMGL', 'RMGR', 'RMHL', 'RMHR', 'SAADL', 'SAADR', 'SAAVL', 'SAAVR', 'SABD', 'SABVL', 'SABVR', 'SDQL',
    'SDQR', 'SIADL', 'SIADR', 'SIAVL', 'SIAVR', 'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR', 'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR', 'SMDDL', 'SMDDR', 'SMDVL',
    'SMDVR', 'URADL', 'URADR', 'URAVL', 'URAVR', 'URBL', 'URBR', 'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL', 'URYVR', 'VA1', 'VA2', 'VA3', 'VA4', 'VA5',
    'VA6', 'VA7', 'VA8', 'VA9', 'VA10', 'VA11', 'VA12', 'VB1', 'VB2', 'VB3', 'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VB10', 'VB11', 'VC1', 'VC2',
    'VC3', 'VC4', 'VC5', 'VC6', 'VD1', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9', 'VD10', 'VD11', 'VD12', 'VD13'
]


neuron_groups = {
    "Sensory Neurons": [
        'ADAL', 'ADAR', 'ADFL', 'ADFR', 'ADLL', 'ADLR', 'ASEL', 'ASER', 'ASHL', 'ASHR',
        'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR', 'AUAL', 'AUAR', 'AWAL', 'AWAR',
        'AWBL', 'AWBR', 'AWCL', 'AWCR', 'BAGL', 'BAGR', 'CEPDL', 'CEPDR', 'CEPVL',
        'CEPVR', 'FLPL', 'FLPR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'PDEL', 'PDER',
        'PHAL', 'PHAR', 'PHBL', 'PHBR', 'PHCL', 'PHCR', 'PLML', 'PLMR', 'PLNL', 'PLNR',
        'PQR', 'SDQL', 'SDQR', 'URADL', 'URADR', 'URAVL', 'URAVR', 'URBL', 'URBR',
        'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL', 'URYVR', 'ADEL', 'ADER', 'AFDL',
        'AFDR', 'ALNL', 'ALNR', 'AS1', 'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7',
        'AS8', 'AS9', 'AS10', 'AS11', 'ASGL', 'ASGR', 'AVL', 'BDUL', 'BDUR'
    ],
    "Interneurons": [
        'AIAL', 'AIAR', 'AIBL', 'AIBR', 'AIML', 'AIMR', 'AINL', 'AINR', 'AIYL', 'AIYR',
        'AIZL', 'AIZR', 'ALA', 'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL',
        'AVER', 'AVFL', 'AVFR', 'AVG', 'AVHL', 'AVHR', 'AVJL', 'AVJR', 'AVKL', 'AVKR',
        'DVA', 'DVB', 'DVC', 'RIAL', 'RIAR', 'RIBL', 'RIBR', 'RICL', 'RICR', 'RID',
        'RIFL', 'RIFR', 'RIGL', 'RIGR', 'RIH', 'RIML', 'RIMR', 'RIPL', 'RIPR', 'RIR',
        'RIS', 'RIVL', 'RIVR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR',
        'RMED', 'RMEL', 'RMER', 'RMEV', 'RMFL', 'RMFR', 'RMGL', 'RMGR', 'RMHL', 'RMHR',
        'SAADL', 'SAADR', 'SAAVL', 'SAAVR', 'SABD', 'SABVL', 'SABVR', 'SIADL', 'SIADR',
        'SIAVL', 'SIAVR', 'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR', 'SMBDL', 'SMBDR', 'SMBVL',
        'SMBVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'PVCL', 'PVCR', 'PVDL', 'PVDR',
        'PVM', 'PVNL', 'PVNR', 'PVPL', 'PVPR', 'PVQL', 'PVQR', 'PVR', 'PVT', 'PVWL',
        'PVWR', 'MCL', 'MCR'
    ],
    "Pharyngeal Neurons": [
        'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6', 'M1', 'M2L', 'M2R', 'M3L',
        'M3R', 'M4', 'M5'
    ],
    "Other Groups": [
        'AQR', 'HSNL', 'HSNR', 'NSML', 'NSMR', 'IL1DL', 'IL1DR', 'IL1L', 'IL1R', 'IL1VL',
        'IL1VR', 'IL2L', 'IL2R', 'IL2DL', 'IL2DR', 'IL2VL', 'IL2VR', 'LUAL', 'LUAR',
        'PDA', 'PDB'
    ],
    "Mechanosensory Neurons": [
        'ALML', 'ALMR', 'AVM'
    ],
    "Touch Neurons": [
        'DE1', 'DE2'
    ],
    "Specialized Neurons": [
        'RIAL', 'RIAR', 'RIGL', 'RIGR', 'RIPL', 'RIPR'
    ],
    "Ring Interneurons": [
        'RIAL', 'RIAR'
    ],
    "Motor Neurons": [
        'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9', 'DB1', 'DB2',
        'DB3', 'DB4', 'DB5', 'DB6', 'DB7', 'DD1', 'DD2', 'DD3', 'DD4', 'DD5', 'DD6',
        'VA1', 'VA2', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7', 'VA8', 'VA9', 'VA10', 'VA11',
        'VA12', 'VB1', 'VB2', 'VB3', 'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VB10',
        'VB11', 'VC1', 'VC2', 'VC3', 'VC4', 'VC5', 'VC6', 'VD1', 'VD2', 'VD3', 'VD4',
        'VD5', 'VD6', 'VD7', 'VD8', 'VD9', 'VD10', 'VD11', 'VD12', 'VD13', 'MI',
        'MDL01', 'MDL02', 'MDL03', 'MDL04', 'MDL05', 'MDL06', 'MDL07', 'MDL08', 'MDL09',
        'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18',
        'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MDL24', 'MDR01', 'MDR02', 'MDR03',
        'MDR04', 'MDR05', 'MDR06', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12',
        'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDR21',
        'MDR22', 'MDR23', 'MDR24', 'MVL01', 'MVL02', 'MVL03', 'MVL04', 'MVL05', 'MVL06',
        'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15',
        'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23', 'MVR01',
        'MVR02', 'MVR03', 'MVR04', 'MVR05', 'MVR06', 'MVR07', 'MVR08', 'MVR09', 'MVR10',
        'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19',
        'MVR20', 'MVR21', 'MVR22', 'MVR23', 'MVR24', 'MVULVA', 'OLLL', 'OLLR', 'PDA', 'PDB'
    ]
}


missing_neurons = [neuron for neuron in all_neuron_names if neuron not in [neuron for group in neuron_groups.values() for neuron in group]]
if missing_neurons:
    print(f"Missing neurons: {missing_neurons}")
else:
    print("All neurons are categorized.")

import pandas as pd
import numpy as np
from util.write_read_txt import flatten_dict_values,read_excel
# Read the Excel file




# Example dictionary with motor neurons
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names

# List of motor neurons
# Path to the Excel file
file_path = 'CElegansNeuronTables.xls'

# Read the Excel data
data = read_excel("CElegansNeuronTables.xlsx")

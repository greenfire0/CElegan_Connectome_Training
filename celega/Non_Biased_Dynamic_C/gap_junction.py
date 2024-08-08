import pandas as pd
import numpy as np
from util.write_read_txt import flatten_dict_values,read_excel
# Read the Excel file


# Find gap junction indices while skipping motor neurons
def find_gap_junction_indices(data, connection_dict, motors):
    gap_junction_indices = []
    values_list = flatten_dict_values(connection_dict)
    
    for row in data:
        origin, target, connection_type, num_connections, neurotransmitter = row
        
        # Skip motor neurons
        if origin in motors or target in motors:
            continue
        
        # Check if the connection type is GapJunction
        if connection_type == 'GapJunction':
            print("")
            # Find the index of this connection in the flattened list
            for i, (post_synaptic, value) in enumerate(values_list):
                if post_synaptic == target and connection_dict[origin][target] == value:
                    gap_junction_indices.append(i)
        print(gap_junction_indices)
        quit()
    
    return gap_junction_indices

# Example dictionary with motor neurons
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names

# List of motor neurons
motors = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "DVA", "DVB", "PVC"]

# Path to the Excel file
file_path = 'CElegansNeuronTables.xls'

# Read the Excel data
data = read_excel("CElegansNeuronTables.xls")

# Find the indices of gap junctions excluding motor neurons
gap_junction_indices = find_gap_junction_indices(data, dict, motors)
print(gap_junction_indices)

import pandas as pd
import numpy as np
from util.write_read_txt import flatten_dict_values,read_excel,retreive_nums
# Read the Excel file


def enf_gap(data, values_list, motors):
    for row in data:
        origin, target, connection_type, num_connections, neurotransmitter = row
        
        # Skip motor neurons
        if origin in motors or target in motors:
            continue
        # Check if the connection type is GapJunction
        if connection_type == 'GapJunction':
            # Find the index of this connection in the flattened list
            for i, (post_synaptic, value,orig) in enumerate(values_list):
#                print(post_synaptic,i) if origin == orig or target==orig else""
                if (orig == origin and post_synaptic == target) :
                    indacy = 0
                    for num,(post,_,originzz) in enumerate(values_list):
                            if ((orig == post and post_synaptic == originzz) or (post_synaptic == post and originzz == origin))and num !=i:
                                indacy=num
                                break


                    if len(values_list[indacy][1]) == 2 and len(values_list[i][1])==1:
                        # Update the second element of values_list[indacy][1] to reference values_list[i][1]
                        values_list[indacy][1][1] = values_list[i][1][0]
                    elif (len(values_list[i][1])==2 and len(values_list[indacy][1])==1 and type(values_list[i][1])==int):
                        values_list[i][1][1] = values_list[indacy][1]
                    elif ((len(values_list[i][1])==1)and len(values_list[indacy][1])==1) and indacy>i:
                        values_list[indacy][1][0] = values_list[i][1][0]
                        #print((indacy,i)) if max(indacy,i) < 1000 else ""
                    else:
                        pass
    return retreive_nums(values_list)

def find_gap_junction_indices(data, values_list, motors):
    accessed_indices = set()  # Initialize an empty set to store accessed indices

    for row in data:
        origin, target, connection_type, num_connections, neurotransmitter = row

        # Skip motor neurons
        if origin in motors or target in motors:
            continue
        if connection_type == 'GapJunction':
            # Find the index of this connection in the flattened list
            for i, (post_synaptic, value, orig) in enumerate(values_list):
                if (orig == origin and post_synaptic == target) or (orig == target and post_synaptic == origin):
                    indacy = 0
                    for num, (post, _, originzz) in enumerate(values_list):
                        if ((orig == post and post_synaptic == originzz) or (post_synaptic == post and originzz == origin)) and num != i:
                            indacy = num
                            break
                    if (len(values_list[i][1]) == 1 and len(values_list[indacy][1]) == 1) and indacy > i:
                        accessed_indices.add(indacy)  # Track the accessed index
                    else:
                        pass

    return sorted(accessed_indices)

# Example dictionary with motor neurons
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names,data

if 0:
    array = flatten_dict_values(dict)
    array[0]=(array[0][0],[5],array[0][2])
    array[20]=(array[20][0],[5],array[20][2])

    # Find the indices of gap junctions excluding motor neurons
    gap_junction_indices = enf_gap(data, array, muscleList)
    gap_junction_indices= (gap_junction_indices)
    ind_to_not_change = (find_gap_junction_indices(data, array, muscleList))
    for a in gap_junction_indices[0:100]:
            print(a[2],a[1],a[0]) #if len(a[1]) > 1 else""
def find_motor_ind(dictionary, muscles):
    frozen_indices = []
    post_synaptic_set = set()
    
    # Flatten the dictionary to a list of tuples (post_synaptic, value)
    flattened_connections = []
    for sub_dict in dictionary.values():
        flattened_connections.extend(sub_dict.items())
    
    # Iterate through the flattened list and check for matching prefixes

    for index, (post_synaptic, value) in enumerate(flattened_connections):
        if any(post_synaptic.startswith(muscle) for muscle in muscles):
            post_synaptic_set.add(post_synaptic)
            frozen_indices.append(index)
    
    # Print the set of all unique post-synaptic elements
    #print(post_synaptic_set)
    
    return frozen_indices


def get_indicies_to_change(frozen,length):
    import numpy as np
    all_numbers = np.arange(length)

    # Convert frozen_indices to a numpy array
    frozen_indices_np = np.array(frozen)

    # Use numpy's set difference function to remove frozen indices from the list
    filtered_numbers = np.setdiff1d(all_numbers, frozen_indices_np)
    # Convert the result back to a list (if needed)
    filtered_numbers_list = filtered_numbers.tolist()
    # Print the filtered list
    return(filtered_numbers_list)
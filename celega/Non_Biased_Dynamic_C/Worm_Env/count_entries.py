# Import the weights dictionary
# Function to count the number of connections
from Worm_Env.weight_dict import dict,all_neuron_names
def count_total_entries(weights_dict):
    connections = 0
    neurons= 0
    for inner_dict in weights_dict.values():
        neurons+=1
        connections += len(inner_dict)
    print(len(all_neuron_names),connections)
    return connections

# Main function
if __name__ == "__main__":
    total_entries = count_total_entries(dict)
    print(f"Total number of entries: {total_entries}")
import pandas as pd

def process_sheet(df, how_do_i_do_this):

    data_dict = {}
    arr = ['Origin', 'Target'] 
    if how_do_i_do_this:
        arr = ['Neuron', 'Muscle']
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        from_neuron = row[arr[0]] 
        to_neuron = row[arr[1]]
        if arr[0] == 'Origin':
            type_con = row['Type']
            
        else:
            type_con = 'Send'
        weight = row['Number of Connections']
        neuromodulator = row['Neurotransmitter']

        # Convert weight to negative if neuromodulator is GABA
        if neuromodulator == 'GABA':
            weight = -weight

        # Initialize dictionary entry for the sending neuron if not already present
        if from_neuron not in data_dict:
            data_dict[from_neuron] = {}

        # Add or update the connection
        
        if to_neuron in data_dict[from_neuron]:
            data_dict[from_neuron][to_neuron]+=(weight)
        else:
            data_dict[from_neuron][to_neuron] = weight

    return data_dict

def combine_neuron_data(file_path):
    # Load the Excel file
    connectome_df = pd.read_excel(file_path, sheet_name='Connectome')
    neurons_to_muscle_df = pd.read_excel(file_path, sheet_name='NeuronsToMuscle')

    # Process both sheets
    connectome_data = process_sheet(connectome_df, 0)
    neurons_to_muscle_data = process_sheet(neurons_to_muscle_df, 1)

    # Combine the dictionaries
    combined_data = connectome_data.copy()

    # Merge data from neurons_to_muscle_data into combined_data
    for neuron, connections in neurons_to_muscle_data.items():
        if neuron in combined_data:
            for target, weight in connections.items():
                if target in combined_data[neuron]:
                    print("gasdhjghasdjghjasdhjkl",neuron,target)

                    combined_data[neuron][target] += weight
                else:
                    combined_data[neuron][target] = weight
        else:
            combined_data[neuron] = connections
    return combined_data


def get_all_neuron_names(combined_data):
    neuron_names = set()

    # Iterate through the keys and values in the combined data
    for from_neuron, connections in combined_data.items():
        neuron_names.add(from_neuron)
        for to_neuron in connections.keys():
            neuron_names.add(to_neuron)

    return sorted(neuron_names)
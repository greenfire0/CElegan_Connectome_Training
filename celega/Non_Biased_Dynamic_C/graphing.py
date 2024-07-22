import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics
from util.multimode import custom_multimode
from collections import defaultdict

muscles = ['MVU', 'MVL', 'MDL', 'MVR', 'MDR']

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
    "Chemosensory Neurons": [
        'ASEL', 'ASER', 'ASGL', 'ASGR', 'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR',
        'ASHL', 'ASHR', 'PLNL', 'PLNR'
    ],
    "Mechanosensory Neurons": [
        'ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVD',  'ALML', 'ALMR', 'AVM'
    ],
    "Thermosensory Neurons": [
        'AFDL', 'AFDR', 'AFD'
    ],
    "Photosensory Neurons": [
        'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR', 'URBL', 'URBR'
    ],
    "Multimodal Sensory Neurons": [
        'ADAL', 'ADAR', 'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AUAL', 'AUAR', 'AWAL', 'AWAR',
        'AWBL', 'AWBR', 'AWCL', 'AWCR', 'BAGL', 'BAGR', 'FLPL', 'FLPR', 'OLQDL', 'OLQDR',
        'OLQVL', 'OLQVR', 'PDEL', 'PDER', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'PHCL', 'PHCR',
        'PQR', 'SDQL', 'SDQR', 'URADL', 'URADR', 'URAVL', 'URAVR', 'URXL', 'URXR', 'URYDL',
        'URYDR', 'URYVL', 'URYVR', 'ADEL', 'ADER', 'AFDL', 'AFDR', 'ALNL', 'ALNR', 'AS1',
        'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AS10', 'AS11', 'ASGL',
        'ASGR', 'AVL', 'BDUL', 'BDUR'
    ],
    "Locomotion-related Interneurons": [
        'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER', 'RIML', 'RIMR',
        'SAADL', 'SAADR', 'SAAVL', 'SAAVR', 'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR', 'SMDDL',
        'SMDDR', 'SMDVL', 'SMDVR', 'PVCL', 'PVCR', 'PVDL', 'PVDR'
    ],
    "Feeding-related Interneurons": [
        'AIBL', 'AIBR', 'AIML', 'AIMR', 'AINL', 'AINR', 'RIBL', 'RIBR', 'RICL', 'RICR',
        'RID', 'RIFL', 'RIFR', 'RIGL', 'RIGR', 'RIH', 'RIPL', 'RIPR', 'RIR', 'RIS', 'RIVL',
        'RIVR', 'RMED', 'RMEL', 'RMER', 'RMEV', 'RMFL', 'RMFR', 'RMGL', 'RMGR', 'RMHL',
        'RMHR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR'
    ],
    "Reproductive Interneurons": [
        'HSNL', 'HSNR', 'PVNL', 'PVNR', 'PVQL', 'PVQR', 'PVR', 'PVT', 'PVWL', 'PVWR',
        'PVM', 'PVPL', 'PVPR'
    ],
    "Sensory Integration Interneurons": [
        'AIAL', 'AIAR', 'AIYL', 'AIYR', 'AIZL', 'AIZR', 'ALA', 'AVFL', 'AVFR', 'AVHL',
        'AVHR', 'AVJL', 'AVJR', 'AVKL', 'AVKR', 'DVA', 'DVB', 'DVC', 'RIAL', 'RIAR', 'RIH',
        'RIS', 'AVG'
    ],
    "Neuroendocrine Interneurons": [
        'ALA', 'NSM'
    ],
    "Pharyngeal Neurons": [
        'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6', 'M1', 'M2L', 'M2R', 'M3L',
        'M3R', 'M4', 'M5', 'MCL', 'MCR'
    ],
    "Other Groups": [
        'AQR', 'HSNL', 'HSNR', 'NSML', 'NSMR', 'IL1DL', 'IL1DR', 'IL1L', 'IL1R', 'IL1VL',
        'IL1VR', 'IL2L', 'IL2R', 'IL2DL', 'IL2DR', 'IL2VL', 'IL2VR', 'LUAL', 'LUAR',
        'PDA', 'PDB', 'SABD', 'SABVL', 'SABVR', 'SIADL', 'SIADR', 'SIAVL', 'SIAVR',
        'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR'
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

def graph(combined_weights, connections_dict, generation,old_wm,shortest_distances):
    assert type(combined_weights) == type(np.array([0])) ,  f"Expected type {type(np.array([0]))}, but got type {type(combined_weights)}"
    def plot_weight_distribution(ax, weight_matrix1, weight_matrix2, num_bins=30):
            non_zero_weights1 = weight_matrix1[weight_matrix1 != 0]
            non_zero_weights2 = weight_matrix2[weight_matrix2 != 0]

            # Set x-axis limits
            xlim = [-40, 40]

            # Generate evenly spaced bins over the specified xlim range
            bins = np.linspace(xlim[0], xlim[1], num_bins + 1)

            # Calculate histogram data
            hist1, _ = np.histogram(non_zero_weights1, bins=bins)
            hist2, _ = np.histogram(non_zero_weights2, bins=bins)
            
            # Calculate the difference between the histograms
            hist_diff = hist1

            # Plotting histogram difference using ax.bar
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, hist_diff, width=np.diff(bins), color='lightcoral', edgecolor='black')

            ax.set_ylim([-35, 35])
            ax.set_xlim(xlim)
            
            ax.set_xlabel('Weight')
            ax.set_ylabel('Difference in Frequency')
            ax.set_title('Difference in Weight Distribution (Non-zero)')
    

    neuron_labels = list(all_neuron_names)
    matrix_size = len(neuron_labels)
    n_neurons = len(connections_dict.keys())
    
    # Create a zero matrix and fill with weights
    square_weight_matrix = np.zeros((n_neurons, matrix_size))
    index = 0
    for i, pre_neuron in enumerate(connections_dict.keys()):
        connections = connections_dict[pre_neuron]
        for j, post_neuron in enumerate(neuron_labels):
            if post_neuron in connections:
                square_weight_matrix[i, j] = combined_weights[index]
                index += 1

    weight_matrix_df = pd.DataFrame(square_weight_matrix, index=list(connections_dict.keys()), columns=neuron_labels)
    row_weights = weight_matrix_df.sum(axis=1)
    sorted_row_indices = np.argsort(row_weights)[::-1]
    sorted_matrix_df = weight_matrix_df.iloc[sorted_row_indices]

    max_weight = 20
    vmin = -max_weight
    vmax = max_weight

    plt.figure(figsize=(18, 15))

    # Weight Matrix Subplot
    plt.subplot(3, 3, 1)
    c = plt.pcolormesh(sorted_matrix_df, cmap='twilight', vmin=vmin, vmax=vmax)
    plt.title('Weight Matrix')
    plt.xlabel('Post Neurons')
    plt.ylabel('Pre Neurons')
    plt.gca().invert_yaxis()
    plt.colorbar(c, label='Weight')

    # Precompute sums
    abs_combined_weights = np.abs(combined_weights)
    neuron_indices = {neuron: i for i, neuron in enumerate(all_neuron_names)}
    weight_sums = {post_neuron: 0 for post_neuron in all_neuron_names}
    weight_quots = {post_neuron: 0 for post_neuron in all_neuron_names}
    
    tot_product = 0
    tot_quot = 0
    
    for post_neuron in all_neuron_names:
        pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
        if pre_neurons:
            indices = [neuron_indices[pre_neuron] for pre_neuron in pre_neurons]
            weights = abs_combined_weights[indices]
            n = len(pre_neurons)
            weight_product = np.sum(weights) * n
            weight_quotient = np.sum(weights) / n
            weight_sums[post_neuron] = weight_product
            weight_quots[post_neuron] = weight_quotient
            tot_product += weight_product
            tot_quot += weight_quotient
            
    post_neurons = list(weight_sums.keys())
    avg_weight_sums = [weight_sums[post_neuron] / tot_product for post_neuron in post_neurons]
    avg_weight_quots = [weight_quots[post_neuron] / tot_quot for post_neuron in post_neurons]

    # Average Weight Sum Subplot
    plt.subplot(3, 3, 2)
    plt.bar(post_neurons, avg_weight_sums, color='skyblue')
    plt.xlabel('Post Neurons')
    plt.ylabel('Normalized Weight Product')
    plt.ylim(0, 0.1)
    plt.title('Weight Sum * Number of Connections')

    # Average Weight Quotient Subplot
    plt.subplot(3, 3, 3)
    plt.bar(post_neurons, avg_weight_quots, color='skyblue')
    plt.xlabel('Post Neurons')
    plt.ylabel('Normalized Weight Quotient')
    plt.ylim(0, 0.01)
    plt.title('Weight Sum / Number of Connections')




    # Identify transitions
    neg_to_pos_indices = np.where((old_wm < 0) & (combined_weights > 0))[0]
    pos_to_neg_indices = np.where((old_wm > 0) & (combined_weights < 0))[0]

    # Calculate the counts
    neg_to_pos = len(neg_to_pos_indices)
    pos_to_neg = len(pos_to_neg_indices)
    #print("Indices where old_wm is negative and combined_weights is positive:", neg_to_pos_indices)
    #print("Indices where old_wm is positive and combined_weights is negative:", pos_to_neg_indices)

    # Plotting the results
    labels = ['Negative to Positive', 'Positive to Negative']
    counts = [neg_to_pos, pos_to_neg]
    plt.subplot(3, 3, 4)
    plt.bar(labels, counts, color=['blue', 'red'])
    plt.xlabel('Transition Type')
    plt.ylabel('Count')
    plt.title('Count of Transitions Between Negative and Positive Values')
    plt.ylim(0,20)


    # Percentage of Weights Greater Than 20 by Number of Pre-Neurons Subplot
    neuron_to_group = {}
    for group, neurons in neuron_groups.items():
        for neuron in neurons:
            neuron_to_group[neuron] = group
    #print(neuron_to_group)  
    # Initialize dictionaries to store the sums
    group_sums_new = {group: 0 for group in neuron_groups.keys()}

    groups = list(neuron_groups.keys())


    def print_connection_info(neuron_to_group):
        #print(shortest_distances)
        group_sums_new = {group: [] for group in neuron_groups.keys()}
        print("Mode connection distance from motor neuron")
        for pre_neuron in connections_dict.keys():
            if pre_neuron[:3] not in muscles:
                neuron_connections_old = connections_dict[pre_neuron]
                #sum_old = 0
                #sum_new = 0
                group = neuron_to_group.get(pre_neuron)     
                val = shortest_distances.get(pre_neuron)
                group_sums_new[group].append(val)
        for a in group_sums_new.keys():
            if group_sums_new[a] !=[]:
                print(a,custom_multimode(group_sums_new[a]))
        print("Direct Connections (Connections with 1 distance) to the Motor Neurons by neuron group")
        lst= []
        neuron_to_group = shortest_distances
        dist_groups = np.unique(list(neuron_to_group.values()))
        # Initialize dictionaries to store the sums
        group_sums_new = {group: 0 for group in dist_groups}
        c = 0
        for pre_neuron in connections_dict.keys():
            if pre_neuron[:3] not in muscles:
                neuron_connections_old = connections_dict[pre_neuron]
                #sum_old = 0
                #sum_new = 0
                group = neuron_to_group.get(pre_neuron)
                if group ==1:
                    lst.append(pre_neuron)
                    
        group_counts = {group: 0 for group in neuron_groups}

        # Count the preneurons in lst for each group
        for neuron in lst:
            for group, neurons in neuron_groups.items():
                if neuron in neurons:
                    group_counts[group] += 1
                    break  # Move to the next neuron in lst once a match is found
        # Output the counts
        print(group_counts)
    
    def plot_loco_neuron_changes(subplot, loco_neuron_change, t, c):
        """
        Plots a bar graph of locomotion-related neuron changes with a solid horizontal line at y=0.

        Parameters:
        subplot (matplotlib.axes.Axes): The subplot to plot the graph on.
        loco_neuron_change (dict): Dictionary of neuron names and their corresponding changes.
        """
        neuron_names = list(loco_neuron_change.keys())
        changes = list(loco_neuron_change.values())

        # Plot the bar chart with adjusted width and spacing
        bar_width = 0.6  # Width of bars
        x = np.arange(len(neuron_names))  # Positions of bars
        subplot.bar(x, changes, width=bar_width, color=c,edgecolor='black',)

        # Set the labels and title
        subplot.set_xlabel('Neuron Names')
        subplot.set_ylabel('Changes')
        subplot.set_ylim([-5, 5])
        subplot.tick_params(axis='x', labelsize=6)
        subplot.set_title(t)
        subplot.tick_params(axis='x', rotation=90)

        # Draw solid horizontal line at y=0

        # Adjust x-ticks to match the bars
        subplot.set_xticks(x)
        subplot.set_xticklabels(neuron_names)

    loco_neuron_change_neg = defaultdict(lambda: 0)
    loco_neuron_change_pos = defaultdict(lambda: 0)
    modal_neuron_change_neg = defaultdict(lambda: 0)
    modal_neuron_change_pos = defaultdict(lambda: 0)


    neuron_to_group = shortest_distances
    dist_groups = np.unique(list(neuron_to_group.values()))
    # Initialize dictionaries to store the sums
    group_sums_new = {group: 0 for group in dist_groups}
    c = 0
    for pre_neuron in connections_dict.keys():

        if pre_neuron[:3] not in muscles:
            neuron_connections_old = connections_dict[pre_neuron]
            #sum_old = 0
            #sum_new = 0
            group = neuron_to_group.get(pre_neuron)
            for n, a in neuron_connections_old.items():
                vn = combined_weights[c]
                vo = a
                c += 1

                v= (-1 if vn < vo else (0 if vn >vo else 0))
                if pre_neuron in neuron_groups['Locomotion-related Interneurons']:
                        loco_neuron_change_neg[pre_neuron] += v
                elif pre_neuron in neuron_groups['Multimodal Sensory Neurons']:
                        modal_neuron_change_neg[pre_neuron] += v

                group_sums_new[group] += v
            
    # Calculate the differences for each group
    group_diffs = {group: (group_sums_new[group]) for group in dist_groups}
    # Prepare data for plotting
    groups = dist_groups
    values = [group_diffs[group] for group in groups]


    # Plot the histogram
    plt.subplot(3,3,9)
    plt.bar(groups, values, edgecolor='black', alpha=0.7)
    plt.xlabel('Distance from Motor Neuron')
    plt.ylabel('Number of Increases or Decreases')
    plt.title('Number of Neuron connection Decreases By distance from motors')





    neuron_to_group = shortest_distances
    dist_groups = np.unique(list(neuron_to_group.values()))
    # Initialize dictionaries to store the sums
    group_sums_new = {group: 0 for group in dist_groups}
    c = 0
    for pre_neuron in connections_dict.keys():
        if pre_neuron[:3] not in muscles:
            neuron_connections_old = connections_dict[pre_neuron]
            #sum_old = 0
            #sum_new = 0
            group = neuron_to_group.get(pre_neuron)
            for n, a in neuron_connections_old.items():
                vn = combined_weights[c]
                vo = a
                c += 1
            
                v= (0 if vn < vo else (1 if vn >vo else 0))
                if pre_neuron in neuron_groups['Locomotion-related Interneurons']:
                        loco_neuron_change_pos[pre_neuron] += v
                elif pre_neuron in neuron_groups['Multimodal Sensory Neurons']:
                        modal_neuron_change_pos[pre_neuron] += v

                group_sums_new[group] += v
            
    # Calculate the differences for each group
    group_diffs = {group: (group_sums_new[group]) for group in dist_groups}

    # Prepare data for plotting
    groups = dist_groups
    values = [group_diffs[group] for group in groups]

    # Plot the histogram
    plt.subplot(3,3,9)
    #plt.ylim= [-10,10]
    plt.bar(groups, values, edgecolor='black', alpha=0.7)
 
    plt.xlabel('Distance from Motor Neuron')
    plt.ylabel('Number of Increases or Decreases ')
    plt.title('Number of Increases by Distance from Motors')
    plt.ylim(-10, 10) 


    plot_loco_neuron_changes(plt.subplot(3,3,8),loco_neuron_change_pos,"Locomotion-related Neuron Changes","orange")
    plot_loco_neuron_changes(plt.subplot(3,3,8),loco_neuron_change_neg,"Locomotion-related Neuron Changes","skyblue")
    plot_loco_neuron_changes(plt.subplot(3,3,7),modal_neuron_change_pos,"Modal-related Neuron Changes","orange")
    plot_loco_neuron_changes(plt.subplot(3,3,7),modal_neuron_change_neg,"Modal-related Neuron Changes","skyblue")

    # Distribution of Weights Subplot
    #ax6 = plt.subplot(3, 3, 6)
    #plot_weight_distribution(ax6,combined_weights,old_wm)
    def Cululative_gains_plot(ax,shortest_distances,pos,t):
        neuron_to_group = {}
        for group, neurons in neuron_groups.items():
            for neuron in neurons:
                neuron_to_group[neuron] = group
        group_sums_new = {group: 0 for group in neuron_groups.keys()}
        c = 0
        for pre_neuron in connections_dict.keys():
            if pre_neuron[:3] not in muscles:
                neuron_connections_old = connections_dict[pre_neuron]
                #sum_old = 0
                #sum_new = 0
                group = neuron_to_group.get(pre_neuron)
                for n, a in neuron_connections_old.items():
                    vn = combined_weights[c]
                    vo = a
                    c += 1                
                    group_sums_new[group] += (pos-1 if vn < vo else (pos  if vn >vo else 0))
        # Calculate the differences for each group
        group_diffs = {group: (group_sums_new[group]) for group in neuron_groups.keys()}
        # Prepare data for plotting
        groups = list(neuron_groups.keys())
        values = [group_diffs[group] for group in groups]

        # Plot the histogram
        if pos: ax.set_ylim([0,10]) 
        else: ax.set_ylim([-10,0])
        ax.bar(groups, values, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Neuron Group')
        ax.set_ylabel(t)
        ax.set_title(t)
        plt.xticks(rotation=90)
    Cululative_gains_plot(plt.subplot(3,3,5),shortest_distances,1,"Number of Increases")
    Cululative_gains_plot(plt.subplot(3,3,6),shortest_distances,0,"Number of Decreases")
    plt.tight_layout()
    #plt.show()
    filename = f'/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/tmp_img/weight_matrix_generation_{10000+generation}.png'
    plt.savefig(filename)
    plt.close()
    del square_weight_matrix, weight_matrix_df, sorted_matrix_df




def graph_comparison(combined_weights1, combined_weights2, connections_dict,generation):
    # Create a list of neuron labels from connections_dict
    neuron_labels = list(all_neuron_names)

    # Calculate the size of the square weight matrix
    matrix_size = len(neuron_labels)
    
    # Function to reshape combined_weights into a square matrix
    def create_weight_matrix(combined_weights):
        square_weight_matrix = np.zeros((len(connections_dict.keys()), matrix_size))
        index = 0
        for i, pre_neuron in enumerate(connections_dict.keys()):
            connections = connections_dict[pre_neuron]
            for j, post_neuron in enumerate(neuron_labels):
                if post_neuron in connections:
                    square_weight_matrix[i, j] = combined_weights[index]
                    index += 1
        return square_weight_matrix
    

    
    # Plotting weight matrix differences
    plt.figure(figsize=(18, 15))

    def plot_weight_matrix(ax, matrix_df, title):
        c = ax.pcolormesh(matrix_df, cmap='magma', vmin=0, vmax=10)
        ax.set_title(title)
        ax.set_xlabel('Post Neurons')
        ax.set_ylabel('Pre Neurons')
        ax.set_ylim()
        ax.invert_yaxis()
        plt.colorbar(c, ax=ax, label='Absolute Weight Difference')






    def get_weight_sum_by_connections(combined_weights):
        # Calculate total weight sum
        tot = 0
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                tot += (np.abs(np.sum(np.abs(weights))) * n)
        
        # Calculate average weight sums divided by the number of connections
        post_neurons = []
        avg_weight_sums = []
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                weight_product = np.sum(np.abs(weights)) * n
                post_neurons.append(post_neuron)
                avg_weight_sums.append((weight_product / tot))

        return avg_weight_sums,post_neurons


    def plot_dif_avg_matrix(ax, dif_avg,post1, title):
        ax.bar(post1, dif_avg, color='skyblue')
        ax.set_xlabel('Post Neurons')
        ax.set_ylabel('Normalized Weight Quotient')
        plt.ylim([-0.005, 0.005])
        ax.set_title(title)

    def get_avg_weight_by_connections(combined_weights):

        tot = 0
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                tot += (np.abs(np.sum(np.abs(weights)) / n))
        
        # Calculate average weight sums divided by the number of connections
        post_neurons = []
        avg_weight_sums = []
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                weight_product = np.sum(np.abs(weights)) / n
                post_neurons.append(post_neuron)
                avg_weight_sums.append((weight_product / tot))
        
        return avg_weight_sums,post_neurons


    def plot_avg_weight_by_connections(ax, avg_weight_sums,post_neurons,title):
        ax.bar(post_neurons, avg_weight_sums, color='skyblue')
        ax.set_xlabel('Post Neurons')
        ax.set_ylabel('Normalized Weight Quotient')
        ax.set_ylim([-0.005, 0.005])
        ax.set_title(title)
        #ax.xticks(rotation=90)  # Rotate x-axis labels for better readability if needed

    def calculate_n_weight_sum(combined_weights):
        n_values = []
        weight_sum_values = []
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                weight_sum = np.abs(np.sum(weights))
                n_values.append(n)
                weight_sum_values.append(weight_sum / n)
        return n_values, weight_sum_values
    

    def plot_n_weight_sum_correlation(ax, n_values, weight_sum_values):

        ax.scatter(n_values, weight_sum_values, color='purple')
        ax.set_xlabel('Number of Connections (n)')
        ax.set_ylabel('Weight Sum / Number of Connections')
        ax.set_ylim([-5,5])
        ax.set_title('Correlation between Weight Sum / n and Number of Connections')


    def calculate_weight_percentage_by_pre_neurons(combined_weights):
        weights_by_pre_neurons = {}

        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            num_pre_neurons = len(pre_neurons)
            if num_pre_neurons > 0:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                if num_pre_neurons not in weights_by_pre_neurons:
                    weights_by_pre_neurons[num_pre_neurons] = []
                weights_by_pre_neurons[num_pre_neurons].extend(weights)

        num_pre_neurons_array = np.array(list(weights_by_pre_neurons.keys()))

        # Automatically determine bin edges using numpy's histogram function
        num_bins = 10  # You can adjust the number of bins if needed
        bin_edges = np.linspace(num_pre_neurons_array.min(), num_pre_neurons_array.max(), num_bins + 1)

        # Initialize a dictionary to hold weights for each bin
        weights_by_bin = {edge: [] for edge in bin_edges}

        # Assign weights to the appropriate bins based on the number of pre-neurons
        for num_pre_neurons, weights in weights_by_pre_neurons.items():
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= num_pre_neurons < bin_edges[i + 1]:
                    weights_by_bin[bin_edges[i]].extend(weights)
                    break

        # Calculate the percentage of weights greater than the 95th percentile for each bin
        percentages = []
        bin_labels = []

        for i in range(len(bin_edges) - 1):
            lower_edge = bin_edges[i]
            upper_edge = bin_edges[i + 1]
            weights = weights_by_bin[lower_edge]
            if weights:
                # Determine the 95th percentile value
                percentile_95 = np.percentile(np.abs(weights), 95)
                # Count weights greater than the 95th percentile value
                count_above_95 = sum(np.abs(w) > percentile_95 for w in weights)
                percentage = count_above_95 / len(weights)  # Percentage as a fraction (out of 1)
            else:
                percentage = 0
            percentages.append(percentage)
            bin_labels.append(f'{int(lower_edge)}-{int(upper_edge)}')

        return bin_labels, percentages



    def plot_weight_percentage_distribution(ax, bin_labels, percentages):
        ax.bar(bin_labels, percentages, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Number of Pre-Neurons (Bin Ranges)')
        ax.set_ylim([-0.01,0.01])
        ax.set_ylabel('Percentage of Weights in 9%th Percentile')
        ax.set_title('Percentage of Weights in 95%th Percentile by Number of Pre-Neurons')
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability if needed

    def plot_weight_distribution(ax, weight_matrix1, weight_matrix2, num_bins=30):
        non_zero_weights1 = weight_matrix1[weight_matrix1 != 0]
        non_zero_weights2 = weight_matrix2[weight_matrix2 != 0]

        # Set x-axis limits
        xlim = [-40, 40]

        # Generate evenly spaced bins over the specified xlim range
        bins = np.linspace(xlim[0], xlim[1], num_bins + 1)

        # Calculate histogram data
        hist1, _ = np.histogram(non_zero_weights1, bins=bins)
        hist2, _ = np.histogram(non_zero_weights2, bins=bins)
        
        # Calculate the difference between the histograms
        hist_diff = hist1 - hist2

        # Plotting histogram difference using ax.bar
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, hist_diff, width=np.diff(bins), color='lightcoral', edgecolor='black')

        ax.set_ylim([-150, 150])
        ax.set_xlim(xlim)
        
        ax.set_xlabel('Weight')
        ax.set_ylabel('Difference in Frequency')
        ax.set_title('Difference in Weight Distribution (Non-zero)')
    


    weight_matrix1 = create_weight_matrix(combined_weights1)
    weight_matrix2 = create_weight_matrix(combined_weights2)
    difference_matrix = np.maximum(np.abs(weight_matrix1) - np.abs(weight_matrix2), 0)
    difference_matrix_df = pd.DataFrame(difference_matrix, index=list(connections_dict.keys()), columns=neuron_labels)
    max_weight_diff = np.max(np.abs(difference_matrix_df.values))
    vmin = 0
    vmax = max_weight_diff
    ax1 = plt.subplot(3, 3, 1)
    plot_weight_matrix(ax1, difference_matrix_df, 'Absolute Differences in Weight Matrices')

    val1,post1 = get_weight_sum_by_connections(combined_weights1)
    val2,_ = get_weight_sum_by_connections(combined_weights2)
    dif_asum = np.array(val1)-np.array(val2)
    ax2 = plt.subplot(3, 3, 2)
    plot_dif_avg_matrix(ax2,dif_asum,post1,'Weight Sum * Number of Connections')

    val12,post21 = get_avg_weight_by_connections(combined_weights1)
    val22,_ = get_avg_weight_by_connections(combined_weights2)
    
    dif_avg = np.array(val12)-np.array(val22)
    ax3 = plt.subplot(3, 3, 3)
    plot_avg_weight_by_connections(ax3,dif_avg,post21,'Weight Sum / Number of Connections')

    n_values, weight_sum_values1 = calculate_n_weight_sum(combined_weights1)
    n_values, weight_sum_values2 = calculate_n_weight_sum(combined_weights2)
    ax4 = plt.subplot(3, 3, 4)
    # Plot the results using the provided ax
    plot_n_weight_sum_correlation(ax4, n_values, np.array(weight_sum_values1)-np.array(weight_sum_values2))
    bin, per1 = calculate_weight_percentage_by_pre_neurons(combined_weights1)
    bin, per2 = calculate_weight_percentage_by_pre_neurons(combined_weights2)
    ax5 = plt.subplot(3, 3, 5)
    # Plot the results using the provided ax
    plot_weight_percentage_distribution(ax5, bin, np.array(per1)-np.array(per2))

    ax6 = plt.subplot(3, 3, 6)
    plot_weight_distribution(ax6,combined_weights1,combined_weights2)



    plt.tight_layout(h_pad=10)  # Adjust layout to prevent overlap
    plt.show()
    #filename = f'/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/tmp_img/weight_matrix_generation_{10000+generation}.png'
    #plt.savefig(filename)
    #plt.close()
    del difference_matrix_df,weight_matrix1,weight_matrix2,val1,val12,val2,val22,per1,per2,bin
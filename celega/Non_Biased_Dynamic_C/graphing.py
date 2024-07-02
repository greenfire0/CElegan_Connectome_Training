import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from trained_connectome import all_neuron_names

def graph(combined_weights, connections_dict, generation):
    # Create a list of neuron labels from connections_dict
    neuron_labels = list(all_neuron_names)

    # Calculate the size of the square weight matrix
    matrix_size = len(neuron_labels)
    
    # Reshape combined_weights into a square matrix
    square_weight_matrix = np.zeros((len(connections_dict.keys()), matrix_size))
    index = 0
    for i, pre_neuron in enumerate(connections_dict.keys()):
        connections = connections_dict[pre_neuron]
        for j, post_neuron in enumerate(neuron_labels):
            if post_neuron in connections:
                square_weight_matrix[i, j] = combined_weights[index]
                index += 1
    
    # Create a DataFrame from the square_weight_matrix and neuron_labels
    weight_matrix_df = pd.DataFrame(square_weight_matrix, index=list(connections_dict.keys()), columns=neuron_labels)
    
    # Sort neurons by the sum of weights
    row_weights = weight_matrix_df.sum(axis=1)
    sorted_row_indices = np.argsort(row_weights)[::-1]
    sorted_matrix_df = weight_matrix_df.iloc[sorted_row_indices]
        
    max_weight = np.max(np.abs(sorted_matrix_df.values))
    vmin = -max_weight
    vmax = max_weight
    
    # Plotting weight matrix
    plt.figure(figsize=(18, 15))  # Adjust figure size as needed
    
    # Subplot for weight matrix
    plt.subplot(3, 3, 1)
    c = plt.pcolormesh(sorted_matrix_df, cmap='twilight', vmin=vmin, vmax=vmax)
    plt.title('Weight Matrix')
    plt.xlabel('Post Neurons')
    plt.ylabel('Pre Neurons')
    plt.gca().invert_yaxis()
    plt.colorbar(c, label='Weight')
    
    # Plotting average weight sum divided by connections
    tot = 0
    for post_neuron in all_neuron_names:
        pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
        if pre_neurons:
            indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
            weights = combined_weights[indices]
            n = len(pre_neurons)
            tot += (np.abs(np.sum(np.abs(weights)) * n))
    
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
    
    # Subplot for average weight sum divided by connections
    plt.subplot(3, 3, 2)
    plt.bar(post_neurons, avg_weight_sums, color='skyblue')
    plt.xlabel('Post Neurons')
    plt.ylabel('Normalized Weight Product')
    plt.ylim= [0, 0.1]
    plt.title('Weight Sum * Number of Connections')
    
    # Plotting average weight sum divided by connections
    tot = 0
    for post_neuron in all_neuron_names:
        pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
        if pre_neurons:
            indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
            weights = combined_weights[indices]
            n = len(pre_neurons)
            tot += (np.abs(np.sum(np.abs(weights)) / n))
    
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
    
    # Subplot for average weight quotient
    plt.subplot(3, 3, 3)
    plt.bar(post_neurons, avg_weight_sums, color='skyblue')
    plt.xlabel('Post Neurons')
    plt.ylabel('Normalized Weight Quotient')
    plt.ylim= [0, 0.01]
    plt.title('Weight Sum / Number of Connections')

    # Calculate n * sum(w) and n for each post-neuron
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
    
    # Subplot for correlation between n * sum(w) and n
    plt.subplot(3, 3, 4)
    plt.scatter(n_values, weight_sum_values, color='purple')
    plt.xlabel('Number of Connections (n)')
    plt.ylabel('n / Sum of Weights')
    plt.title('Correlation between n / Sum(w) and n')
    
    # Subplot for distribution of weights
    # Subplot for distribution of weights
    
    
    # Subplot for distribution of weights
     # Reshape combined_weights into a square matrix
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

    num_pre_neurons_list = list(weights_by_pre_neurons.keys())
    num_pre_neurons_array = np.array(num_pre_neurons_list)

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

    # Calculate percentage of weights > 90 for each bin
    percentages = []
    bin_labels = []

    for i in range(len(bin_edges) - 1):
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i + 1]
        weights = weights_by_bin[lower_edge]
        if weights:
            count_above_90 = sum(w > 90 for w in weights)
            percentage = count_above_90 / len(weights)  # Percentage as a fraction (out of 1)
        else:
            percentage = 0
        percentages.append(percentage)
        bin_labels.append(f'{int(lower_edge)}-{int(upper_edge)}')

    # Plotting
    plt.subplot(3, 3, 5)
    plt.bar(bin_labels, percentages, color='lightcoral', edgecolor='black')
    plt.xlabel('Number of Pre-Neurons (Bin Ranges)')
    plt.ylabel('Percentage of Weights > 90')
    plt.title('Percentage of Weights Greater Than 90 by Number of Pre-Neurons')
    plt.xticks(rotation=45)  # Rotate x-ticks if necessary for readability




    plt.subplot(3, 3, 6)
    # Flatten the weight matrix values
    flattened_weights = weight_matrix_df.values.flatten()
    
    # Filter out zero values
    non_zero_weights = flattened_weights[flattened_weights != 0]
    
    # Calculate histogram data
    num_bins = 30
    hist, bins = np.histogram(non_zero_weights, bins=num_bins)
    
    # Plotting histogram using plt.bar
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, hist, width=np.diff(bins), color='lightcoral', edgecolor='black')
    
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Weights (Non-zero)')


    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    #plt.show()
    
    filename = f'mtp/weight_matrix_generation_{10000+generation}.png'
    plt.savefig(filename)
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from trained_connectome import all_neuron_names
import numpy as np


def graph(combined_weights, connections_dict):
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
        
    # Plotting weight matrix
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    
    # Subplot for weight matrix
    plt.subplot(1, 2, 1)
    plt.pcolormesh(sorted_matrix_df, cmap='twilight')
    plt.title('Weight Matrix')
    plt.xlabel('Post Neurons')
    plt.ylabel('Pre Neurons')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Weight')
    
    # Plotting average weight sum divided by connections
    post_neurons = []
    avg_weight_sums = []
    
    # Calculate average weight sums
    for post_neuron in all_neuron_names:
        pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
        if pre_neurons:
            indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
            weights = combined_weights[indices]
            avg_weight_sum = np.sum(np.abs(weights)) * len(pre_neurons)
            post_neurons.append(post_neuron)
            avg_weight_sums.append(avg_weight_sum)
    
    # Subplot for average weight sum divided by connections
    plt.subplot(1, 2, 2)
    plt.bar(post_neurons, avg_weight_sums, color='skyblue')
    plt.xlabel('Weight Sum * Number of Connections')
    plt.ylabel('Post Neurons')
    plt.title('Weight Sum Times by Number of Connections for Post Neurons')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def graph_weight_sum_divided_by_connections(combined_weights, connections_dict):
    # Initialize lists to store data
    post_neurons = []
    avg_weight_sums = []
    pre_neuron = []

    # Iterate over each post neuron
    for post_neuron in all_neuron_names:
        # Find all pre_neurons connected to this post_neuron
        pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
        
        if pre_neurons:
            indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
            weights = combined_weights[indices]
            avg_weight_sum = np.sum(np.abs(weights)) * len(pre_neurons)  # Calculate average weight sum divided by number of connections
            post_neurons.append(post_neuron)
            avg_weight_sums.append(avg_weight_sum)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(post_neurons,avg_weight_sums,  color='skyblue')
    plt.ylabel('Average Weight Sum * Number of Connections')
    plt.xlabel('Post Neurons')
    plt.title('Average Weight Sum Times by Number of Connections for Post Neurons')
    plt.show()
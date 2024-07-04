import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Worm_Env.trained_connectome import all_neuron_names
##gabriel
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

    max_weight=30
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

    # Calculate percentage of weights > 90 for each bin
    percentages = []
    bin_labels = []

    for i in range(len(bin_edges) - 1):
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i + 1]
        weights = weights_by_bin[lower_edge]
        if weights:
            count_above_90 = sum(np.abs(w) > 15 for w in weights)  # Changed threshold to 90 as per your description
            percentage = count_above_90 / len(weights)  # Percentage as a fraction (out of 1)
        else:
            percentage = 0
        percentages.append(percentage)
        bin_labels.append(f'{int(lower_edge)}-{int(upper_edge)}')

    # Plotting
    plt.subplot(3, 3, 5)
    plt.bar(bin_labels, percentages, color='lightcoral', edgecolor='black')
    plt.xlabel('Number of Pre-Neurons (Bin Ranges)')
    plt.ylabel('Percentage of Weights > 20')
    plt.title('Percentage of Weights Greater Than 20 by Number of Pre-Neurons')
    plt.xticks(rotation=45, ha='right') 




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
    plt.xlim([vmin, vmax])
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Weights (Non-zero)')


    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    #plt.show()
    
    filename = f'/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/tmp_img/weight_matrix_generation_{10000+generation}.png'
    plt.savefig(filename)
    plt.close()
    del square_weight_matrix, weight_matrix_df, sorted_matrix_df, flattened_weights, non_zero_weights




def graph_comparison(combined_weights1, combined_weights2, connections_dict, generation):
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
    
    weight_matrix1 = create_weight_matrix(combined_weights1)
    weight_matrix2 = create_weight_matrix(combined_weights2)
    
    # Create DataFrames from the weight matrices and neuron_labels
    weight_matrix_df1 = pd.DataFrame(weight_matrix1, index=list(connections_dict.keys()), columns=neuron_labels)
    weight_matrix_df2 = pd.DataFrame(weight_matrix2, index=list(connections_dict.keys()), columns=neuron_labels)
    
    # Sort neurons by the sum of weights
    def sort_matrix(weight_matrix_df):
        row_weights = weight_matrix_df.sum(axis=1)
        sorted_row_indices = np.argsort(row_weights)[::-1]
        return weight_matrix_df.iloc[sorted_row_indices]
    
    sorted_matrix_df1 = sort_matrix(weight_matrix_df1)
    sorted_matrix_df2 = sort_matrix(weight_matrix_df2)
    
    max_weight1 = np.max(np.abs(sorted_matrix_df1.values))
    max_weight2 = np.max(np.abs(sorted_matrix_df2.values))
    
    max_weight = max(max_weight1, max_weight2)
    max_weight = 30  # Adjust as needed
    vmin = -max_weight
    vmax = max_weight
    
    # Plotting weight matrix comparison
    plt.figure(figsize=(30, 15))  # Adjust figure size as needed
    
    def plot_weight_matrix(ax, matrix_df, title):
        c = ax.pcolormesh(matrix_df, cmap='twilight', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Post Neurons')
        ax.set_ylabel('Pre Neurons')
        ax.invert_yaxis()
        plt.colorbar(c, ax=ax, label='Weight')
    
    ax1 = plt.subplot(3, 6, 1)
    plot_weight_matrix(ax1, sorted_matrix_df1, 'Weight Matrix 1')
    
    ax2 = plt.subplot(3, 6, 2)
    plot_weight_matrix(ax2, sorted_matrix_df2, 'Weight Matrix 2')
    
    # Plotting average weight sum divided by connections
    def calculate_avg_weight_sums(combined_weights, tot_factor):
        tot = 0
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                tot += (np.abs(np.sum(np.abs(weights)) * n / tot_factor))
        
        post_neurons = []
        avg_weight_sums = []
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                weight_product = np.sum(np.abs(weights)) * n / tot_factor
                post_neurons.append(post_neuron)
                avg_weight_sums.append(weight_product / tot)
        return post_neurons, avg_weight_sums
    
    post_neurons1, avg_weight_sums1 = calculate_avg_weight_sums(combined_weights1, tot_factor=1)
    post_neurons2, avg_weight_sums2 = calculate_avg_weight_sums(combined_weights2, tot_factor=1)
    
    def plot_avg_weight_sums(ax, post_neurons, avg_weight_sums, title, ylabel, ylim):
        ax.bar(post_neurons, avg_weight_sums, color='skyblue')
        ax.set_xlabel('Post Neurons')
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_title(title)
    
    ax3 = plt.subplot(3, 6, 3)
    plot_avg_weight_sums(ax3, post_neurons1, avg_weight_sums1, 'Weight Sum * Number of Connections 1', 'Normalized Weight Product', [0, 0.1])
    
    ax4 = plt.subplot(3, 6, 4)
    plot_avg_weight_sums(ax4, post_neurons2, avg_weight_sums2, 'Weight Sum * Number of Connections 2', 'Normalized Weight Product', [0, 0.1])
    
    # Plotting average weight quotient
    def calculate_avg_weight_quotients(combined_weights, tot_factor):
        tot = 0
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                tot += (np.abs(np.sum(np.abs(weights)) / n / tot_factor))
        
        post_neurons = []
        avg_weight_sums = []
        for post_neuron in all_neuron_names:
            pre_neurons = [pre_neuron for pre_neuron, post_neurs in connections_dict.items() if post_neuron in post_neurs]
            if pre_neurons:
                indices = [all_neuron_names.index(pre_neuron) for pre_neuron in pre_neurons]
                weights = combined_weights[indices]
                n = len(pre_neurons)
                weight_product = np.sum(np.abs(weights)) / n / tot_factor
                post_neurons.append(post_neuron)
                avg_weight_sums.append(weight_product / tot)
        return post_neurons, avg_weight_sums
    
    post_neurons1, avg_weight_quotients1 = calculate_avg_weight_quotients(combined_weights1, tot_factor=1)
    post_neurons2, avg_weight_quotients2 = calculate_avg_weight_quotients(combined_weights2, tot_factor=1)
    
    ax5 = plt.subplot(3, 6, 5)
    plot_avg_weight_sums(ax5, post_neurons1, avg_weight_quotients1, 'Weight Sum / Number of Connections 1', 'Normalized Weight Quotient', [0, 0.01])
    
    ax6 = plt.subplot(3, 6, 6)
    plot_avg_weight_sums(ax6, post_neurons2, avg_weight_quotients2, 'Weight Sum / Number of Connections 2', 'Normalized Weight Quotient', [0, 0.01])
    
    # Calculate n * sum(w) and n for each post-neuron
    def calculate_n_sum_weights(combined_weights):
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
    
    n_values1, weight_sum_values1 = calculate_n_sum_weights(combined_weights1)
    n_values2, weight_sum_values2 = calculate_n_sum_weights(combined_weights2)
    
    def plot_n_sum_weights(ax, n_values, weight_sum_values, title):
        ax.scatter(n_values, weight_sum_values, color='purple')
        ax.set_xlabel('Number of Connections (n)')
        ax.set_ylabel('n / Sum of Weights')
        ax.set_title(title)
    
    ax7 = plt.subplot(3, 6, 7)
    plot_n_sum_weights(ax7, n_values1, weight_sum_values1, 'Correlation between n / Sum(w) and n 1')
    
    ax8 = plt.subplot(3, 6, 8)
    plot_n_sum_weights(ax8, n_values2, weight_sum_values2, 'Correlation between n / Sum(w) and n 2')
    
    # Plotting distribution of weights
    def plot_distribution_of_weights(ax, weight_matrix_df, title):
        # Flatten the weight matrix values
        flattened_weights = weight_matrix_df.values.flatten()
        
        # Filter out zero values
        non_zero_weights = flattened_weights[flattened_weights != 0]
        
        # Calculate histogram data
        num_bins = 30
        hist, bins = np.histogram(non_zero_weights, bins=num_bins)
        
        # Plotting histogram using plt.bar
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, hist, width=np.diff(bins), color='lightcoral', edgecolor='black')
        ax.set_xlim([vmin, vmax])
        ax.set_xlabel('Weight')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
    
    ax9 = plt.subplot(3, 6, 9)
    plot_distribution_of_weights(ax9, weight_matrix_df1, 'Distribution of Weights (Non-zero) 1')
    
    ax10 = plt.subplot(3, 6, 10)
    plot_distribution_of_weights(ax10, weight_matrix_df2, 'Distribution of Weights (Non-zero) 2')
    
    # Plotting percentage of weights greater than threshold
    def calculate_percentage_weights_greater_than_threshold(combined_weights, threshold):
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
    
        # Calculate percentage of weights > threshold for each bin
        percentages = []
        bin_labels = []
    
        for i in range(len(bin_edges) - 1):
            lower_edge = bin_edges[i]
            upper_edge = bin_edges[i + 1]
            weights = weights_by_bin[lower_edge]
            if weights:
                count_above_threshold = sum(np.abs(w) > threshold for w in weights)  # Changed threshold to 90 as per your description
                percentage = count_above_threshold / len(weights)  # Percentage as a fraction (out of 1)
            else:
                percentage = 0
            percentages.append(percentage)
            bin_labels.append(f'{int(lower_edge)}-{int(upper_edge)}')
        return bin_labels, percentages
    
    threshold = 20  # Adjust as needed
    bin_labels1, percentages1 = calculate_percentage_weights_greater_than_threshold(combined_weights1, threshold)
    bin_labels2, percentages2 = calculate_percentage_weights_greater_than_threshold(combined_weights2, threshold)
    
    def plot_percentage_weights_greater_than_threshold(ax, bin_labels, percentages, title):
        ax.bar(bin_labels, percentages, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Number of Pre-Neurons (Bin Ranges)')
        ax.set_ylabel('Percentage of Weights > 20')
        ax.set_title(title)
        ax.set_xticks(rotation=45, ha='right')
    
    ax11 = plt.subplot(3, 6, 11)
    plot_percentage_weights_greater_than_threshold(ax11, bin_labels1, percentages1, 'Percentage of Weights > 20 by Pre-Neurons 1')
    
    ax12 = plt.subplot(3, 6, 12)
    plot_percentage_weights_greater_than_threshold(ax12, bin_labels2, percentages2, 'Percentage of Weights > 20 by Pre-Neurons 2')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
   
    plt.show()
    filename = f'/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/tmp_img/weight_matrix_comparison_generation_{10000+generation}.png'
    plt.savefig(filename)
    plt.close()
    del weight_matrix1, weight_matrix2, weight_matrix_df1, weight_matrix_df2, sorted_matrix_df1, sorted_matrix_df2

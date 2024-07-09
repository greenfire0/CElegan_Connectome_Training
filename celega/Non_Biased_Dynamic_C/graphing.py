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
    #plt.show()
    filename = f'/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/tmp_img/weight_matrix_generation_{10000+generation}.png'
    plt.savefig(filename)
    plt.close()
    del difference_matrix_df,weight_matrix1,weight_matrix2,val1,val12,val2,val22,per1,per2,bin
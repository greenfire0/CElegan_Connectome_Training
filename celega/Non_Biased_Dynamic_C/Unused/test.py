import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_samples = 100000  # Number of samples
num_intervals = 10    # Number of uniform intervals
interval_length = 1.0  # Length of each interval

# Initialize an array to collect the sums
collected_data = np.zeros(num_samples)

# Loop to generate and add uniform distribution arrays from different intervals
for _ in range(num_samples):
    # Choose a random interval for each sample
    interval_index = np.random.randint(num_intervals)
    interval_start = interval_index * interval_length
    interval_end = (interval_index + 1) * interval_length
    
    # Generate uniform data within the chosen interval
    uniform_data = np.random.uniform(low=interval_start, high=interval_end)
    collected_data[_] = uniform_data

# Compute the histogram data
hist, bin_edges = np.histogram(collected_data, bins=100, density=True)

# Calculate the width of each bin
bin_width = bin_edges[1] - bin_edges[0]

# Plotting the distribution of the collected data
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=bin_width, alpha=0.6, color='g')
plt.title('Histogram of Uniform Distribution Over Multiple Intervals')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()

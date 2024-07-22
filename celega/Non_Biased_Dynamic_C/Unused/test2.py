import numpy as np

# Define the probability and size of the probability test array
prob = 0.1
size = 1000000

# Generate the probability test array
prob_array = (np.random.rand(size) < prob).astype(int)

# Define two arrays of size 50
array1 = np.arange(1, 51)  # Example array with values from 1 to 50
array2 = np.arange(51, 101)  # Example array with values from 51 to 100

# Generate the final array based on the probability test array
# Reshape the probability array to the desired shape
prob_array_small = (np.random.rand(50) < prob).astype(int)

# Use np.where to select values from array1 or array2 based on prob_array_small
final_array = np.where(prob_array_small, array1, array2)

print(final_array)
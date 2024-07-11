import matplotlib.pyplot as plt
import numpy as np

# Original matrix
original_matrix = np.array([2.0, 1.0, 3.0, 5.0, 1.0, 3.0, 1.0, 1.0, 7.0, 8.0, 1.0, 2.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
updated_matrix = np.array([2.0, 1.0, 3.0, 5.0, 1.0, 3.0, 1.0, 1.0, 7.0, 8.0, 1.0, 2.0, 1.0, 1.0, -4.227098498217728, 1.0, 1.0, 1.0, 5.0, 15.0, 1.0, 2.0])
print(len(original_matrix),len(updated_matrix))
# Ensure both arrays have the same length
if len(original_matrix) != len(updated_matrix):
    raise ValueError("The original and updated matrices must be of the same length.")

# Convert arrays to numpy arrays if they are not already
original_matrix = np.asarray(original_matrix)
updated_matrix = np.asarray(updated_matrix)

# Identify transitions
neg_to_pos = np.sum((original_matrix < 0) & (updated_matrix >= 0))
pos_to_neg = np.sum((original_matrix >= 0) & (updated_matrix < 0))

# Plotting the results
labels = ['Negative to Positive', 'Positive to Negative']
counts = [neg_to_pos, pos_to_neg]

plt.bar(labels, counts, color=['blue', 'red'])
plt.xlabel('Transition Type')
plt.ylabel('Count')
plt.title('Count of Transitions Between Negative and Positive Values')
plt.show()
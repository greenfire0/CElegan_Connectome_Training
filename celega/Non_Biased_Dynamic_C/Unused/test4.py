import networkx as nx
import matplotlib.pyplot as plt
from Worm_Env.weight_dict import dict  # Import the dictionary

# Define the dictionary
neuron_dict = dict

# Motor neuron prefixes
motor_prefixes = ['MVU', 'MVL', 'MDL', 'MVR', 'MDR']

# Create an unweighted graph
G = nx.Graph()

# Add nodes and edges from the dictionary
for node, neighbors in neuron_dict.items():
    for neighbor in neighbors.keys():
        G.add_edge(node, neighbor)

# Identify motor neurons
motor_neurons = [node for node in G.nodes if any(node.startswith(prefix) for prefix in motor_prefixes)]

# Compute shortest path lengths from each neuron to the nearest motor neuron
shortest_distances = {}
for neuron in G.nodes:
    # Use BFS to find the shortest path from neuron to any motor neuron
    try:
        length = nx.single_source_shortest_path_length(G, neuron)
        # Find the minimum distance to any motor neuron
        min_distance = min(length.get(motor_neuron, float('inf')) for motor_neuron in motor_neurons)
        shortest_distances[neuron] = min_distance
    except nx.NetworkXNoPath:
        shortest_distances[neuron] = float('inf')

# Create a list of distances (excluding 'inf' values)
distances = [dist for dist in shortest_distances.values() if dist != float('inf')]
print(distances)
# Plot histogram of distances
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=range(int(max(distances)) + 2), edgecolor='black', alpha=0.7)
plt.xlabel('Distance to Nearest Motor Neuron')
plt.ylabel('Frequency')
plt.title('Histogram of Distances to Nearest Motor Neuron')
plt.grid(True)
plt.show()

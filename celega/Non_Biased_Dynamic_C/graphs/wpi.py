import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles, muscleList, mLeft, mRight, all_neuron_names
from tqdm import tqdm
import matplotlib.pyplot as plt

def zero_out_connection(weights, index):
    """
    Create a new weight matrix with the specified connection set to zero.
    """
    modified_weights = np.copy(weights)
    modified_weights[index] = 0
    return modified_weights

def evaluate_fitness(candidate_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes):
    """
    Evaluate the fitness of a weight matrix.
    """
    sum_rewards = 0
    for pattern in prob_type:
        candidate = WormConnectome(weight_matrix=candidate_weights, all_neuron_names=nur_name)
        env.reset(pattern)
        for _ in range(episodes):  # Total episodes
            observation = env._get_observations()
            for _ in range(interval):  # Training interval
                movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)
                next_observation, reward, _ = env.step(movement, 0, candidate)
                observation = next_observation
                sum_rewards += reward
    return sum_rewards

@ray.remote
def evaluate_connection_zeroed(index, original_weights, baseline_fitness, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes):
    """
    Evaluate the performance impact of zeroing out a specific connection.
    """
    zeroed_weights = zero_out_connection(original_weights, index)
    zeroed_fitness = evaluate_fitness(zeroed_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes)
    fitness_impact = baseline_fitness - zeroed_fitness
    return index, fitness_impact

def search_connection_impacts(original_genome, matrix_shape, env, prob_type, interval, episodes, batch_size=32):
    """
    Perform a parallelized search for connection impacts.

    Args:
        original_genome (np.ndarray): The original weight matrix.
        matrix_shape (int): Total number of connections in the matrix.
        env: Worm environment instance.
        prob_type (list): List of food patterns for evaluation.
        interval (int): Training interval for the simulation.
        episodes (int): Number of episodes for the simulation.
        batch_size (int): Batch size for parallelization.

    Returns:
        sorted_impacts (list): List of tuples (index, fitness impact), sorted by impact.
    """
    # Calculate the baseline fitness once
    baseline_fitness = evaluate_fitness(
        original_genome, all_neuron_names, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes
    )

    # Divide indices into batches
    indices = np.arange(matrix_shape)
    index_batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    impacts_array = np.zeros(matrix_shape)

    # Process batches in parallel
    for batch in tqdm(index_batches, desc="Processing Connection Batches"):
        futures = [
            evaluate_connection_zeroed.remote(
                index, original_genome, baseline_fitness, all_neuron_names, env, prob_type,
                mLeft, mRight, muscleList, muscles, interval, episodes
            ) for index in batch
        ]
        results = ray.get(futures)
        for index, impact in results:
            impacts_array[index] = impact  # Fill the impact for the corresponding connection index

    return impacts_array

def graph_wsi(length,arr_3,arr_5,arr_diff):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].bar(np.arange(length), arr_3, color='blue')
    axs[0].set_title("Connection Impact Distribution (prob_type=[3])")
    axs[0].set_xlabel("Connection Index")
    axs[0].set_ylabel("Fitness Impact")
    
    axs[1].bar(np.arange(length), arr_5, color='green')
    axs[1].set_title("Connection Impact Distribution (prob_type=[5])")
    axs[1].set_xlabel("Connection Index")
    axs[1].set_ylabel("Fitness Impact")

    axs[2].bar(np.arange(length), arr_diff, color='red')
    axs[2].set_title("Difference in Impact (prob_type=[5] - prob_type=[3])")
    axs[2].set_xlabel("Connection Index")
    axs[2].set_ylabel("Difference in Fitness Impact")

    plt.savefig("wsi.png")

def calc_simular(array1,array2):
    # Define a threshold for "important connections"
    threshold_3 = np.percentile(array1, 90)  # Top 10% connections for prob_type=[3]
    threshold_5 = np.percentile(array2, 90)  # Top 10% connections for prob_type=[5]

    # Find indices of important connections
    important_3 = set(np.where(array1 >= threshold_3)[0])
    important_5 = set(np.where(array2 >= threshold_5)[0])

    # Calculate intersection and Jaccard similarity
    intersection = important_3.intersection(important_5)
    jaccard_similarity = len(intersection) / len(important_3.union(important_5))

    print(f"Number of important connections in prob_type=[3]: {len(important_3)}")
    print(f"Number of important connections in prob_type=[5]: {len(important_5)}")
    print(f"Number of shared important connections: {len(intersection)}")
    print(f"Jaccard Similarity: {jaccard_similarity:.2f}")

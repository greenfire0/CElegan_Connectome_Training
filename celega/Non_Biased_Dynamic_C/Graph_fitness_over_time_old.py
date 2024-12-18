import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from graphing import graph
from Worm_Env.weight_dict import dict, muscles, muscleList, mLeft, mRight, all_neuron_names
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
import os
from util.write_read_txt import read_arrays_from_csv_pandas
import random
import matplotlib.ticker as ticker


class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[5], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        self.population_size = population_size
        print(pattern)
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = np.array(genome, dtype=float) if genome is not None else np.zeros(matrix_shape)
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, genomes=None):
        if genomes is None:
            raise ValueError("Genomes must be provided to initialize the population.")
        if len(genomes) > 400:
            genomes = genomes[:400]  # Limiting to first 400 genomes if necessary
        for g in genomes:
            self.population.append(WormConnectome(weight_matrix=np.array(g, dtype=float), all_neuron_names=all_neuron_names))

    @staticmethod
    @ray.remote
    def evaluate_fitness_ray(candidate_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes):
        sum_rewards = 0
        for a in prob_type:
            candidate = WormConnectome(weight_matrix=candidate_weights, all_neuron_names=nur_name)
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for _ in range(interval):  # training_interval
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    observation = next_observation
                    sum_rewards += reward
        return sum_rewards

    def calculate_euclidean_distance(self, candidate_weights):
        candidate_weights = np.array(candidate_weights, dtype=float)
        if candidate_weights.shape != self.original_genome.shape:
            raise ValueError("Shape of candidate_weights and original_genome must be the same.")
        distance = np.linalg.norm(candidate_weights - self.original_genome)
        return distance

    def generate_random_color(self):
        return '#%06x' % random.randint(0, 0xFFFFFF)

    def run(self, env, path='Results', batch_size=10, jitter_strength=10):
        ray.init(
            ignore_reinit_error=True,
            object_store_memory=14 * 1024 * 1024 * 1024,
            num_cpus=16,
        )
        folder_path = 'Results_good_sq_nolasso'
        base_dir = os.path.dirname(__file__)
        print(base_dir)
        full_folder_path = os.path.join(base_dir, folder_path)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), sharex=True)
        
        # Set font sizes
        title_fontsize = 24
        label_fontsize = 20
        tick_fontsize = 20

        ax1.set_title('Fitness on Square Food Pattern Over Generations', fontsize=title_fontsize)
        ax1.set_ylabel('Number of Food Eaten', fontsize=label_fontsize)
        ax2.set_ylabel('Euclidean Distance', fontsize=label_fontsize)
        ax2.set_title('Euclidean Distance Over Generations', fontsize=title_fontsize)
        ax2.set_xlabel('Generation', fontsize=label_fontsize)

        # Adjust tick label sizes
        ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        fitnesses_dict = {'blue': [], 'green': [], 'red': [], 'cyan': [],"teal": []}
        distances_dict = {'blue': [], 'green': [], 'red': [], 'cyan': [],"teal": []}

        for filename in os.listdir(full_folder_path):
            self.population = []
            genomes = read_arrays_from_csv_pandas(os.path.join(full_folder_path, filename))
            self.initialize_population(genomes)

            population_batches = [self.population[i:i + batch_size] for i in range(0, len(self.population), batch_size)]
            fitnesses = []
            distances = []
            jitter = 0  # You might want to adjust or remove jitter based on your requirements

            for batch in population_batches:
                # Evaluate fitness in parallel using Ray
                fitnesses.extend(ray.get([
                    self.evaluate_fitness_ray.remote(
                        candidate.weight_matrix,
                        all_neuron_names,
                        env,
                        self.food_patterns,
                        mLeft,
                        mRight,
                        muscleList,
                        muscles,
                        self.training_interval,
                        self.total_episodes
                    ) for worm_num, candidate in enumerate(batch)
                ]))
                # Calculate Euclidean distances
                batch_distances = [
                    self.calculate_euclidean_distance(candidate.weight_matrix) + jitter for candidate in batch
                ]
                distances.extend(batch_distances)

            # Determine color based on filename or distance criteria
            color = "blue"
            if len(distances) > 90:
                print(distances[90])
            if "evo" in filename:
                color = "green"
            if len(distances) > 90 and distances[90] > 500:
                color = "red"
            if "random" in filename:
                color = "cyan"
            if "NO" in filename:
                color = "teal"

            fitnesses_dict[color].append(fitnesses)
            distances_dict[color].append(distances)

            ax1.plot(fitnesses, color=color, alpha=0.3)
            ax2.plot(distances, color=color, alpha=0.3)

        # Plot average lines with full opacity
        for color, fitnesses_list in fitnesses_dict.items():
            if fitnesses_list:
                avg_fitness = np.mean(fitnesses_list, axis=0)
                if color == "blue":
                    model = "Of Nomad Assisted Search"
                elif color == "green":
                    model = "Of Evolutionary Algorithm"
                elif color == "red":
                    model = "Of Searches With Large Differences from the Original Connectome"
                elif color == "cyan":
                    model = "Of Random Searches"
                else:
                    model = "Unknown Model"
                ax1.plot(avg_fitness, color=color, alpha=1, linewidth=2, label=f'Average Performance {model}')

        for color, distances_list in distances_dict.items():
            if distances_list:
                avg_distance = np.mean(distances_list, axis=0)
                if color == "blue":
                    model = "Of Nomad Assisted Search"
                elif color == "green":
                    model = "Of Evolutionary Algorithm"
                elif color == "red":
                    model = "Of Searches With Large Differences from the Original Connectome"
                elif color == "cyan":
                    model = "Of Random Searches"
                else:
                    model = "Unknown Model"

        # Create a combined legend for both subplots
        ax1.set_xscale('log')
        ax2.set_xscale('log')

        #handles1, labels1 = ax1.get_legend_handles_labels()
        #handles2, labels2 = ax2.get_legend_handles_labels()
        #handles = handles1 + handles2
        #labels = labels1 + labels2
        #fig.legend(handles, labels, loc='upper right', fontsize=legend_fontsize)

        plt.savefig("fig7.svg")
        ray.shutdown()
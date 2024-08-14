import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from graphing import graph, graph_comparison
from Worm_Env.weight_dict import dict, muscles, muscleList, mLeft, mRight, all_neuron_names
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
import os
from util.write_read_txt import read_arrays_from_csv_pandas
import random

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[5], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        self.population_size = population_size
        print(pattern)
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, genomes=None):
        if (len(genomes) > 400) and False:
            genomes = genomes[0:400]
        for g in (genomes):
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
                    # env.render(worm_num)
                    observation = next_observation
                    sum_rewards += reward
        return sum_rewards

    def calculate_differences(self, candidate_weights):
        return np.count_nonzero(candidate_weights != self.original_genome)

    def generate_random_color(self):
        return '#%06x' % random.randint(0, 0xFFFFFF)

    def run(self, env, path='Results', batch_size=10, jitter_strength=10):
        ray.init(
            ignore_reinit_error=True,  # Allows reinitialization if Ray is already running
            object_store_memory=14 * 1024 * 1024 * 1024,  # 14 GB in bytes
            num_cpus=16,  # Number of CPU cores
        )
        folder_path = 'Results'
        base_dir = os.path.dirname(__file__)  # Get the directory of the current script
        full_folder_path = os.path.join(base_dir, folder_path)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        ax1.set_title('Fitness Over Generations')
        ax2.set_title('Differences Over Generations')
        ax2.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax2.set_ylabel('Differences')

        for filename in os.listdir(full_folder_path):
            self.population = []
            self.initialize_population(read_arrays_from_csv_pandas(os.path.join(full_folder_path, filename)))

            population_batches = [self.population[i:i + batch_size] for i in range(0, len(self.population), batch_size)]
            fitnesses = []
            differences = []
            jitter = random.uniform(-jitter_strength, jitter_strength)
            ##ask jordi about jitter
            for batch in population_batches:
                fitnesses.extend(ray.get([self.evaluate_fitness_ray.remote(candidate.weight_matrix, all_neuron_names, env, self.food_patterns, mLeft, mRight, muscleList, muscles, self.training_interval, self.total_episodes) for worm_num, candidate in enumerate(batch)]))
                batch_differences = [self.calculate_differences(candidate.weight_matrix)+jitter for candidate in batch]
                # Add jitter to the differences
                differences.extend(batch_differences)
            
            # Generate a random color for this fileself.generate_random_color()
            color = "Blue"
            if 'mark' in filename:
                color = "Red"
            # Plot fitness and differences on separate subplots
            ax1.plot(fitnesses, label=f'Fitness {filename}', color=color)
            ax2.plot(differences, label=f'Differences {filename}', color=color)
        
        plt.tight_layout()

        # Show plot
        plt.show()
        ray.shutdown()
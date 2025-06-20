import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles, muscleList, mLeft, mRight, all_neuron_names
from matplotlib import pyplot as plt
import os
from util.write_read_txt import read_arrays_from_csv_pandas
import random


class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[5], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, genomes=None):
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

    def calculate_differences(self, candidate_weights):
        count  =np.count_nonzero(candidate_weights != self.original_genome)
        assert count == self.calculate_differences2(candidate_weights)
        return count
    
    def calculate_differences2(self, candidate_weights):
        candidate_weights = np.array(candidate_weights)
        if candidate_weights.shape != self.original_genome.shape:
            raise ValueError("Shape of candidate_weights and original_genome must be the same.")
        differences = np.where(candidate_weights != self.original_genome)[0]
        num_differences = len(differences)
        return num_differences

    def generate_random_color(self):
        return '#%06x' % random.randint(0, 0xFFFFFF)
    

    def run(self, env, path='Results_good_tri_all', batch_size=10, jitter_strength=10):
        folder_path = 'Results_good_tri_all'
        base_dir = os.path.dirname(__file__)
        print(base_dir)
        full_folder_path = os.path.join(base_dir, folder_path)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10), sharex=True)
        ax1.set_title('Fitness on Triangle Food Pattern Over Generations', fontsize=24)
        ax1.set_xlabel('Generation', fontsize=20)
        ax1.set_ylabel('Number Of Food Eaten', fontsize=20)

        fitnesses_dict = {'blue': [], 'orange': [], 'red': [], 'cyan': [], "teal": []}
        differences_dict = {'blue': [], 'orange': [], 'red': [], 'cyan': [], "teal": []}

        for filename in os.listdir(full_folder_path):
            self.population = []
            self.initialize_population(read_arrays_from_csv_pandas(os.path.join(full_folder_path, filename)))

            population_batches = [self.population[i:i + batch_size] for i in range(0, len(self.population), batch_size)]
            fitnesses = []
            differences = []
            jitter = 0

            for batch in population_batches:
                fitnesses.extend(ray.get([
                    self.evaluate_fitness_ray.remote(
                        candidate.weight_matrix, all_neuron_names, env, self.food_patterns, mLeft, mRight, muscleList, muscles, 
                        self.training_interval, self.total_episodes
                    ) for worm_num, candidate in enumerate(batch)
                ]))
                batch_differences = [self.calculate_differences(candidate.weight_matrix) + jitter for candidate in batch]
                differences.extend(batch_differences)

            color = "blue"
            if "evo" in filename:
                color = "orange"
            if len(differences) > 90 and differences[90] > 3000:
                color = "red"
            if "random" in filename:
                color = "cyan"
            if "NO" in filename:
                color = "teal"

            fitnesses_dict[color].append(fitnesses)
            differences_dict[color].append(differences)

            # Identify generations where fitness falls by more than 200
            #for i in range(1, len(fitnesses)):
            #    if fitnesses[i - 1] - fitnesses[i] > 50:
            #        print(f"File: {filename}, Generation: {i}, Fitness Drop: {fitnesses[i - 1] - fitnesses[i]}")

            ax1.plot(fitnesses, color=color, alpha=0.3)

        for color, fitnesses_list in fitnesses_dict.items():
            if fitnesses_list:
                avg_fitness = np.mean(fitnesses_list, axis=0)
                #model = (
                #    "Of Nomad Assisted Search" if color == "blue" 
                #    else "Of Evolutionary Algorithm" if color == "green" 
                #    else "Of Searches With Large Differences from the Original Connectome"
                #) not used anymore
                ax1.plot(avg_fitness, color=color, alpha=1, linewidth=2)
        
        ax1.set_xscale('log')

        #handles, labels = ax1.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='center right')
        plt.tight_layout()
        plt.savefig("fig66.svg")
        ray.shutdown()

    def run_and_print_last_generations(self, env, path='Results_good_tri_all', batch_size=10):
        """
        Runs the genetic algorithm and prints the fitness values for the last 10 generations.
        
        Args:
            env: Worm simulation environment instance.
            path (str): Path to the folder containing the connectome data.
            batch_size (int): Size of each batch of population.
        """

        folder_path = path
        base_dir = os.path.dirname(__file__)
        print(base_dir)
        full_folder_path = os.path.join(base_dir, folder_path)

        fitnesses_dict = {}
        for filename in os.listdir(full_folder_path):
            self.population = []

            self.initialize_population(read_arrays_from_csv_pandas(os.path.join(full_folder_path, filename)))

            population_batches = [self.population[i:i + batch_size] for i in range(0, len(self.population), batch_size)]
            fitnesses = []

            for batch in population_batches:
                batch_fitnesses = ray.get([
                    self.evaluate_fitness_ray.remote(
                        candidate.weight_matrix, all_neuron_names, env, self.food_patterns, mLeft, mRight, muscleList, muscles, 
                        self.training_interval, self.total_episodes
                    ) for candidate in batch
                ])
                fitnesses.extend(batch_fitnesses)

            # Store fitness values
            fitnesses_dict[filename] = fitnesses

            # Print the last 10 generations' fitnesses
            print(f"\n--- Fitness for the last 10 generations in file: {filename} ---")
            print(fitnesses[-10:] if len(fitnesses) >= 10 else fitnesses)

        ray.shutdown()
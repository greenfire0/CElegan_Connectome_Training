import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from graphing import graph,graph_comparison
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv



class Genetic_Dyn_Algorithm:
    def __init__(self, population_size,pattern= [5],  total_episodes=10, training_interval=25, genome=None,matrix_shape= 3689):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = self.initialize_population(genome)

    def initialize_population(self, genome=None):
        population = []
        if genome:
            population.append(WormConnectome(weight_matrix=np.array(genome, dtype=float), all_neuron_names=all_neuron_names))
        for _ in range(self.population_size-1):
                population.append(WormConnectome(weight_matrix=np.random.uniform(low=-20, high=20, size=self.matrix_shape).astype(np.float32), all_neuron_names=all_neuron_names))
        return population

    @staticmethod
    @ray.remote
    def evaluate_fitness_ray(candidate_weights,nur_name, env, prob_type, mLeft, mRight, muscleList, muscles,interval,episodes):
        
        sum_rewards = 0
        for a in prob_type:
            candidate = WormConnectome(weight_matrix=candidate_weights,all_neuron_names=nur_name)
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for _ in range(interval):  # training_interval
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    #env.render(worm_num)
                    observation = next_observation
                    sum_rewards+=reward
        return sum_rewards

    def run(self, env , generations=50, batch_size=32):
        last_best = self.original_genome
        ray.init(
            ignore_reinit_error=True,  # Allows reinitialization if Ray is already running
            object_store_memory=7 * 1024 * 1024 * 1024,  # 20 GB in bytes
            num_cpus=16,                                # Number of CPU cores
            )       
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses = []
                for batch in population_batches:
                    fitnesses.extend(([self.evaluate_fitness_ray.remote(candidate.weight_matrix, all_neuron_names, env, self.food_patterns, mLeft, mRight, muscleList, muscles,self.training_interval, self.total_episodes) for worm_num, candidate in enumerate(batch)]))
                #print(fitnesses)
                fitnesses= ray.get(fitnesses)
                if fitnesses:
                    # First element in fitnesses
                    first_element = fitnesses[0]
                    # Greatest element in fitnesses
                    
                    # Count numbers greater than the first element
                    count_greater_than_first = sum(1 for x in fitnesses if x > first_element)
                    
                    # Count numbers smaller than the greatest element
                    count_smaller_than_first = sum(1 for x in fitnesses if x < first_element)
                    
                    # Data for the bar chart
                    data = [count_greater_than_first, count_smaller_than_first]
                    labels = ['Random Weights Perform Better', 'C. Elegan Weights Perform Better']
                    print([count_greater_than_first, count_smaller_than_first])
                    
                    # Plotting the bar chart
                    plt.bar(labels, data)
                    plt.ylabel('Counts')
                    plt.title('Performance Comparison: Randomly Generated Weights vs. C. Elegans Weights')
                    plt.show()

        
        finally:
            ray.shutdown()
import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from graphing import graph,graph_comparison
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
import os
from util.write_read_txt import read_arrays_from_csv_pandas


class Genetic_Dyn_Algorithm:
    def __init__(self, population_size,pattern= [5],  total_episodes=10, training_interval=25, genome=None,matrix_shape= 3689):
        self.population_size = population_size
        print(pattern)
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, genomes=None):
        if (len(genomes)>400):
            genomes = genomes[0:400]
        for g in (genomes):

                self.population.append(WormConnectome(weight_matrix=np.array(g, dtype=float), all_neuron_names=all_neuron_names))

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

    def run(self, env , path='Results/Results_for_paper' , batch_size=10):
        fitness_hold =[]
        ray.init(
            ignore_reinit_error=True,  # Allows reinitialization if Ray is already running
            object_store_memory=14 * 1024 * 1024 * 1024,  # 20 GB in bytes
            num_cpus=16,                                # Number of CPU cores
            )       
        folder_path = 'Results/Results_for_paper'  
        base_dir = os.path.dirname(__file__)  # Get the directory of the current script
        full_folder_path = os.path.join(base_dir, folder_path)
        for filename in os.listdir(full_folder_path):
            self.population = []
            self.initialize_population(read_arrays_from_csv_pandas(os.path.join(full_folder_path,filename)))
            
            population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
            fitnesses = []
            for batch in population_batches:
                    fitnesses.extend(ray.get([self.evaluate_fitness_ray.remote(candidate.weight_matrix, all_neuron_names, env, self.food_patterns, mLeft, mRight, muscleList, muscles,self.training_interval, self.total_episodes) for worm_num, candidate in enumerate(batch)]))
            best_index = np.argmax(fitnesses)
            print(fitnesses[best_index])
        
            if '25' in filename:
                color = 'red'
            elif '50' in filename:
                 color = 'white'
            elif '100' in filename:
                color = 'white'
            elif 'genetic' in filename:
                color = 'blue'
            else:
                color = 'black'

            plt.plot((fitnesses), label=filename, color=color)
            fitness_hold.extend(fitnesses)
                
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()

            # Show plot
        plt.show()
        ray.shutdown()
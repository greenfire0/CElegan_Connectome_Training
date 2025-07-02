import numpy as np
import numpy.typing as npt
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight,all_neuron_names
from Algorithms.algo_utils import initialize_population_with_random_worms, select_parents, crossover\
,evaluate_fitness_ray,mutate
from tqdm import tqdm
import csv
from util.snip import write_worm_to_csv

class Genetic_Dyn_Algorithm:
    def __init__(self,genome:npt.NDArray[np.float64], population_size,pattern= [5],  total_episodes=10, training_interval=25,matrix_shape= 3683):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = initialize_population_with_random_worms(self.population_size,self.matrix_shape,genome)
    
    def run(self, env , generations=50, batch_size=32,filename:str = "Evolutionary_algorithm"):
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses = []
                for batch in population_batches:
                    fitnesses.extend(([evaluate_fitness_ray.remote(candidate.weight_matrix, all_neuron_names, env, self.food_patterns, mLeft, mRight, muscleList, muscles,self.training_interval, self.total_episodes) for worm_num, candidate in enumerate(batch)]))

                fitnesses = ray.get(fitnesses)
                for a,result in enumerate(fitnesses):
                    lasso_penalty = env.lasso_reg(self.population[a].weight_matrix,self.original_genome)
                    fitnesses[a]=(np.max([(result+lasso_penalty),0]))
                    
                best_index = np.argmax(fitnesses)
                best_fitness = fitnesses[best_index]
                best_weights = np.copy(self.population[best_index].weight_matrix)
                best_worm = WormConnectome(weight_matrix=best_weights, all_neuron_names=all_neuron_names)


                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                self.population = select_parents(self.population,fitnesses, self.population_size // 2)
                offspring = crossover(self.population, fitnesses, self.population_size - len(self.population)-1,self.matrix_shape)
                offspring = mutate(offspring,self.matrix_shape)
                self.population.extend(offspring)
                self.population.append(best_worm)
                write_worm_to_csv(filename, best_worm,max_rows=generations)
               
            return best_worm.weight_matrix
        
        finally:
            ray.shutdown()
import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight,all_neuron_names
from tqdm import tqdm
import csv
from Algorithms.algo_utils import initialize_population, select_parents,\
crossover, evaluate_fitness_ray,evaluate_fitness_static,BlackboxWrapper,evaluate_fitness_nomad
from util.snip import write_worm_to_csv
import numpy.typing as npt


class Genetic_Dyn_Algorithm:
    def __init__(self, population_size:int,genome:npt.NDArray[np.float64],pattern:list= [5],  total_episodes:int=0, training_interval:int=250,matrix_shape:int= 3689,indicies=[]):
        self.population_size:int = population_size
        self.indicies = indicies
        self.matrix_shape:int = matrix_shape
        self.total_episodes:int = total_episodes
        self.training_interval:int = training_interval
        self.original_genome = genome
        self.food_patterns:list = pattern
        assert(len(genome) == matrix_shape)
        self.population = initialize_population(self.population_size,genome)

    def run(self, env, generations=50, batch_size=32,filename:str = "Pure_nomad"):
        last_best = 0
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses,futures = [],[]
                for batch in population_batches:
                    for candidate in (batch):
                            futures.append(evaluate_fitness_nomad.remote(
                                evaluate_fitness_static,
                                candidate_weights= candidate.weight_matrix,
                                nur_name = all_neuron_names,
                                env = env, 
                                prob_type = self.food_patterns,
                                mLeft = mLeft,
                                mRight = mRight,
                                muscleList =muscleList,
                                muscles= muscles,
                                interval = self.training_interval,
                                episodes = self.total_episodes,
                                ind=np.random.choice(self.matrix_shape, size=49, replace=False),
                                bounds = 4,
                                bb_eval = 250,
                                verify = False # turn this on for debugging
                            ))           #    np.random.choice(self.matrix_shape, size=49, replace=False)
                   


                results = ray.get(futures)
                for a,result in enumerate(results):
                        
                        self.population[a].weight_matrix[result[0][0]] = np.copy(result[0][1])
                        fitnesses.append(np.max([(result[1]),0]))


                best_index = np.argmax(fitnesses)
                best_fitness = fitnesses[best_index]
                best_weights = np.copy(self.population[best_index].weight_matrix)
                best_worm = WormConnectome(weight_matrix=best_weights, all_neuron_names=all_neuron_names)

                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                self.population = select_parents(self.population,fitnesses, self.population_size // 2 )
                self.population.extend(crossover(self.population, fitnesses, self.population_size - len(self.population),self.matrix_shape))
                self.population.append(best_worm)
                
                #remove or true if you only want improvements
                write_worm_to_csv(filename, best_worm,max_rows=generations)

            return best_weights
        finally:
            ray.shutdown()
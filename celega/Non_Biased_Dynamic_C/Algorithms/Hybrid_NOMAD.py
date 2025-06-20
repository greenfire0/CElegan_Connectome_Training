import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight,all_neuron_names
import PyNomad
from tqdm import tqdm
import csv
from Algorithms.algo_utils import initialize_population_with_random_worms, select_parents, crossover\
,evaluate_fitness_ray,evaluate_fitness_static,mutate, evaluate_fitness_nomad
from util.snip import write_worm_to_csv
import numpy.typing as npt

class Genetic_Dyn_Algorithm:
    def __init__(self,genome:npt.NDArray[np.float64], population_size,pattern= [5],  total_episodes=0, training_interval=250, matrix_shape= 3689,indicies=[]):
        self.population_size:int = population_size
        self.indicies:int = indicies
        self.matrix_shape:int = matrix_shape
        self.total_episodes:int = total_episodes
        self.training_interval:int = training_interval
        self.original_genome = genome
        self.food_patterns:list = pattern
        assert(len(genome) == matrix_shape)
        self.population = initialize_population_with_random_worms(self.population_size, self.matrix_shape, genome)


    def run(self, env, generations=50, batch_size=32,filename:str = "Hybrid_nomad"):
        
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses = []
                futures = []
                record_ind = []
                for batch in population_batches:
                    for candidate in (batch):
                        ind = (np.where(candidate.weight_matrix != self.original_genome)[0])
                        if (len(ind) < 50) and (len(ind) > 0) and not any(np.array_equal(ind, arr) for arr in record_ind):
                            record_ind.append(ind)
                            #print(record_ind)
                            # Submit task to Ray and collect future
                            
                            futures.append(evaluate_fitness_nomad.remote(
                                func=evaluate_fitness_static,
                                candidate_weights=candidate.weight_matrix,
                                nur_name= all_neuron_names,
                                env=env,
                                prob_type=self.food_patterns,
                                mLeft=mLeft,
                                mRight=mRight,
                                muscleList=muscleList,
                                muscles=muscles,
                                interval=self.training_interval,
                                episodes=self.total_episodes,
                                ind = ind,
                                bounds = 2,
                                bb_eval = 25,
                                verify = False # turn this on for debugging
                            ))
                        else:
                            # Submit task to Ray and collect future
                            futures.append(evaluate_fitness_ray.remote(
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
                            ))
                results = ray.get(futures)
                # Process results
                fitnesses = []
                for a,result in enumerate(results):
                    if isinstance(result, tuple):
                        self.population[a].weight_matrix[result[0][0]] = np.copy(result[0][1])
                        lasso_penalty = env.lasso_reg(self.population[a].weight_matrix,self.original_genome)
                        fitnesses.append(np.max([(result[1]+lasso_penalty),0]))
                    else:
                        lasso_penalty = env.lasso_reg(self.population[a].weight_matrix,self.original_genome)
                        fitnesses.append(np.max([(result+lasso_penalty),0]))

                best_index = np.argmax(fitnesses)  
                best_fitness = fitnesses[best_index]
                best_candidate = WormConnectome(weight_matrix=np.copy(self.population[best_index].weight_matrix), all_neuron_names=all_neuron_names)
                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                # Select parents from the entire population
                self.population = select_parents(self.population,fitnesses, self.population_size // 2)
                
                # Generate offspring through crossover and mutation
                offspring = crossover(self.population, fitnesses, self.population_size - len(self.population) - 1,self.matrix_shape)
                offspring = mutate(offspring,self.matrix_shape)
                self.population.extend(offspring)
                self.population.append(best_candidate)
                
                #remove or true if you only want improvements
                write_worm_to_csv(filename, best_candidate,max_rows=generations)
            return best_candidate.weight_matrix
        
        finally:
            ray.shutdown()


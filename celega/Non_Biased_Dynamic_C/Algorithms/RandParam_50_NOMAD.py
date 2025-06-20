import numpy as np
import numpy.typing as npt
import ray
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight,all_neuron_names
import PyNomad
from tqdm import tqdm
import csv
from Algorithms.algo_utils import initialize_population\
,evaluate_fitness_ray,evaluate_fitness_static,mutate, evaluate_fitness_nomad
from util.snip import write_worm_to_csv

class Genetic_Dyn_Algorithm:
    def __init__(self,genome:npt.NDArray[np.float64], population_size,pattern= [5],  total_episodes=0, training_interval=250,matrix_shape= 3689,indicies=[]):
        self.population_size = population_size
        self.indicies = indicies
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        assert(len(genome) == matrix_shape)
        self.population = initialize_population(self.population_size,genome)

    def run(self, env, generations=50, batch_size=32,filename:str = "Random_50_nomad"):
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
                            # Submit task to Ray and collect future
                            
                            futures.append(evaluate_fitness_nomad.remote(
                                evaluate_fitness_static,
                                candidate.weight_matrix,
                                all_neuron_names,
                                env,
                                self.food_patterns,
                                mLeft,
                                mRight,
                                muscleList,
                                muscles,
                                self.training_interval,
                                self.total_episodes,
                                ind,                                
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
                        lasso_penalty =0
                        fitnesses.append(np.max([(result[1]+lasso_penalty),0]))
                    else:
                        lasso_penalty =0
                        fitnesses.append(np.max([(result+lasso_penalty),0]))

                best_index = np.argmax(fitnesses)  
                best_fitness = fitnesses[best_index]
                best_candidate = self.population[best_index]

                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                write_worm_to_csv(filename, best_candidate,max_rows=generations)

                if (generation//4) ==0:
                    self.population = mutate(self.population,self.matrix_shape,n=2) # 2 mutations
                

            
            return best_candidate.weight_matrix
        
        finally:
            ray.shutdown()
import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight,all_neuron_names
import PyNomad
from tqdm import tqdm
import csv
from Algorithms.algo_utils import initialize_population_with_random_worms, select_parents, crossover\
,evaluate_fitness_ray,evaluate_fitness_static,mutate, BlackboxWrapper
from util.snip import write_worm_to_csv

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size,pattern= [5],  total_episodes=0, training_interval=250, genome=None,matrix_shape= 3689,indicies=[]):
        self.population_size = population_size
        self.indicies = indicies
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        assert(len(genome) == matrix_shape)
        self.population = initialize_population_with_random_worms(self.population_size, self.matrix_shape, genome)


    def run(self, env, generations=50, batch_size=32):
        
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
                            
                            futures.append(self.evaluate_fitness_nomad.remote(
                                evaluate_fitness_static,
                                self.original_genome,
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
                                ind
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
                best_candidate = self.population[best_index]

                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                # Select parents from the entire population
                self.population = select_parents(self.population,fitnesses, self.population_size // 2)
                
                # Generate offspring through crossover and mutation
                offspring = crossover(self.population, fitnesses, self.population_size - len(self.population) - 1,self.matrix_shape)
                offspring = mutate(offspring,self.matrix_shape)
                self.population.extend(offspring)
                self.population.append(best_candidate)
                
                #remove or true if you only want improvements
                write_worm_to_csv('nomad_hybrid.csv', best_candidate)
            return best_candidate.weight_matrix
        
        finally:
            ray.shutdown()
    ##prevent already searched shit from vbieng searchged
    @staticmethod
    @ray.remote
    def evaluate_fitness_nomad(func,ori, candidate_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind):
        if ind.size == 0:
                raise ValueError("No difference between candidate weights and original weights")
        x0 = np.array(candidate_weights[ind])
        lower_bounds = (x0 - 2).tolist()
        upper_bounds = (x0 + 2).tolist()
        x0 = x0.tolist()
        
        params = [
            'DISPLAY_DEGREE 0', 
            'DISPLAY_STATS BBE BLK_SIZE OBJ', 
            'BB_MAX_BLOCK_SIZE 4',
            'MAX_BB_EVAL 25'
        ]
        wrapper = BlackboxWrapper(func,env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind,ori)
        result = PyNomad.optimize(wrapper.blackbox_block, x0, lower_bounds, upper_bounds,params)
        # Use NOMAD's minimize function with blackbox_block and pass additional args
        w_test = np.copy(candidate_weights)
        w_test.setflags(write=True)        
        w_test[ind] = np.copy(result['x_best'])
        fitness_verify = func(
                                    w_test,
                                    all_neuron_names,
                                    env,
                                    prob_type,
                                    mLeft,
                                    mRight,
                                    muscleList,
                                    muscles,
                                    interval,
                                    episodes)
        #print("fitness",-result['f_best'],"fitness",fitness_verify)
        assert abs(fitness_verify+result['f_best'])<2,( w_test[ind]==result['x_best'], "\nResults\n",fitness_verify,result['f_best'])
        # Reconstruct the full candidate weights with optimized values
        #optimized_weights[ind] = result.x
        return ([ind,result['x_best']],-result['f_best'])


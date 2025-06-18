import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight,all_neuron_names
import PyNomad
from tqdm import tqdm
import csv
from Algorithms.algo_utils import initialize_population, select_parents,\
crossover, evaluate_fitness_ray,evaluate_fitness_static,BlackboxWrapper


class Genetic_Dyn_Algorithm:
    def __init__(self, population_size:int,pattern:list= [5],  total_episodes:int=0, training_interval:int=250, genome=None,matrix_shape:int= 3689,indicies=[]):
        self.population_size:int = population_size
        self.indicies = indicies
        self.matrix_shape:int = matrix_shape
        self.total_episodes:int = total_episodes
        self.training_interval:int = training_interval
        self.original_genome:list = genome
        self.food_patterns:list = pattern
        assert(len(genome) == matrix_shape)
        self.population = initialize_population(self.population_size,genome)

    def run(self, env, generations=50, batch_size=32,filename="arrays"):
        last_best = 0
        ray.init(
            ignore_reinit_error=True,
            object_store_memory=15 * 1024 * 1024 * 1024,
            num_cpus=16,
        )
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses,futures = [],[]
                for batch in population_batches:
                    for candidate in (batch):
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
                                np.random.choice(self.matrix_shape, size=49, replace=False)
                            ))           #    np.random.choice(self.matrix_shape, size=49, replace=False)
                   


                results = ray.get(futures)
                for a,result in enumerate(results):
                        
                        self.population[a].weight_matrix[result[0][0]] = np.copy(result[0][1])
                        fitnesses.append(np.max([(result[1]),0]))


                best_index = np.argmax(fitnesses)
                best_fitness = fitnesses[best_index]
                best_weights = np.copy(self.population[best_index].weight_matrix)


                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                self.population = select_parents(self.population,fitnesses, self.population_size // 2 )
                self.population.extend(crossover(self.population, fitnesses, self.population_size - len(self.population),self.matrix_shape))
                self.population.append(WormConnectome(weight_matrix=best_weights, all_neuron_names=all_neuron_names))
                
                #remove or true if you only want improvements
                if True or ( best_fitness>last_best) :
                    last_best = best_fitness
                    with open((filename+'.csv'), 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(best_weights.tolist())

            
            return best_weights
        

        finally:
            ray.shutdown()

    @staticmethod
    @ray.remote
    def evaluate_fitness_nomad(func,ori, candidate_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind):
        if ind.size == 0:
                raise ValueError("No difference between candidate weights and original weights")
        x0 = np.array(candidate_weights[ind])
        lower_bounds = (x0 - 4).tolist()
        upper_bounds = (x0 + 4).tolist()
        x0 = x0.tolist()
        
        params = [
            'DISPLAY_DEGREE 0', 
            'DISPLAY_STATS BBE BLK_SIZE OBJ', 
            'BB_MAX_BLOCK_SIZE 4',
            'MAX_BB_EVAL 250'
        ]
        wrapper = BlackboxWrapper(func,env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind,candidate_weights)
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
        del wrapper
        return ([ind,result['x_best']],-result['f_best'])


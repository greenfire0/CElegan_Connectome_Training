import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names
import PyNomad
from tqdm import tqdm
import csv
import copy
from util.write_read_txt import read_arrays_from_csv_pandas
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
        self.population = self.initialize_population(genome)


    def initialize_population(self, genome=None):
        population = []
        population.append(WormConnectome(weight_matrix=np.array(genome, dtype=np.float32), all_neuron_names=all_neuron_names))
        for _ in range(self.population_size-1):
                population.append(self.give_random_worm())
        return population
    
    def give_random_worm(self):
        return WormConnectome(weight_matrix=np.random.uniform(low=-20, high=20, size=self.matrix_shape).astype(np.float32), all_neuron_names=all_neuron_names)

    def select_parents(self, fitnesses, num_parents):
        parents = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents]
    
    ###fix the corssover function so that for each theers a crossvoer
    def crossover(self, parents, fitnesses, num_offspring):
        offspring = []
        parent_fitnesses = np.array([fitnesses[i] for i in np.argsort(fitnesses)[-len(parents):]])
        fitness_probs = parent_fitnesses / np.sum(parent_fitnesses)

        for _ in range(num_offspring):
            parent1 = np.random.choice(parents, p=fitness_probs)
            parent2 = np.random.choice(parents, p=fitness_probs)
            crossover_prob = (fitness_probs[parents.index(parent1)] / (fitness_probs[parents.index(parent1)] + fitness_probs[parents.index(parent2)]))**1.2
            prob_array = (np.random.rand(self.matrix_shape) < crossover_prob).astype(int)
            final_array = np.where(prob_array, parent1.weight_matrix, parent2.weight_matrix)
            offspring.append(WormConnectome(weight_matrix=final_array,all_neuron_names=all_neuron_names))
        return offspring

    def mutate(self, offspring, n=5):
        for child in offspring:
                indices_to_mutate = np.random.choice(self.indicies, size=n, replace=False)
                new_values = np.random.uniform(low=-20, high=20, size=n)
                child.weight_matrix[indices_to_mutate] = new_values
        return offspring

    @staticmethod
    def evaluate_fitness(candidate_weights,nur_name, env, prob_type, mLeft, mRight, muscleList, muscles,interval,episodes):
        sum_rewards = 0
        for a in prob_type:
            candidate = WormConnectome(weight_matrix=candidate_weights,all_neuron_names=nur_name)
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for _ in range(interval):  # training_interval
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    observation = next_observation
                    sum_rewards+=reward
        return sum_rewards
    
    @staticmethod
    @ray.remote
    def evaluate_fitness_ray_evo(candidate_weights,nur_name, env, prob_type, mLeft, mRight, muscleList, muscles,interval,episodes):
        sum_rewards = 0
        for a in prob_type:
            candidate = WormConnectome(weight_matrix=candidate_weights,all_neuron_names=nur_name)
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for _ in range(interval):  # training_interval
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    observation = next_observation
                    sum_rewards+=reward
        
        return sum_rewards

    def run(self, env, generations=50, batch_size=32):
        last_best = 0
        ray.init(
            ignore_reinit_error=True,
            object_store_memory=15 * 1024 * 1024 * 1024,
            num_cpus=16,
        )
        #if self.testing_mode:
            #while 1:
                #print(self.evaluate_fitness(self.population[0],env,self.food_patterns))
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses = []
                futures = []
                record_ind = []
                ff = []
                for batch in population_batches:
                    for candidate in (batch):
                        ind = (np.where(candidate.weight_matrix != self.original_genome)[0])
                        if (len(ind) < 50) and (len(ind) > 0) and not any(np.array_equal(ind, arr) for arr in record_ind):
                            record_ind.append(ind)
                            #print(record_ind)
                            # Submit task to Ray and collect future
                            
                            futures.append(self.evaluate_fitness_nomad.remote(
                                self.evaluate_fitness,
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
                            futures.append(self.evaluate_fitness_ray_evo.remote(
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
                results.extend(ff)
                fitnesses = []
                xax = []
                for a,result in enumerate(results):
                    if isinstance(result, tuple):
                        xax = result[0][0]
                        self.population[a].weight_matrix[result[0][0]] = np.copy(result[0][1])
                        fitnesses.append(result[1])
                    else:
                        fitnesses.append(result)

                # Evaluate fitness using NOMAD in parallel
                
                best_index = np.argmax(fitnesses)  
                best_fitness = fitnesses[best_index]
                best_candidate = self.population[best_index]

                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                # Select parents from the entire population
                self.population = self.select_parents(fitnesses, self.population_size // 2)
                
                # Generate offspring through crossover and mutation
                offspring = self.crossover(self.population, fitnesses, self.population_size - len(self.population) - 1)
                offspring = self.mutate(offspring)
                self.population.extend(offspring)
                self.population.append(best_candidate)
                
                if (    best_fitness>last_best) or True:
                    

                    last_best = best_fitness
                    with open('arrays.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        writer.writerow(best_candidate.weight_matrix.tolist())

            
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
            'MAX_BB_EVAL 50'
        ]
        wrapper = BlackboxWrapper(func,env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind,ori)
        result = PyNomad.optimize(wrapper.blackbox_block, x0, lower_bounds, upper_bounds,params)
        # Use NOMAD's minimize function with blackbox_block and pass additional args
        
        # Reconstruct the full candidate weights with optimized values
        #optimized_weights[ind] = result.x
        return ([ind,result['x_best']],-result['f_best'])

class BlackboxWrapper:
    def __init__(self, func, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,index,ori):
        self.env = env
        self.func = func
        self.prob_type = prob_type
        self.mLeft = mLeft
        self.mRight = mRight
        self.muscleList = muscleList
        self.muscles = muscles
        self.interval = interval
        self.episodes = episodes
        self.ind = index
        self.old_worm = ori

    def blackbox(self, eval_point):
            
            candidate_edit = []
            candidate_weights = np.copy(self.old_worm).astype(np.float32)
            candidate_weights.setflags(write=True)
            for a in range(len(self.ind)):
                candidate_edit.append(eval_point.get_coord(a))

            candidate_weights[self.ind] = candidate_edit
            eval_value = -1*self.func(
                    candidate_weights, all_neuron_names, self.env, self.prob_type, 
                    self.mLeft, self.mRight, self.muscleList, self.muscles, self.interval, self.episodes)
            eval_point.setBBO(str(eval_value).encode('utf-8'))
            del candidate_weights
            return True

    def blackbox_block(self, eval_block):
        eval_state = []
        for index in range(eval_block.size()):
            eval_point = eval_block.get_x(index)
            eval_state.append(self.blackbox(eval_point))
        return eval_state
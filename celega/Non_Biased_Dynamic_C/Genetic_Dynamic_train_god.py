import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names
import PyNomad
from tqdm import tqdm
import csv
from util.write_read_txt import read_arrays_from_csv_pandas
import copy

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
        for _ in range(self.population_size):
            population.append(WormConnectome(weight_matrix=np.array(genome, dtype=np.float32), all_neuron_names=all_neuron_names))
        return population
    

    def select_parents(self, fitnesses, num_parents):
        parents = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents]
    
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
    


    def run(self, env, generations=50, batch_size=32):
        last_best = 0
        ray.init(
            ignore_reinit_error=True,
            object_store_memory=15 * 1024 * 1024 * 1024,
            num_cpus=16,
        )
        import os
        os.environ["RAY_DEDUP_LOGS"] = "0"

        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses,futures = [],[]
                for batch in population_batches:
                    for candidate in (batch):
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
                self.population = self.select_parents(fitnesses, self.population_size // 2 )
                
                # Generate offspring through crossover and mutation
                offspring = self.crossover(self.population, fitnesses, self.population_size - len(self.population))
                self.population.extend(offspring)
                self.population.append(WormConnectome(weight_matrix=best_weights, all_neuron_names=all_neuron_names))
                
                #remove or true if you only want improvements
                if True or ( best_fitness>last_best) :
                    last_best = best_fitness
                    with open('arrays.csv', 'a', newline='') as csvfile:
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
        wrapper = BlackboxWrapper(func,env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind,candidate_weights,ori)
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

class BlackboxWrapper:
    def __init__(self, func, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,index,cand,ori):
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
        self.candidate = cand
        self.ori = ori

    def blackbox(self, eval_point):
            
            self.candidate_edit = []
            self.candidate_weights = np.copy(self.candidate).astype(np.float64)
            for a in range(len(self.ind)):
                self.candidate_edit.append(eval_point.get_coord(a))
           # print(self.candidate_weights[self.ind],self.candidate_edit)
            #print(type(self.candidate_edit))
            self.candidate_weights[self.ind] = self.candidate_edit
            #print(self.ind,(np.where(self.candidate_weights != self.ori)[0]))
            eval_value = -1*self.func(
                    self.candidate_weights, all_neuron_names, self.env, self.prob_type, 
                    self.mLeft, self.mRight, self.muscleList, self.muscles, self.interval, self.episodes)
            eval_point.setBBO(str(eval_value).encode('utf-8'))
            del self.candidate_weights
            return True

    def blackbox_block(self, eval_block):
        eval_state = []
        for index in range(eval_block.size()):
            eval_point = eval_block.get_x(index)
            eval_state.append(self.blackbox(eval_point))
        return eval_state
from Worm_Env.connectome import WormConnectome
import numpy as np
from Worm_Env.weight_dict import all_neuron_names
import ray
import PyNomad
import numpy.typing as npt
import time

@staticmethod
def initialize_population(population_size:int, genome):
    population = []
    for _ in range(population_size):
        population.append(WormConnectome(weight_matrix=np.array(genome, dtype=np.float32), all_neuron_names=all_neuron_names))
    return population

@staticmethod
def initialize_population_with_random_worms(population_size,shape, genome=None):
        """
        Initializes the population with population_size-1 random one and one worm with the original connectome
        """
        population = []
        population.append(WormConnectome(weight_matrix=np.array(genome, dtype=float), all_neuron_names=all_neuron_names))
        for _ in range(population_size-1):
                population.append(WormConnectome(weight_matrix=np.random.uniform(low=-20, high=20, size=shape).astype(np.float32), all_neuron_names=all_neuron_names))
        return population

@staticmethod
def select_parents(population, fitnesses, num_parents):
    parents = np.argsort(fitnesses)[-num_parents:]
    return [population[i] for i in parents]

@staticmethod
def crossover(parents, fitnesses, num_offspring,matrix_shape):
    offspring = []
    parent_fitnesses = np.array([fitnesses[i] for i in np.argsort(fitnesses)[-len(parents):]]) ### this might be able to be removed
    fitness_probs = parent_fitnesses / np.sum(parent_fitnesses)
    for _ in range(num_offspring):
        parent1 = np.random.choice(parents, p=fitness_probs)
        parent2 = np.random.choice(parents, p=fitness_probs)
        crossover_prob = (fitness_probs[parents.index(parent1)] / (fitness_probs[parents.index(parent1)] + fitness_probs[parents.index(parent2)]))**1.2
        prob_array = (np.random.rand(matrix_shape) < crossover_prob).astype(int)
        final_array = np.where(prob_array, parent1.weight_matrix, parent2.weight_matrix)
        offspring.append(WormConnectome(weight_matrix=final_array,all_neuron_names=all_neuron_names))
    return offspring

@ray.remote
def evaluate_fitness_ray(candidate_weights,nur_name, env, prob_type, mLeft, mRight, muscleList, muscles,interval,episodes):
    sum_rewards = 0
    candidate = WormConnectome(weight_matrix=candidate_weights,all_neuron_names=nur_name)
    for a in prob_type:
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
def evaluate_fitness_static(candidate_weights,nur_name, env, prob_type, mLeft, mRight, muscleList, muscles,interval,episodes):
    sum_rewards = 0
    candidate = WormConnectome(weight_matrix=candidate_weights,all_neuron_names=nur_name)
    for a in prob_type:
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
def give_random_worm(self):
    return WormConnectome(weight_matrix=np.random.uniform(low=-20, high=20, size=self.matrix_shape).astype(np.float32), all_neuron_names=all_neuron_names)
@staticmethod
def mutate(offspring,matrix_shape, n=5):
    for child in offspring:
                indices_to_mutate = np.random.choice(matrix_shape, size=n, replace=False)
                new_values = np.random.uniform(low=-20, high=20, size=n)
                child.weight_matrix[indices_to_mutate] = new_values
    return offspring



@ray.remote # type: ignore[arg-type]
def evaluate_fitness_nomad(func, candidate_weights:npt.NDArray[np.float64], nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind,bounds:int,bb_eval,verify:bool=False):
        wall_start = time.perf_counter() 
        cpu_start  = time.process_time()
        if ind.size == 0:
                raise ValueError(f"Please pass indicies,{ind}")
        x0 = np.array(candidate_weights[ind])
        lower_bounds = (x0 - bounds).tolist()
        upper_bounds = (x0 + bounds).tolist()
        x0 = x0.tolist()
        
        params = [
            'DISPLAY_DEGREE 0', 
            'DISPLAY_STATS BBE BLK_SIZE OBJ', 
            'BB_MAX_BLOCK_SIZE 4',
            f'MAX_BB_EVAL {bb_eval}'
        ]
        wrapper = BlackboxWrapper(func,env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,ind,candidate_weights)
        result = PyNomad.optimize(wrapper.blackbox_block, x0, lower_bounds, upper_bounds,params)
        # Use NOMAD's minimize function with blackbox_block and pass additional args

        if verify:
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
            del w_test,fitness_verify
        
        cpu_total  = time.process_time() - cpu_start
        wall_total = time.perf_counter() - wall_start
        cpu_fit    = wrapper.time                # only fitness calls
        cpu_nomad  = max(cpu_total - cpu_fit, 0) # PyNomad + copy + misc
        del wrapper
        return ([ind,result['x_best']],-result['f_best'],cpu_fit,wall_total,cpu_nomad)



class BlackboxWrapper:
    def __init__(self, func, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes,index,cand):
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
        self.time =0.0

    def blackbox(self, eval_point):
            
            self.candidate_edit = []
            self.candidate_weights = np.copy(self.candidate).astype(np.float64)
            for a in range(len(self.ind)):
                self.candidate_edit.append(eval_point.get_coord(a))
            self.candidate_weights[self.ind] = self.candidate_edit
            tic = time.process_time()
            eval_value = -1*self.func(
                    self.candidate_weights, all_neuron_names, self.env, self.prob_type, 
                    self.mLeft, self.mRight, self.muscleList, self.muscles, self.interval, self.episodes)
            self.time += time.process_time() - tic
            eval_point.setBBO(str(eval_value).encode('utf-8'))
            del self.candidate_weights
            return True

    def blackbox_block(self, eval_block):
        eval_state = []
        for index in range(eval_block.size()):
            eval_point = eval_block.get_x(index)
            eval_state.append(self.blackbox(eval_point))
        return eval_state




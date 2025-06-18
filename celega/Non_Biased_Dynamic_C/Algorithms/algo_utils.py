from Worm_Env.trained_connectome import WormConnectome
import numpy as np
from Worm_Env.weight_dict import all_neuron_names
import ray

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
    parent_fitnesses = np.array([fitnesses[i] for i in np.argsort(fitnesses)[-len(parents):]])
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

    def blackbox(self, eval_point):
            
            self.candidate_edit = []
            self.candidate_weights = np.copy(self.candidate).astype(np.float64)
            for a in range(len(self.ind)):
                self.candidate_edit.append(eval_point.get_coord(a))
            self.candidate_weights[self.ind] = self.candidate_edit
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
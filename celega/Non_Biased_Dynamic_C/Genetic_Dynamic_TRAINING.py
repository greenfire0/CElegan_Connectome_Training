import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from graphing import graph,graph_comparison
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names
from tqdm import tqdm
import csv

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size,pattern= [5],  total_episodes=10, training_interval=25, genome=None,matrix_shape= 3683):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = self.initialize_population(genome)

    def initialize_population(self, genome=None):
        population = []
        population.append(WormConnectome(weight_matrix=np.array(genome, dtype=float), all_neuron_names=all_neuron_names))
        for _ in range(self.population_size-1):
                population.append(WormConnectome(weight_matrix=np.random.uniform(low=-20, high=20, size=self.matrix_shape).astype(np.float32), all_neuron_names=all_neuron_names))
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

    def mutate(self, offspring, n=5):
        for child in offspring:
                indices_to_mutate = np.random.choice(self.matrix_shape, size=n, replace=False)
                new_values = np.random.uniform(low=-20, high=20, size=n)
                child.weight_matrix[indices_to_mutate] = new_values
        return offspring
    
    @staticmethod
    @ray.remote
    def evaluate_fitness_ray(candidate_weights,nur_name, env, prob_type, mLeft, mRight, muscleList, muscles,interval,episodes):
        sum_rewards = 0
        for a in prob_type:
            candidate = WormConnectome(weight_matrix=candidate_weights,all_neuron_names=nur_name)
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for _ in range(interval):  # training_interval
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    #env.render(0)
                    observation = next_observation
                    sum_rewards+=reward
        return sum_rewards
    
    def run(self, env , generations=50, batch_size=32):
        last_best =0
        ray.init(
            ignore_reinit_error=True,  # Allows reinitialization if Ray is already running
            object_store_memory=15 * 1024 * 1024 * 1024,  # 20 GB in bytes
            num_cpus=16,                                # Number of CPU cores
            )       
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses = []
                for batch in population_batches:
                    fitnesses.extend(([self.evaluate_fitness_ray.remote(candidate.weight_matrix, all_neuron_names, env, self.food_patterns, mLeft, mRight, muscleList, muscles,self.training_interval, self.total_episodes) for worm_num, candidate in enumerate(batch)]))

                fitnesses = ray.get(fitnesses)
                for a,result in enumerate(fitnesses):
                    lasso_penalty = env.lasso_reg(self.population[a].weight_matrix,self.original_genome)
                    fitnesses[a]=(np.max([(result+lasso_penalty),0]))
                    
                best_index = np.argmax(fitnesses)
                best_fitness = fitnesses[best_index]
                best_candidate = self.population[best_index]
                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                self.population = self.select_parents(fitnesses, self.population_size // 2)
                
                # Generate offspring through crossover and mutation
                offspring = self.crossover(self.population, fitnesses, self.population_size - len(self.population)-1)
                offspring = self.mutate(offspring)
                self.population.extend(offspring)
                self.population.insert(0,best_candidate)
                
                #remove or true if you only want improvements
                if True or not np.array_equal(last_best, best_candidate.weight_matrix):
                    with open('arrays.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(best_candidate.weight_matrix.flatten().tolist()) 
            return best_candidate.weight_matrix
        
        finally:
            ray.shutdown()
import numpy as np
import ray
from Worm_Env.trained_connectome import wormConnectone
from graphing import graph
from Worm_Env.weight_dict import dict
from tqdm import tqdm

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, matrix_shape, mutation_rate=0.5, total_episodes=10, training_interval=25,genome=None):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.mutation_rate = mutation_rate
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.population = self.initialize_population(genome)

    def initialize_population(self,genome=None):
        population = []
        if genome == None:
            for _ in range(self.population_size):
                population.append(wormConnectone(weight_matrix=np.random.randn(self.matrix_shape)*10))
            #print(len(population))
        else:
            for _ in range(self.population_size):
                population.append(wormConnectone(weight_matrix=np.array(genome, dtype=float))) 
        return population

    def evaluate_fitness(self, candidate,worm_num, env):
        cumulative_rewards = []
        env.reset()
        candidate.modify_combined_weights()
        for _ in range(self.total_episodes):
            observation = env._get_observations()
            #print(candidate.weight_matrix)
            
            #print(list(candidate.combined_weights.items())[0:2])
            #print(candidate.weight_matrix)
            done = False
            for _ in range(self.training_interval):
                movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food)
                next_observation, reward, done, _ = env.step(movement,worm_num,candidate)
                
                #env.render(worm_num)
                observation = next_observation
                cumulative_rewards.append(reward)
            
            
        return np.sum(cumulative_rewards)

    def select_parents(self, fitnesses, num_parents):
        parents = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents]

    def crossover(self, parents, fitnesses, num_offspring):
        offspring = []

        # Get the fitness of selected parents
        parent_fitnesses = np.array([fitnesses[i] for i in np.argsort(fitnesses)[-len(parents):]])

        # Normalize fitness values to get probabilities
        fitness_probs = parent_fitnesses / np.sum(parent_fitnesses)

        for _ in range(num_offspring):
            parent1 = np.random.choice(parents, p=fitness_probs)
            parent2 = np.random.choice(parents, p=fitness_probs)

            # Calculate weighted crossover point based on parents' fitness
            total_length = len(parent1.weight_matrix)
            crossover_prob = fitness_probs[parents.index(parent1)] / (fitness_probs[parents.index(parent1)] + fitness_probs[parents.index(parent2)])
            splice_point = int(crossover_prob * total_length)

            # Ensure at least one element is selected
            splice_point = max(1, min(splice_point, total_length - 1))

            child_weight_matrix = np.concatenate((parent1.weight_matrix[:splice_point], parent2.weight_matrix[splice_point:]))
            offspring.append(wormConnectone(weight_matrix=child_weight_matrix))

        return offspring

    def mutate(self, offspring: np.ndarray) -> np.ndarray:
        for child in offspring:
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.uniform(low=-1, high=1, size=self.matrix_shape)
                child.weight_matrix+=mutation
                #child.weight_matrix = np.clip(child.weight_matrix, -100, 100)
                ###swarm algo higher gen count lower mutation 
                #print(child.weight_matrix)
        return offspring
    
    @ray.remote
    def evaluate_fitness_ray(self, candidate, worm_num, env):
        
        cumulative_reward = self.evaluate_fitness(candidate, worm_num, env)
        return cumulative_reward

    def run(self, env, generations=50):

        # Initialize Ray
        ray.init(ignore_reinit_error=True)

        try:
            for generation in tqdm(range(generations), desc="Generations"):
                fitnesses = []

                # Parallel evaluation of fitness using Ray
                futures = []
                for worm_num, candidate in enumerate(self.population):
                    futures.append(self.evaluate_fitness_ray.remote(self, candidate=candidate, worm_num=worm_num, env=env))
                    #fitnesses.append(self.evaluate_fitness(candidate, worm_num, env))
                # Gather results from Ray futures
                fitnesses = ray.get(futures)

                # Find the best candidate
                best_index = np.argmax(fitnesses)
                best_candidate = self.population[best_index]
                best_fitness = fitnesses[best_index]
                #print(f"Best Fitness: {best_fitness}")
                # Select parents based on fitness
                num_parents = self.population_size // 2
                parents = self.select_parents(fitnesses, num_parents)

                # Generate offspring through crossover and mutation
                num_offspring = self.population_size - num_parents - 1  # Reserve one spot for the best candidate
                offspring = self.crossover(parents, fitnesses, num_offspring)
                offspring = self.mutate(offspring)

                # Create the new population, ensuring the best candidate survives
                self.population = parents + offspring
                self.population.append(best_candidate)

                # Graph the best candidate's weight matrix
                graph(self.population[0].weight_matrix, dict, generation)

                # Explicitly delete unused variables to free memory
                del fitnesses, parents, offspring

                # Optionally, force garbage collection
                import gc
                gc.collect()

        finally:
            # Ensure Ray is shut down even if an error occurs
            ray.shutdown()

        # Return the best solution found
        fitnesses = [self.evaluate_fitness(candidate, worm_num, env) for worm_num, candidate in enumerate(self.population)]
        best_index = np.argmax(fitnesses)

        return self.population[best_index].weight_matrix
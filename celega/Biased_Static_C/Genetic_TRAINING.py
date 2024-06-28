import numpy as np
from trained_connectome_with_bias import wormConnectone
import time 
import ray

class GeneticAlgorithm:
    def __init__(self, population_size, matrix_shape, mutation_rate=0.5, total_episodes=10, training_interval=25):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.mutation_rate = mutation_rate
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            population.append(wormConnectone(weight_m=np.random.randn(self.matrix_shape)*10))
        return population

    def evaluate_fitness(self, candidate,worm_num, env):
        cumulative_rewards = []
        env.reset()
        
        for _ in range(self.total_episodes):
            observation = env._get_observations()
            
            #print(candidate.weight_matrix)
            done = False
            cumulative_reward = 0.0
            for _ in range(self.training_interval):
                movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food,self.training_interval)
                next_observation, reward, done, _ = env.step(movement,worm_num,candidate)
                
                #env.render(worm_num)
                cumulative_reward += reward
                observation = next_observation
            cumulative_rewards.append(cumulative_reward)
            
            
        return np.sum(cumulative_rewards)

    def select_parents(self, fitnesses, num_parents):
        parents = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents]

    def crossover(self, parents, num_offspring):
        offspring = []
        for _ in range(num_offspring):
            parent1 = parents[np.random.randint(len(parents))]
            parent2 = parents[np.random.randint(len(parents))]
            splice_point = np.random.randint(1, len(parent1.weight_matrix))  # Ensure at least one element is selected
            child_weight_matrix = np.concatenate((parent1.weight_matrix[:splice_point], parent2.weight_matrix[splice_point:]))            
            offspring.append(wormConnectone(weight_m=child_weight_matrix))
        return offspring

    def mutate(self, offspring: np.ndarray) -> np.ndarray:
        for child in offspring:
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.standard_cauchy(size=self.matrix_shape) * 5
                child.weight_matrix += mutation
        return offspring
    
    @ray.remote
    def evaluate_fitness_ray(self, candidate, worm_num, env):
        
        cumulative_reward = self.evaluate_fitness(candidate, worm_num, env)
        return cumulative_reward

    def run(self, env, generations=50):
        ray.init(ignore_reinit_error=True)

        prev_weight_matrix = None
        total_diff = 0
        num_diffs = 0

        for generation in range(generations):
            fitnesses = []
            
            # Parallel evaluation of fitness using Ray
            futures = []
            for worm_num, candidate in enumerate(self.population):
                futures.append(self.evaluate_fitness_ray.remote(self=self,candidate=candidate, worm_num=worm_num, env=env))

            # Gather results from Ray futures
            fitnesses = ray.get(futures)

            best_fitness = max(fitnesses)
            print(f"Generation {generation + 1} - Best Fitness: {best_fitness}")

            num_parents = self.population_size // 2
            parents = self.select_parents(fitnesses, num_parents)
            print(f"Generation {generation + 1} - Fitness Standard Deviation: {np.std(fitnesses):.4f}")

            num_offspring = self.population_size - num_parents
            offspring = self.crossover(parents, num_offspring)
            offspring = self.mutate(offspring)
            self.population = parents + offspring
            
            current_weight_matrix = self.population[0].weight_matrix
            
            if prev_weight_matrix is not None:
                diff = np.abs(current_weight_matrix - prev_weight_matrix)
                avg_diff = np.mean(diff)
                total_diff += avg_diff
                num_diffs += 1
                overall_avg_diff = total_diff / num_diffs
                print(f"Generation {generation + 1} - Average Weight Matrix Difference: {overall_avg_diff:.4f}")
            
            prev_weight_matrix = current_weight_matrix

        ray.shutdown()

        # Return the best solution found
        fitnesses = [self.evaluate_fitness(candidate, worm_num, env) for worm_num, candidate in enumerate(self.population)]
        best_index = np.argmax(fitnesses)
        return self.population[best_index].weight_matrix
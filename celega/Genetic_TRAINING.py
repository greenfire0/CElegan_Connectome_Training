import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, matrix_shape, mutation_rate=0.1, total_episodes=10, training_interval=25):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.mutation_rate = mutation_rate
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.randn(*self.matrix_shape) for _ in range(self.population_size)]

    def evaluate_fitness(self, candidate, env, move, PolicyGradientAgent):
        agent = PolicyGradientAgent(candidate)
        cumulative_rewards = []
        for episode in range(self.total_episodes):
            observation = env._get_observation()
            done = False
            cumulative_reward = 0.0
            for step in range(self.training_interval):
                movement = move(observation[0], env.worm.sees_food)
                next_observation, reward, done, _ = env.step(movement)
                agent.store_transition(next_observation, reward)
                cumulative_reward += reward
                if done:
                    break
                observation = next_observation
            cumulative_rewards.append(cumulative_reward)
        return np.mean(cumulative_rewards)

    def select_parents(self, fitnesses, num_parents):
        parents = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents]

    def crossover(self, parents, num_offspring):
        offspring = []
        for _ in range(num_offspring):
            parent1 = parents[np.random.randint(len(parents))]
            parent2 = parents[np.random.randint(len(parents))]
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        for child in offspring:
            if np.random.rand() < self.mutation_rate:
                mutation_matrix = np.random.randn(*child.shape)
                child += mutation_matrix * self.mutation_rate
        return offspring

    def run(self, env, move, PolicyGradientAgent, generations=50):
        for generation in range(generations):
            fitnesses = [self.evaluate_fitness(candidate, env, move, PolicyGradientAgent) for candidate in self.population]
            best_fitness = max(fitnesses)
            print(f"Generation {generation + 1} - Best Fitness: {best_fitness}")

            num_parents = self.population_size // 2
            parents = self.select_parents(fitnesses, num_parents)

            num_offspring = self.population_size - num_parents
            offspring = self.crossover(parents, num_offspring)
            offspring = self.mutate(offspring)

            self.population = parents + offspring

        # Return the best solution found
        fitnesses = [self.evaluate_fitness(candidate, env, move, PolicyGradientAgent) for candidate in self.population]
        best_index = np.argmax(fitnesses)
        return self.population[best_index]

 def initialize_population(self, genome=None):
        population = []
        for _ in range(self.population_size):
            population.append(WormConnectome(weight_matrix=np.array(genome, dtype=np.float32), all_neuron_names=all_neuron_names))
        return population
    pass

 def select_parents(self, fitnesses, num_parents):
        parents = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents]
    pass

def crossover(self, parents, fitnesses, num_offspring):
        offspring = []
        parent_fitnesses = np.array([fitnesses[i] for i in np.argsort(fitnesses)[-len(parents):]])
        fitness_probs = parent_fitnesses / np.sum(parent_fitnesses)
    pass 

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
    pass
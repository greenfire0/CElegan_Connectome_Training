import numpy as np
from trained_connectome import wormConnectone

class GeneticRUN:
    def __init__(self, population_dna, training_interval=25):
        self.population_dna = population_dna

        self.training_interval = training_interval
        self.population = self.initialize_population()

    def initialize_population(self):
        population = [wormConnectone(weight_matrix=self.population_dna)]
        return population

    def evaluate_fitness(self, candidate,worm_num, env):
            env.reset()
            candidate.modify_combined_weights()
            observation = env._get_observations()
            for _ in range(self.training_interval):
                movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food,self.training_interval)
                next_observation, reward, done, _ = env.step(movement,worm_num,candidate)
                
                env.render(worm_num)                
                observation = next_observation

    def run(self, env, generations=50):
        for generation in range(generations):
            for worm_num, candidate in enumerate(self.population):
                self.evaluate_fitness(candidate, worm_num, env)
                candidate.createpostSynaptic()

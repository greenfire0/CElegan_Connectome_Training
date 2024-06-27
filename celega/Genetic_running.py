import numpy as np
from trained_connectome_with_bias import wormConnectone
import time 

class GeneticRUN:
    def __init__(self, population_dna,  total_episodes=10, training_interval=25):
        self.population_dna = population_dna

        self.training_interval = training_interval
        self.population = self.initialize_population()

    def initialize_population(self):
        population = [wormConnectone(weight_m=self.population_dna)]
        return population

    def evaluate_fitness(self, candidate,worm_num, env):
            env.reset()
            observation = env._get_observations()
            for _ in range(self.training_interval):
                movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food,self.training_interval)
                next_observation, reward, done, _ = env.step(movement,worm_num,candidate)
                
                env.render(worm_num)                
                observation = next_observation

    
    def evaluate_fitness_ray(self, candidate, worm_num, env):
        cumulative_reward = self.evaluate_fitness(candidate, worm_num, env)
        return cumulative_reward

    def run(self, env, generations=50):
        for generation in range(generations):
            for worm_num, candidate in enumerate(self.population):
                self.evaluate_fitness(candidate, worm_num, env)
                candidate.createpostSynaptic()

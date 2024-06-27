import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from c_worm import *
from celegan_env import WormSimulationEnv
from trained_connectome_with_bias import all_neuron_names
from Genetic_TRAINING import GeneticAlgorithm

# Genetic Algorithm parameters
population_size = 32
generations = 100
mutation_rate = 0.5
training_interval = 150  # Train the agent every 25 steps
total_episodes = 1  # Number of episodes per evaluation

# Initialize environment
env = WormSimulationEnv(num_worms=population_size)

# Initialize and run the Genetic Algorithm
ga = GeneticAlgorithm(population_size, len(all_neuron_names), mutation_rate, total_episodes, training_interval)
best_weight_matrix = ga.run(env, generations)

print("Best weight matrix found:", best_weight_matrix)

env.close()


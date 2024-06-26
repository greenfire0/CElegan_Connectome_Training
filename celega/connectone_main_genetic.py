import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from c_worm import *
from celegan_env import WormSimulationEnv
from trained_connectome_with_bias import move, createpostSynaptic, weight_matrix
from Genetic_TRAINING import GeneticAlgorithm

# Genetic Algorithm parameters
population_size = 20
generations = 50
mutation_rate = 0.1
training_interval = 25  # Train the agent every 25 steps
total_episodes = 10  # Number of episodes per evaluation
num_worms = 5  # Number of worms in the environment

# Initialize environment
env = WormSimulationEnv(num_worms=num_worms)
createpostSynaptic()

# Initialize and run the Genetic Algorithm
ga = GeneticAlgorithm(population_size, weight_matrix.shape, mutation_rate, total_episodes, training_interval)
best_weight_matrix = ga.run(env, move, generations)

print("Best weight matrix found:", best_weight_matrix)

env.close()

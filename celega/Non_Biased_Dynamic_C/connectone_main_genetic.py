import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from c_worm import *
from celegan_env import WormSimulationEnv
from trained_connectome import all_neuron_names
from Genetic_running import GeneticRUN
# Genetic Algorithm parameters
from Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm
from graphing import graph, graph_weight_sum_divided_by_connections
from weight_dict import dict
population_size = 16
generations = 300
mutation_rate = 0.5
training_interval = 150  # Train the agent every 25 steps
total_episodes = 1  # Number of episodes per evaluation
train_params =3689 #number of connections
##que te vaya bien
env = WormSimulationEnv(num_worms=population_size)
# Initialize and run the Genetic Algorithm
ga = Genetic_Dyn_Algorithm(population_size, train_params, mutation_rate, total_episodes, training_interval)
best_weight_matrix = ga.run(env, generations)
print("Best weight matrix found:", best_weight_matrix)



graph(best_weight_matrix,dict)
#graph_weight_sum_divided_by_connections(best_weight_matrix, dict)
# Run the simulation with the best weight matrix
GeneticRUN(best_weight_matrix, total_episodes, training_interval).run(env)

env.close()
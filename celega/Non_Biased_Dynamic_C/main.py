

import numpy as np
import matplotlib.pyplot as plt
import math
from c_worm import *
from celegan_env import WormSimulationEnv
from trained_connectome import all_neuron_names
from Genetic_running import GeneticRUN
# Genetic Algorithm parameters
from Genetic_Dynamic_TRAINING import * 
#from graphing import graph, graph_dif
from weight_dict import dict
from util.write_read_txt import write_array_to_file
population_size = 16
generations = 500
mutation_rate = 1
training_interval = 250  # Train the agent every 25 steps
total_episodes = 1  # Number of episodes per evaluation
train_params =3689 #number of connections
##que te vaya bien
##constantly validate results
##start from a prexisting model and validate your code by recontruction of results
values_list = []
for sub_dict in dict.values():
     # Extend the values list with the values from each sub-dictionary
    values_list.extend(sub_dict.values())

env = WormSimulationEnv(num_worms=population_size)
# Initialize and run the Genetic Algorithm
ga = Genetic_Dyn_Algorithm(population_size, train_params, mutation_rate, total_episodes, training_interval,values_list)
best_weight_matrix = ga.run(env, generations)
print("Best weight matrix found:", best_weight_matrix)




    
write_array_to_file(best_weight_matrix,"array.txt")


#graph(best_weight_matrix,dict)
#graph(np.array(values_list),dict)
#graph_dif(best_weight_matrix,np.array(values_list),dict)
# Run the simulation with the best weight matrix
GeneticRUN(best_weight_matrix, total_episodes, training_interval).run(env)

env.close()
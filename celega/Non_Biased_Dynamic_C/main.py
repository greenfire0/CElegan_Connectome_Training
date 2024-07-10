from Worm_Env.celegan_env import WormSimulationEnv
from Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm
from Worm_Env.weight_dict import dict
from graphing import graph_comparison
from util.write_read_txt import write_array_to_file, read_array_from_file
import numpy as np 


population_size = 64*8*8
generations = 75
mutation_rate = 1
training_interval = 250
total_episodes = 1  # Number of episodes per evaluation
train_params =3689 #number of connections



##que te vaya bien
##constantly validate results
##start from a prexisting model and validate your code by recontruction of results



values_list = []
for sub_dict in dict.values():
     # Extend the values list with the values from each sub-dictionary
    values_list.extend(sub_dict.values())
old_wm = np.array(read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/random_worm.txt"))
print("Running Genetic Algoritm")
env = WormSimulationEnv(num_worms=population_size)
ga = Genetic_Dyn_Algorithm(population_size, train_params, mutation_rate, total_episodes, training_interval,values_list)
best_weight_matrix = ga.run(env,old_wm, generations)
print("Best weight matrix found:", best_weight_matrix)   
write_array_to_file(best_weight_matrix,"cheaty_worm4.txt")

#best_weight_matrix = read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/cheaty_worm.txt")
#graph_comparison(best_weight_matrix,np.array(values_list),dict)
#graph(best_weight_matrix,dict)
#graph(np.array(values_list),dict)

# Run the simulation with the best weight matrix
#GeneticRUN(best_weight_matrix, training_interval).run(env)
#env.close()
from Worm_Env.celegan_env import WormSimulationEnv
from Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm
#from Genetic_Dynamic_TRAINING_nomad import Genetic_Dyn_Algorithm
from Worm_Env.weight_dict import dict
from graphing import graph_comparison,graph
from util.write_read_txt import write_array_to_file, read_array_from_file, read_arrays_from_csv_pandas,delete_arrays_csv_if_exists
import numpy as np 
from util.dist_dict_calc import dist_calc

population_size = 8**3
generations = 400
training_interval = 250
total_episodes = 1  # Number of episodes per evaluation
food_patterns = [5]

##DISTANCE FROM MOTOR ON Y
##CHANGE IN SYNAPTIC STRENGTH ANALOG
##que te vaya bien
##gandul
##constantly validate results
##start from a prexisting model and validate your code by recontruction of results

clean_env = 0
run_gen = 1
graphing = 1

##normalize reward

##nomad algorithm

values_list = []
for sub_dict in dict.values():
    values_list.extend(sub_dict.values())

if clean_env:
    print("Clearning Environment ")
    delete_arrays_csv_if_exists()
if run_gen:
    print("Running Genetic Algoritm")
    env = WormSimulationEnv(num_worms=population_size)
    ga = Genetic_Dyn_Algorithm(population_size, food_patterns, total_episodes, training_interval,values_list)
    best_weight_matrix = ga.run(env, generations)
    print("Best weight matrix found:", best_weight_matrix)   
if graphing:
    old_wm = np.array(values_list)
    print("Graphing Results")
    dist_dict = dist_calc(dict)
    gen=0
    for a in (read_arrays_from_csv_pandas("arrays.csv")):
        graph(np.array(a), dict, gen,old_wm,dist_dict)
        gen+=1


#best_weight_matrix = read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/cheaty_worm.txt")
#graph_comparison(best_weight_matrix,np.array(values_list),dict)
#   graph(best_weight_matrix,dict)
#graph(np.array(values_list),dict)

# Run the simulation with the best weight matrix
#GeneticRUN(best_weight_matrix, training_interval).run(env)
#env.close()
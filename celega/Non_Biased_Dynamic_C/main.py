from Worm_Env.celegan_env import WormSimulationEnv
#from Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm
from Genetic_Dynamic_TRAINING_nomad import Genetic_Dyn_Algorithm
#from Figure_gen import Genetic_Dyn_Algorithm
#from Graph_fitness_over_time import Genetic_Dyn_Algorithm
from Worm_Env.weight_dict import dict
from graphing import graph_comparison,graph
from util.write_read_txt import write_array_to_file, read_array_from_file, read_arrays_from_csv_pandas,delete_arrays_csv_if_exists
import numpy as np 
from util.dist_dict_calc import dist_calc
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names
from util.movie import compile_images_to_video
# Set up logging to only display ERROR and CRITICAL messages
## guided evolutionary nomadic search
population_size = 64
generations = 100
training_interval = 250
total_episodes = 1  # Number of episodes per evaluation
food_patterns = [5]
path = "/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C"
csv_name = "1:50:00_nomad_tri_669_25"
#[0, 16383] for both
#[1, 16382] for traingle
##imagine you are explaining to a person when writing paper
##CHANGE IN SYNAPTIC STRENGTH ANALOG
##que te vaya bien
##gandul
##ssh miles@upf.gdsa
##constantly validate results
##start from a prexisting model and validate your code by recontruction of results

clean_env = 0
run_gen = 0
graphing = 1
testing_mode = 0

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
    env = WormSimulationEnv()
    ga = Genetic_Dyn_Algorithm(population_size, food_patterns, total_episodes, training_interval,values_list)
    best_weight_matrix = ga.run(env, generations)
    print("Best weight matrix found:", best_weight_matrix)   
if graphing:
    old_wm = np.array(values_list)
    print("Graphing Results")
    dist_dict = dist_calc(dict)
    gen=0
    file_name= csv_name
    for a in (read_arrays_from_csv_pandas(path+"/Results/Results_for_paper/"+file_name+".csv")):
         graph(np.array(a), dict, gen,old_wm,dist_dict)
         gen+=1
    compile_images_to_video(path+"/tmp_img","_".join(file_name.split("_")[1:])+".mp4",fps=3)

    



#best_weight_matrix = read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/cheaty_worm.txt")
#graph_comparison(best_weight_matrix,np.array(values_list),dict)
#   graph(best_weight_matrix,dict)
#graph(np.array(values_list),dict)

# Run the simulation with the best weight matrix
#GeneticRUN(best_weight_matrix, training_interval).run(env)
#env.close()
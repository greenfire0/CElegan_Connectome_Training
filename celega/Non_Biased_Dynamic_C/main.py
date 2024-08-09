from Worm_Env.celegan_env import WormSimulationEnv
from Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm
#from Genetic_Dynamic_TRAINING_nomad import Genetic_Dyn_Algorithm
#from Figure_gen import Genetic_Dyn_Algorithm
#from Graph_fitness_over_time import Genetic_Dyn_Algorithm
from Worm_Env.weight_dict import dict
from graphing import graph_comparison,graph
from util.write_read_txt import write_array_to_file, read_array_from_file, read_arrays_from_csv_pandas,delete_arrays_csv_if_exists
import numpy as np 
from util.dist_dict_calc import dist_calc
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names
from util.movie import compile_images_to_video
from util.findmotor_ind import find_motor_ind,get_indicies_to_change
from util.read_from_xls import combine_neuron_data 
population_size = 64
generations = 100
training_interval = 250
total_episodes = 1  # Number of episodes per evaluation
food_patterns = [3] 

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
freeze_indicies= 0
run_gen = 1
graphing = 0
testing_mode = 0

##normalize reward
     
##nomad algorithm
frozen_indices = []
values_list = []



# Example usage

for sub_dict in dict.values():
    values_list.extend(sub_dict.values())
values_list=np.array(values_list)
length = (len(values_list))


if clean_env:
    print("Clearning Environment ")
    delete_arrays_csv_if_exists()
if freeze_indicies:
    frozen_indices=find_motor_ind(dict,muscles)
    print("Froze",len(frozen_indices),"indicies")
    assert (len(np.where(values_list[frozen_indices]<0)[0])) ==125 ##remove if changing connectome


if run_gen:
    indicies_to_change = (get_indicies_to_change(frozen_indices,length))


    print("Running Genetic Algoritm")
    env = WormSimulationEnv()
    ga = Genetic_Dyn_Algorithm(population_size, food_patterns, total_episodes, training_interval,values_list,indicies_to_change,matrix_shape=length)
    best_weight_matrix = ga.run(env, generations)
    assert np.array_equal(np.array(best_weight_matrix)[frozen_indices],values_list[frozen_indices])
    print("Best weight matrix found:", best_weight_matrix)   
if graphing:
    path = "/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C"
    csv_name = "13:00:35_NOMAD_BOTH_853_100"
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
from Worm_Env.celegan_env import WormSimulationEnv
#from Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm
from Genetic_Dynamic_TRAINING_nomad import Genetic_Dyn_Algorithm
#from Graph_fitness_over_time import Genetic_Dyn_Algorithm
from Worm_Env.weight_dict import dict
from graphing import graph_comparison,graph
from util.write_read_txt import write_array_to_file, read_array_from_file, read_arrays_from_csv_pandas,delete_arrays_csv_if_exists
import numpy as np 
from util.dist_dict_calc import dist_calc
from util.movie import compile_images_to_video
from util.findmotor_ind import find_motor_ind,get_indicies_to_change
from util.read_from_xls import combine_neuron_data 
from util.write_read_txt import flatten_dict_values,read_excel

population_size = 64
generations = 100
training_interval = 250
total_episodes = 1 ##unless environment changes this can just be 1
food_patterns = [3] 

#Live to Create, Create to Die
#Miles Churchland August 13th 2024
#Contact: Mileschurchland@gmail.com

##delete the first one its corrupted


clean_env = 0
freeze_indicies= 0
run_gen = 1
graphing = 0
testing_mode = 0


frozen_indices = []
values_list = []


for sub_dict in dict.values():
    values_list.extend(sub_dict.values())
values_list=np.array(values_list)
length = (len(values_list))



if clean_env:
    print("Clearning Environment ")
    delete_arrays_csv_if_exists()
if run_gen:
    print("Running Genetic Algoritm")
    env = WormSimulationEnv()
    ga = Genetic_Dyn_Algorithm(population_size, food_patterns, total_episodes, training_interval,values_list,matrix_shape=length)
    best_weight_matrix = ga.run(env, generations)
    print("Best weight matrix found:", best_weight_matrix)   

    del ga,env
    env = WormSimulationEnv()
    ga = Genetic_Dyn_Algorithm(population_size, food_patterns, total_episodes, training_interval,values_list,matrix_shape=length)
    best_weight_matrix = ga.run(env, generations)
    print("Best weight matrix found:", best_weight_matrix)   
if graphing:
    path = "/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C" ##path to dir here idk
    csv_name = "13:00:35_NOMAD_BOTH_853_100" #name of csv importantly wthout.csv not sure why
    old_wm = np.array(values_list) 
    print("Graphing Results")
    dist_dict = dist_calc(dict) 
    gen=0
    file_name= csv_name
    for a in (read_arrays_from_csv_pandas(path+"/Results/"+file_name+".csv")):
         graph(np.array(a), dict, gen,old_wm,dist_dict)
         gen+=1
    compile_images_to_video(path+"/tmp_img","_".join(file_name.split("_")[1:])+".mp4",fps=3)

    
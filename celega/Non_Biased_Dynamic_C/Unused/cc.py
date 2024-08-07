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
# Set up logging to only display ERROR and CRITICAL messages
## guided evolutionary nomadic search
population_size = 64
generations = 10000
training_interval = 250
total_episodes = 1  # Number of episodes per evaluation
food_patterns = [5]

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
run_gen = 1
graphing = 0
testing_mode = 0

##normalize reward
     
##nomad algorithm
import matplotlib.pyplot as plt
values_list = []
for sub_dict in dict.values():
    values_list.extend(sub_dict.values())

values_array = np.array(values_list)

# Find indices of values greater than 25
indices = np.where(values_array > 25)
print(np.max(values_array))
print(np.mean(values_array))
# Get values greater than 25
values_greater_than_25 = values_array[indices]

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(values_array, bins=50, alpha=0.7, label='All values')
plt.hist(values_greater_than_25, bins=50, alpha=0.7, label='Values > 25', color='red')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Values in the Dictionary')
plt.legend()
plt.show()


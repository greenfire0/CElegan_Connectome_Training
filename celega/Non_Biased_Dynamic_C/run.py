from Genetic_running import GeneticRUN
from celegan_env import WormSimulationEnv
from util.write_read_txt import read_array_from_file

try:
    read_array = read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/cheaty_worm.txt")
    assert len(read_array) ==3689, "file not read correctly, missing weights or incorrect file"
except:
    print("input weights not read correctly")
training_interval = 150  # Train the agent every 25 steps
total_episodes = 1  # Number of episodes per evaluation
train_params =3689 #number of connections
env = WormSimulationEnv(num_worms=1)

GeneticRUN(read_array, total_episodes, training_interval).run(env)
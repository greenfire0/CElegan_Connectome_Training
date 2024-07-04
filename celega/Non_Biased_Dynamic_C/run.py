import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Non_Biased_Dynamic_C.Genetic_running import GeneticRUN
from Non_Biased_Dynamic_C.celegan_env import WormSimulationEnv


def read_array_from_file(filename):
    """Read an array from a text file, assuming each line is a separate element."""
    try:
        with open(filename, 'r') as file:
            array = [float(line.strip()) for line in file]
        print(f"Array successfully read from {filename}")
        return array
    except Exception as e:
        print(f"An error occurred while reading from the file: {e}")
        return []
    
read_array = read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/cheaty_worm.txt")
population_size = 16
generations = 500
mutation_rate = 0.6
print(len(read_array))
training_interval = 150  # Train the agent every 25 steps
total_episodes = 1  # Number of episodes per evaluation
train_params =3689 #number of connections
env = WormSimulationEnv(num_worms=population_size)

GeneticRUN(read_array, total_episodes, training_interval).run(env)
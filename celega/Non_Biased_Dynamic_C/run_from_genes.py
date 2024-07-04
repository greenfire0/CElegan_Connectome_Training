from Worm_Env.trained_connectome import wormConnectone
from Worm_Env.celegan_env import WormSimulationEnv
from util.write_read_txt import read_array_from_file

class GeneticRUN:
    def __init__(self, population_dna, training_interval=25):
        self.population_dna = population_dna

        self.training_interval = training_interval
        self.population = self.initialize_population()

    def initialize_population(self):
        population = [wormConnectone(weight_matrix=self.population_dna)]
        return population

    def evaluate_fitness(self, candidate,worm_num, env):
            env.reset()
            candidate.modify_combined_weights()
            observation = env._get_observations()
            for _ in range(self.training_interval):
                movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food)
                next_observation, reward, done, _ = env.step(movement,worm_num,candidate)
                env.render(worm_num)                
                observation = next_observation

    def run(self, env, generations=50):
        for _ in range(generations):
            for worm_num, candidate in enumerate(self.population):
                self.evaluate_fitness(candidate, worm_num, env)
                candidate.createpostSynaptic()


try:
    read_array = read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/cheaty_worm.txt")
    assert len(read_array) ==3689, "file not read correctly, missing weights or incorrect file"
except:
    print("input weights not read correctly")
training_interval = 150  # Train the agent every 25 steps
train_params =3689 #number of connections
env = WormSimulationEnv(num_worms=1)

GeneticRUN(read_array,  training_interval).run(env)
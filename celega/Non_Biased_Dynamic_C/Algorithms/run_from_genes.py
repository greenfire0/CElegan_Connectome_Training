from Worm_Env.connectome import WormConnectome
from Worm_Env.celegan_env import WormSimulationEnv
import numpy as np
import matplotlib
from util.write_read_txt import read_arrays_from_csv_pandas
matplotlib.use('TkAgg')  # TkAgg is a commonly used backend for interactive sessions
from Worm_Env.weight_dict import dict,muscles,muscleList,mLeft,mRight,all_neuron_names


class GeneticRUN:
    def __init__(self, random_dna, training_interval=25):
        self.random_dna = random_dna
        self.training_interval = training_interval
        self.population_random = self.initialize_population()

    def initialize_population(self):
        population = []
        for a in [self.random_dna]:
            population.append(WormConnectome(weight_matrix=np.array(a, dtype=np.float32), all_neuron_names=all_neuron_names))
        return population

    def evaluate_fitness(self, candidate, env, prob_type):
        sum_rewards = 0
        for a in prob_type:
            env.reset(a)
            candidate = WormConnectome(weight_matrix=(candidate.weight_matrix),all_neuron_names=all_neuron_names)
            for _ in range(1):  # total_episodes
                observation = env._get_observations()
                for _ in range(self.training_interval):  # training_interval
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    env.render()
                    observation = next_observation
                    sum_rewards+=reward
        return sum_rewards

    def run(self, env, generations=100):
        pattern =  [3]
        for _ in range(generations):
            for  candidate in (self.population_random):
                random_reward = self.evaluate_fitness(candidate, env,pattern)
                print(f"Random worm reward: {random_reward}")

if __name__ == "__main__":
    test_genome = np.array(read_arrays_from_csv_pandas("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/Results/50_nomad_sq.csv"))
    test_genome = test_genome[len(test_genome)-1]
    print(len(test_genome))
    values_list = []
    for sub_dict in dict.values():
        values_list.extend(sub_dict.values())
    print(len(np.where(test_genome != values_list))," Number of Differences from original genome")
    test_genome=test_genome
    assert len(test_genome) == 3682, "Worm file not read correctly, missing weights or incorrect file"


    training_interval = 250  # Train the agent every 250 steps
    env = WormSimulationEnv() ##make sure to manually change pattern
    GeneticRUN(test_genome, training_interval).run(env)

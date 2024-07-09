from Worm_Env.trained_connectome import wormConnectone
from Worm_Env.celegan_env import WormSimulationEnv
from util.write_read_txt import read_array_from_file

class GeneticRUN:
    def __init__(self, random_dna, training_interval=25):
        self.random_dna = random_dna

        self.training_interval = training_interval
        self.population_random = self.initialize_population(self.random_dna)

    def initialize_population(self, dna):
        population = [wormConnectone(weight_matrix=dna)]
        return population

    def evaluate_fitness(self, candidate, worm_num, env,pat):
        for _ in range(1):
            for a in pat:
                env.reset(a)
                candidate.modify_combined_weights()
                observation = env._get_observations()
                rewards = []
                for _ in range(self.training_interval):
                    movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food)
                    next_observation, reward, done, _ = env.step(movement, worm_num, candidate)
                    env.render(worm_num)
                    observation = next_observation
                    rewards.append(reward)
        return sum(rewards) /len(rewards)

    def run(self, env, generations=100):
        pattern =  ["triangle"]
        for _ in range(generations):
            # Evaluate pattern worm

                
            # Evaluate random worm
            for worm_num, candidate in enumerate(self.population_random):
                random_reward = self.evaluate_fitness(candidate, worm_num, env,pattern)
                #candidate.createpostSynaptic()

            # Print the difference in rewards
            print(f"Random worm reward: {random_reward}")

try:
    random_dna = read_array_from_file("/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/cheaty_worm3.txt")
    assert len(random_dna) == 3689, "Random worm file not read correctly, missing weights or incorrect file"
except:
    print("Input weights not read correctly")

training_interval = 250  # Train the agent every 25 steps
env = WormSimulationEnv(num_worms=1)

GeneticRUN(random_dna, training_interval).run(env)

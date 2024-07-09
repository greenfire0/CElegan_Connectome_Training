import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from graphing import graph,graph_comparison
from Worm_Env.weight_dict import dict
from tqdm import tqdm
import time 
muscles = ['MVU', 'MVL', 'MDL', 'MVR', 'MDR']

muscleList = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']

mLeft = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']
mRight = ['MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']
# Used to accumulate muscle weighted values in body muscles 07-23 = worm locomotion
musDleft = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23']
musVleft = ['MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']
musDright = ['MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23']
musVright = ['MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']
all_neuron_names = [
    'ADAL', 'ADAR', 'ADEL', 'ADER', 'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AFDL', 'AFDR',
    'AIAL', 'AIAR', 'AIBL', 'AIBR', 'AIML', 'AIMR', 'AINL', 'AINR', 'AIYL', 'AIYR',
    'AIZL', 'AIZR', 'ALA', 'ALML', 'ALMR', 'ALNL', 'ALNR', 'AQR', 'AS1', 'AS10', 'AS11',
    'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'ASEL', 'ASER', 'ASGL', 'ASGR',
    'ASHL', 'ASHR', 'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR', 'AUAL', 'AUAR', 'AVAL',
    'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER', 'AVFL', 'AVFR', 'AVG', 'AVHL',
    'AVHR', 'AVJL', 'AVJR', 'AVKL', 'AVKR', 'AVL', 'AVM', 'AWAL', 'AWAR', 'AWBL', 'AWBR',
    'AWCL', 'AWCR', 'BAGL', 'BAGR', 'BDUL', 'BDUR', 'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR', 'DA1',
    'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9', 'DB1', 'DB2', 'DB3', 'DB4', 'DB5',
    'DB6', 'DB7', 'DD1', 'DD2', 'DD3', 'DD4', 'DD5', 'DD6', 'DVA', 'DVB', 'DVC', 'FLPL', 'FLPR',
    'HSNL', 'HSNR', 'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6', 'IL1DL', 'IL1DR', 'IL1L',
    'IL1R', 'IL1VL', 'IL1VR', 'IL2L', 'IL2R', 'IL2DL', 'IL2DR', 'IL2VL', 'IL2VR', 'LUAL', 'LUAR',
    'M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5',  'MCL', 'MCR', 'MDL01', 'MDL02', 'MDL03',
    'MDL04', 'MDL05', 'MDL06', 'MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14',
    'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MDL24', 'MDR01',
    'MDR02', 'MDR03', 'MDR04', 'MDR05', 'MDR06', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12',
    'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDR21', 'MDR22', 'MDR23',
    'MDR24', 'MI', 'MVL01', 'MVL02', 'MVL03', 'MVL04', 'MVL05', 'MVL06', 'MVL07', 'MVL08', 'MVL09',
    'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20',
    'MVL21', 'MVL22', 'MVL23', 'MVR01', 'MVR02', 'MVR03', 'MVR04', 'MVR05', 'MVR06', 'MVR07', 'MVR08',
    'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19',
    'MVR20', 'MVR21', 'MVR22', 'MVR23', 'MVR24', 'MVULVA', 'NSML', 'NSMR', 'OLLL', 'OLLR', 'OLQDL',
    'OLQDR', 'OLQVL', 'OLQVR', 'PDA', 'PDB', 'PDEL', 'PDER', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'PHCL',
    'PHCR', 'PLML', 'PLMR', 'PLNL', 'PLNR', 'PQR', 'PVCL', 'PVCR', 'PVDL', 'PVDR', 'PVM', 'PVNL', 'PVNR',
    'PVPL', 'PVPR', 'PVQL', 'PVQR', 'PVR', 'PVT', 'PVWL', 'PVWR', 'RIAL', 'RIAR', 'RIBL', 'RIBR', 'RICL',
    'RICR', 'RID', 'RIFL', 'RIFR', 'RIGL', 'RIGR', 'RIH', 'RIML', 'RIMR', 'RIPL', 'RIPR', 'RIR', 'RIS',
    'RIVL', 'RIVR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMED', 'RMEL', 'RMER', 'RMEV',
    'RMFL', 'RMFR', 'RMGL', 'RMGR', 'RMHL', 'RMHR', 'SAADL', 'SAADR', 'SAAVL', 'SAAVR', 'SABD', 'SABVL',
    'SABVR', 'SDQL', 'SDQR', 'SIADL', 'SIADR', 'SIAVL', 'SIAVR', 'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR',
    'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'URADL', 'URADR', 'URAVL',
    'URAVR', 'URBL', 'URBR', 'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL', 'URYVR', 'VA1', 'VA10', 'VA11',
    'VA12', 'VA2', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7', 'VA8', 'VA9', 'VB1', 'VB10', 'VB11', 'VB2', 'VB3',
    'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VC1', 'VC2', 'VC3', 'VC4', 'VC5', 'VC6', 'VD1', 'VD10',
    'VD11', 'VD12', 'VD13', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9'
]



class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, matrix_shape, mutation_rate=0.5, total_episodes=10, training_interval=25, genome=None):
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.mutation_rate = mutation_rate
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.population = self.initialize_population(genome)

    def initialize_population(self, genome=None):
        population = []
        if genome is None:
            for _ in range(self.population_size):
                population.append(WormConnectome(weight_matrix=np.random.randn(self.matrix_shape) * 10))
        else:
            for _ in range(self.population_size):
                population.append(WormConnectome(weight_matrix=np.array(genome, dtype=float), all_neuron_names=all_neuron_names))
        return population

    def evaluate_fitness(self, candidate, worm_num, env, prob_type):
        cumulative_rewards = []
        for a in prob_type:
            env.reset(a)
            for _ in range(self.total_episodes):
                observation = env._get_observations()
                done = False
                for _ in range(self.training_interval):
                    movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, done = env.step(movement, worm_num, candidate)
                    #env.render(worm_num)
                    observation = next_observation
                    cumulative_rewards.append(reward)
        return np.sum(cumulative_rewards)

    def select_parents(self, fitnesses, num_parents):
        parents = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents]

    def crossover(self, parents, fitnesses, num_offspring):
        offspring = []
        parent_fitnesses = np.array([fitnesses[i] for i in np.argsort(fitnesses)[-len(parents):]])
        fitness_probs = parent_fitnesses / np.sum(parent_fitnesses)

        for _ in range(num_offspring):
            parent1 = np.random.choice(parents, p=fitness_probs)
            parent2 = np.random.choice(parents, p=fitness_probs)
            total_length = len(parent1.weight_matrix)
            crossover_prob = fitness_probs[parents.index(parent1)] / (fitness_probs[parents.index(parent1)] + fitness_probs[parents.index(parent2)])
            splice_point = int(crossover_prob * total_length)
            splice_point = max(1, min(splice_point, total_length - 1))
            child_weight_matrix = np.concatenate((parent1.weight_matrix[:splice_point], parent2.weight_matrix[splice_point:]))
            offspring.append(WormConnectome(weight_matrix=child_weight_matrix,all_neuron_names=all_neuron_names))
        return offspring

    def mutate(self, offspring, n=10):
        for child in offspring:
            if np.random.rand() < self.mutation_rate:
                flat_weights = child.weight_matrix.flatten()
                indices_to_mutate = np.random.choice(len(flat_weights), size=n, replace=False)
                new_values = np.random.uniform(low=-20, high=20, size=n)
                flat_weights[indices_to_mutate] = new_values
                child.weight_matrix = flat_weights.reshape(self.matrix_shape)
        return offspring

    def mutate_addition(self, offspring):
        for child in offspring:
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.uniform(low=-1, high=1, size=self.matrix_shape)
                child.weight_matrix += mutation
        return offspring

    @staticmethod
    @ray.remote
    def evaluate_fitness_ray(candidate_weights,nur_name, worm_num, env, prob_type, mLeft, mRight, muscleList, muscles):
        candidate = WormConnectome(weight_matrix=candidate_weights,all_neuron_names=nur_name)
        cumulative_rewards = []
        for a in prob_type:
            env.reset(a)
            for _ in range(10):  # total_episodes
                observation = env._get_observations()
                done = False
                for _ in range(25):  # training_interval
                    movement = candidate.move(observation[worm_num][0], env.worms[worm_num].sees_food, mLeft, mRight, muscleList, muscles)
                    next_observation, reward, done = env.step(movement, worm_num, candidate)
                    #env.render(worm_num)
                    observation = next_observation
                    cumulative_rewards.append(reward)
        return np.sum(cumulative_rewards)

    def run(self, env, old_wm, generations=50):
        ray.init(ignore_reinit_error=True)
        pattern = [5]

        try:
            for generation in tqdm(range(generations), desc="Generations"):
                start_time = time.time()
                fitnesses = ray.get([self.evaluate_fitness_ray.remote(candidate.weight_matrix, all_neuron_names, worm_num, env, pattern, mLeft, mRight, muscleList, muscles) for worm_num, candidate in enumerate(self.population)])
                best_index = np.argmax(fitnesses)
                best_candidate = self.population[best_index]
                best_fitness = fitnesses[best_index]
                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                self.population = self.select_parents(fitnesses, self.population_size // 2)
                offspring = self.crossover(self.population, fitnesses, self.population_size - len(self.population))
                offspring = self.mutate(offspring)
                offspring = self.mutate_addition(offspring)
                self.population.extend(offspring)
            fitnesses = [self.evaluate_fitness(candidate, worm_num, env, pattern) for worm_num, candidate in enumerate(self.population)]
            best_index = np.argmax(fitnesses)
            return self.population[best_index].weight_matrix
        finally:
            ray.shutdown()
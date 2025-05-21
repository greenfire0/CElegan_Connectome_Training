import numpy as np
import ray
from numba import njit
from Worm_Env.trained_connectome import WormConnectome
from Worm_Env.weight_dict import dict, muscles, muscleList, mLeft, mRight, all_neuron_names
import PyNomad
from tqdm import tqdm
import csv
from util.write_read_txt import read_arrays_from_csv_pandas

@njit
def motor_control(post_synaptic, mLeft, mRight, muscleList, accumleft, accumright):
    """
    Resets post_synaptic activity and returns accumleft and accumright.
    """
    for muscle in muscleList:
        # Reset all activity
        post_synaptic[muscle][:] = 0.0
    return accumleft, accumright

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size1=400, hidden_size2=8, output_size=2):
        """
        Network Architecture:
        Input(2) -> Hidden1(400) -> Hidden2(8) -> Output(2)

        Weights:
        Input->Hidden1: 2 * 400 = 800
        Hidden1->Hidden2: 400 * 8 = 3200
        Hidden2->Output: 8 * 2 = 16
        Total ~ 4016 weights.

        Activation: Sigmoid
        Loss: MSE
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Weight initialization (no biases)
        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * 0.01
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * 0.01
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * 0.01

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self, X):
        self.z1 = X.dot(self.W1)  # (batch, hidden1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1.dot(self.W2)  # (batch, hidden2)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = self.a2.dot(self.W3)  # (batch, output)
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        # Output layer error
        dz3 = (self.a3 - y) * (self.a3*(1-self.a3))
        dW3 = (1/m)*self.a2.T.dot(dz3)

        # Hidden2 error
        dz2 = dz3.dot(self.W3.T) * (self.a2*(1-self.a2))
        dW2 = (1/m)*self.a1.T.dot(dz2)

        # Hidden1 error
        dz1 = dz2.dot(self.W2.T) * (self.a1*(1-self.a1))
        dW1 = (1/m)*X.T.dot(dz1)

        # Update weights
        self.W1 -= lr*dW1
        self.W2 -= lr*dW2
        self.W3 -= lr*dW3

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[5], total_episodes=0, training_interval=250, genome=None, matrix_shape=3689, indicies=[]):
        self.population_size = population_size
        self.indicies = indicies
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        assert(len(genome) == matrix_shape)
        self.population = self.initialize_population(genome)

        # Neural network with ~4000 weights
        self.nn = NeuralNetwork()

    def initialize_population(self, genome=None):
        population = []
        population.append(WormConnectome(weight_matrix=np.array(genome, dtype=np.float32), all_neuron_names=all_neuron_names))
        for _ in range(self.population_size-1):
            population.append(self.give_random_worm())
        return population

    def give_random_worm(self):
        return WormConnectome(weight_matrix=np.random.uniform(low=-20, high=20, size=self.matrix_shape).astype(np.float32), all_neuron_names=all_neuron_names)

    def movement(self, obs_value, sees_food, post_synaptic):
        """
        NN outputs accumleft and accumright directly.
        """
        input_data = np.array([[obs_value, float(sees_food)]])
        nn_output = self.nn.forward(input_data)  # shape (1,2)
        accumleft, accumright = nn_output[0,0], nn_output[0,1]
        accumleft, accumright = motor_control(post_synaptic, mLeft, mRight, muscleList, accumleft, accumright)
        return accumleft, accumright

    def custom_evaluate_fitness(self, env, prob_type, episodes, interval):
        sum_rewards = 0
        post_synaptic = {muscle: np.zeros(2, dtype=np.float32) for muscle in muscleList}
        for a in prob_type:
            env.reset(a)
            for _ in range(episodes):
                observation = env._get_observations()
                for _ in range(interval):
                    obs_value = observation[0][0]
                    sees_food = env.worms[0].sees_food
                    accumleft, accumright = self.movement(obs_value, sees_food, post_synaptic)
                    movement = [accumleft, accumright]
                    next_observation, reward, _ = env.step(movement, 0, None)
                    observation = next_observation
                    sum_rewards += reward
        return sum_rewards

    def run(self, env, generations=50, batch_size=32):
        ray.init(
            ignore_reinit_error=True,
            object_store_memory=15 * 1024 * 1024 * 1024,
            num_cpus=16,
        )

        # Train the neural network for demonstration
        X_train = np.random.randn(100, self.nn.input_size)
        y_train = np.random.randn(100, self.nn.output_size)

        for epoch in range(5):
            y_pred = self.nn.forward(X_train)
            loss = self.nn.loss(y_pred, y_train)
            self.nn.backward(X_train, y_train, lr=0.01)
            print(f"Neural Net Training Epoch {epoch+1}, Loss: {loss}")

        # Evaluate fitness using the custom method
        fitness = self.custom_evaluate_fitness(env, self.food_patterns, self.total_episodes, self.training_interval)
        print("Initial fitness (custom):", fitness)

        # Here you can integrate the GA steps (selection, crossover, mutation).
        # This snippet focuses on the neural network weight count approximation.

        ray.shutdown()
        return

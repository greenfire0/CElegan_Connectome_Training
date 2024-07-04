import numpy as np

class C_Elegans_Agent:
    def __init__(self, obs_space, action_space, lr=0.01, epsilon=0.1):
        self.obs_space = obs_space.shape[0]
        self.action_space = action_space.n
        self.lr = lr
        self.gamma = 0.99  # Discount factor for reward
        self.epsilon = epsilon  # Exploration parameter

        # Define a neural network with four hidden layers
        self.weights = self._init_weights([self.obs_space, 32, 32, 32, self.action_space])
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []

    def _init_weights(self, layers):
        weights = []
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])  # He initialization
            weights.append(weight)
        return weights

    def _forward(self, x):
        for weight in self.weights[:-1]:
            x = np.tanh(np.dot(x, weight))
        logits = np.dot(x, self.weights[-1])
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def choose_action(self, observation):
        if np.random.rand() < self.epsilon:
            # Randomly choose action
            action = np.random.randint(0, self.action_space)
        else:
            # Choose action according to the policy
            probs = self._forward(observation)
            action = np.random.choice(range(self.action_space), p=probs)
        return action

    def store_transition(self, observation, action, reward):
        self.episode_observations.append(observation)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def _discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float64)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discounted[i] = cumulative
        return discounted

    def learn(self):
        discounted_rewards = self._discount_rewards(self.episode_rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        for t in range(len(self.episode_observations)):
            obs = self.episode_observations[t]
            action = self.episode_actions[t]
            reward = discounted_rewards[t]

            probs = self._forward(obs)
            dsoftmax = probs
            dsoftmax[action] -= 1
            dsoftmax *= reward

            deltas = [dsoftmax]
            layer_activations = [obs]

            # Forward pass to store activations
            for weight in self.weights[:-1]:
                obs = np.tanh(np.dot(obs, weight))
                layer_activations.append(obs)

            for i in range(len(self.weights) - 1, 0, -1):
                if i == len(self.weights) - 1:
                    deltas.append(np.dot(deltas[-1], self.weights[i].T))
                else:
                    deltas.append(np.dot(deltas[-1], self.weights[i].T) * (1 - layer_activations[i] ** 2))

            deltas = deltas[::-1]

            for i in range(len(self.weights)):
                if i == 0:
                    grad = np.outer(layer_activations[i], deltas[i])
                else:
                    grad = np.outer(layer_activations[i], deltas[i])

                self.weights[i] -= self.lr * grad

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

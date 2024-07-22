import numpy as np
import random

class PolicyGradientAgent:
    def __init__(self, obs_space, action_space, lr=0.1):
        self.obs_space = obs_space.shape[0]
        self.action_space = action_space.n
        self.lr = lr
        # Define the layers with 5 nodes per layer and 2 extra layers
        self.weights = self._init_weights([self.obs_space, 5, 5, 5, 5, 5, self.action_space])
        # Increase the weight for the chosen action to make its initial probability higher
        self.weights[-1][0] += 0.5  # Adjust this value as needed
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []

    def _init_weights(self, layers):
        weights = []
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i + 1]) * 0.1
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
        probs = self._forward(observation)
        #print(probs)
        action = np.random.choice(range(self.action_space), p=probs)
        return action

    def store_transition(self, observation, action, reward):
        self.episode_observations.append(observation)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def learn(self):
        discounted_rewards = self._discount_and_normalize_rewards()
        for t in range(len(self.episode_observations)):
            obs = self.episode_observations[t]
            action = self.episode_actions[t]
            reward = discounted_rewards[t]

            probs = self._forward(obs)
            dsoftmax = probs
            dsoftmax[action] -= 1
            dlog = dsoftmax * reward

            deltas = [dlog]
            for i in range(len(self.weights) - 1, 0, -1):
                if i == len(self.weights) - 1:
                    deltas.append(np.dot(deltas[-1], self.weights[i].T))
                else:
                    deltas.append(np.dot(deltas[-1], self.weights[i].T) * (1 - np.tanh(np.dot(obs, self.weights[i])) ** 2))

            deltas = deltas[::-1]

            for i in range(len(self.weights)):
                if i == 0:
                    grad = np.outer(obs, deltas[i])
                else:
                    grad = np.outer(obs if i == 0 else np.tanh(np.dot(obs, self.weights[i - 1])), deltas[i])

                self.weights[i] -= self.lr * grad

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

    def _discount_and_normalize_rewards(self, gamma=0.99):
        discounted_rewards = np.zeros_like(self.episode_rewards, dtype=np.float64)
        cumulative = 0.0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * gamma + self.episode_rewards[t]
            discounted_rewards[t] = cumulative

        # Compute the mean and standard deviation
        mean_reward = np.mean(discounted_rewards)
        std_reward = np.std(discounted_rewards)

        # Add a small epsilon value to avoid division by zero or very small standard deviation
        epsilon = 1e-8

        # Normalize the rewards
        if std_reward > 0:
            discounted_rewards -= mean_reward
            discounted_rewards /= std_reward
        else:
            discounted_rewards -= mean_reward
            discounted_rewards /= (std_reward + epsilon)
        return discounted_rewards

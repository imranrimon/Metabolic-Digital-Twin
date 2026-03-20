import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from metabolic_twin.utils.training import ValidationCheckpoint, load_model_state, progress, update_progress


CHECKPOINT_PATH = "f:/Diabetics Project/metabolic_policy.pth"


class MetabolicEnv:
    """
    A simulated environment for metabolic control.
    State: current glucose.
    Action: meal composition bucket.
    Reward: time-in-range and minimized spike.
    """

    def __init__(self, simulator_model=None):
        self.simulator = simulator_model
        self.state = 110.0

    def step(self, action):
        carbs, prot, fat = action
        spike = carbs * 0.5 - (prot * 0.1 + fat * 0.05)
        new_glucose = self.state + spike * 30.0

        reward = 0.0
        if 70 <= new_glucose <= 140:
            reward += 10.0
        else:
            reward -= abs(new_glucose - 110) * 0.1

        self.state = new_glucose
        return np.array([new_glucose]), reward, False, {}

    def reset(self):
        self.state = 110.0
        return np.array([self.state])


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, greedy=False):
        if (not greedy) and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_t)
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state_t)).item()

            state_t = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_t).detach().clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_t), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def rollout_policy(agent, episodes=5, horizon=10, greedy=True):
    env = MetabolicEnv()
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(horizon):
            action = agent.act(state, greedy=greedy)
            carb_val = (action + 1) * 0.2
            real_action = [carb_val, 0.2, 0.1]
            next_state, reward, done, _ = env.step(real_action)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return float(np.mean(rewards))


def train_metabolic_rl():
    print("\n--- Training Offline Metabolic RL Policy (SOTA Control) ---")
    env = MetabolicEnv()
    agent = DQNAgent(state_dim=1, action_dim=5)
    checkpoint = ValidationCheckpoint(CHECKPOINT_PATH, metric_name="val_reward", mode="max")

    episodes = 50
    episode_bar = progress(range(1, episodes + 1), desc="MetabolicRL episodes")
    for episode in episode_bar:
        state = env.reset()
        total_reward = 0.0
        for _ in range(10):
            action = agent.act(state)
            carb_val = (action + 1) * 0.2
            real_action = [carb_val, 0.2, 0.1]
            next_state, reward, done, _ = env.step(real_action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(32)
        val_reward = rollout_policy(agent, episodes=5, horizon=10, greedy=True)
        checkpoint.update(
            agent.model,
            episode,
            val_reward,
            extra_metadata={"epsilon": float(agent.epsilon), "train_reward": float(total_reward)},
        )
        update_progress(episode_bar, train_reward=total_reward, val_reward=val_reward, epsilon=agent.epsilon)

    load_model_state(agent.model, CHECKPOINT_PATH, map_location="cpu")
    test_reward = rollout_policy(agent, episodes=10, horizon=10, greedy=True)
    print(f"Best validation reward: {checkpoint.best_metric:.4f} at episode {checkpoint.best_epoch}")
    print(f"Final greedy evaluation reward: {test_reward:.4f}")
    print(f"Metabolic control policy saved to {CHECKPOINT_PATH}.")


if __name__ == "__main__":
    train_metabolic_rl()

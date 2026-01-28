import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class MetabolicEnv:
    """
    A simulated environment for metabolic control.
    State: Current Glucose, HR, and metabolic trend (from Neural CDE).
    Action: Meal Composition [Carbs, Protein, Fat].
    Reward: Time-in-Range (70-140 mg/dl) and minimized spike.
    """
    def __init__(self, simulator_model=None):
        self.simulator = simulator_model
        self.state = 110.0 # Baseline glucose
        
    def step(self, action):
        # Action is [Carbs, Protein, Fat] normalized
        carbs, prot, fat = action
        
        # Simple metabolic response model (if NeuralCDE is not yet fully integrated for RL)
        # In a real SOTA system, we pass (state, action) through NeuralCDE
        spike = carbs * 0.5 - (prot * 0.1 + fat * 0.05)
        new_glucose = self.state + spike * 30.0 # Multiplier for effect
        
        # Reward function
        reward = 0
        if 70 <= new_glucose <= 140:
            reward += 10 # TIR bonus
        else:
            reward -= abs(new_glucose - 110) * 0.1 # Penalty for deviation
            
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
            nn.Linear(64, self.action_dim)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_t)
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state_t)).item())
            
            state_t = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_t)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_t), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_metabolic_rl():
    print("\n--- Training Offline Metabolic RL Policy (SOTA Control) ---")
    env = MetabolicEnv()
    agent = DQNAgent(state_dim=1, action_dim=5) # 5 meal types
    
    episodes = 50
    for e in range(episodes):
        state = env.reset()
        total_r = 0
        for time in range(10): # 10 sequential meals
            action = agent.act(state)
            # Map action back to Carb intensity
            carb_val = (action + 1) * 0.2 
            real_action = [carb_val, 0.2, 0.1]
            next_state, reward, done, _ = env.step(real_action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
        
        agent.replay(32)
        if (e+1) % 10 == 0:
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_r:.4f}, Epsilon: {agent.epsilon:.2f}")

    torch.save(agent.model.state_dict(), 'f:/Diabetics Project/metabolic_policy.pth')
    print("Metabolic Control Policy saved.")

if __name__ == "__main__":
    train_metabolic_rl()

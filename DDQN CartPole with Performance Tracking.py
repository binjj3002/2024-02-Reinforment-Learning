import os
# Đặt trước các import khác để giải quyết lỗi OMP
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# Neural Network for Q-learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer for experience storing
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# DDQN Agent
class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Main and target networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and hyperparameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        
        self.steps = 0
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train(self):
        # Check if enough samples in replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using Double DQN
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to compute Q values
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

# Test the agent with human mode visualization
def test_ddqn(agent, env, num_episodes=5):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and take action
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Render environment in human mode
            env.render()  # Show environment in the GUI
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        print(f"Test Episode {episode+1}, Total Reward: {total_reward}")

# Main training loop
def train_ddqn():
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DDQNAgent(state_dim, action_dim)
    
    # Training parameters
    num_episodes = 500
    max_steps = 500
    
    # Training loop
    episode_rewards = []
    
    # Đo thời gian
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select and take action
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    # Tính toán thời gian
    total_time = time.time() - start_time
    
    # Close environment
    env.close()
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('DDQN: Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    
    print(f"DDQN Training Time: {total_time:.2f} seconds")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards)}")
    
    # Test the agent in human mode
    env = gym.make('CartPole-v1',render_mode= 'human')  # Create environment again for testing
    test_ddqn(agent, env, num_episodes=5)
    
    return episode_rewards, total_time 

# Run training
if __name__ == "__main__":
    rewards, training_time = train_ddqn()

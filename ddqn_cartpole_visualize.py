import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

# [Giữ nguyên các class DQN, ReplayBuffer, DDQNAgent như trong mã gốc]

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
# Thêm hàm để vẽ đồ thị
def plot_training_metrics(rewards, losses, episodes_to_plot=None):
    """
    Vẽ đồ thị rewards và losses trong quá trình training
    
    Parameters:
    - rewards: Danh sách rewards của từng episode
    - losses: Danh sách losses của từng episode
    - episodes_to_plot: Số lượng episode cuối cùng để vẽ (mặc định là tất cả)
    """
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    if episodes_to_plot:
        rewards = rewards[-episodes_to_plot:]
    plt.plot(rewards, label='Rewards')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot losses
    plt.subplot(1, 2, 2)
    if episodes_to_plot:
        losses = losses[-episodes_to_plot:]
    plt.plot(losses, label='Losses', color='red')
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Sửa đổi hàm training để theo dõi losses
def train_ddqn():
    # Tạo môi trường
    env = gym.make('CartPole-v1')
    
    # Lấy số chiều state và action
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Khởi tạo agent
    agent = DDQNAgent(state_dim, action_dim)
    
    # Các tham số training
    num_episodes = 200
    max_steps = 500
    
    # Theo dõi rewards và losses
    episode_rewards = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(max_steps):
            # Chọn và thực hiện action
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Lưu trữ kinh nghiệm trong replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train agent và theo dõi loss
            current_loss = agent.train()
            if current_loss is not None:
                episode_loss += current_loss
                loss_count += 1
            
            # Cập nhật state và reward
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        # Tính trung bình loss của episode
        avg_episode_loss = episode_loss / loss_count if loss_count > 0 else 0
        
        episode_rewards.append(total_reward)
        episode_losses.append(avg_episode_loss)
        
        # In tiến trình
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Avg Loss: {avg_episode_loss:.4f}, Epsilon: {agent.epsilon:.2f}")
    
    # Đóng môi trường
    env.close()
    
    # Vẽ đồ thị
    plot_training_metrics(episode_rewards, episode_losses)
    
    return episode_rewards, episode_losses

# Chạy training
if __name__ == "__main__":
    rewards, losses = train_ddqn()
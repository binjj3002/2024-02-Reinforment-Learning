import os
# Đặt trước các import khác để giải quyết lỗi OMP
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import pickle

# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(64, output_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        policy = F.softmax(self.actor(shared_features), dim=-1)
        value = self.critic(shared_features)
        return policy, value

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.action_dim = action_dim

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.network(state_tensor)
        action_probs = policy.detach().numpy()[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def compute_returns(self, rewards, dones, values):
        """
        Compute returns using Generalized Advantage Estimation (GAE)
        """
        returns = []
        R = 0
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def train(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        
        # Compute policies and values
        policies, values = self.network(states)
        _, next_values = self.network(next_states)
        
        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones, values.detach().numpy())
        returns = torch.FloatTensor(returns)
        advantages = returns - values.squeeze()
        
        # Actor loss (Policy Gradient)
        action_log_probs = torch.log(policies.gather(1, actions.unsqueeze(1)).squeeze(1))
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Critic loss (Value Approximation)
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_checkpoint(self, filename):
        """Save model and optimizer state"""
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, filename):
        """Load model and optimizer state"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Training function
def train_actor_critic():
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = ActorCriticAgent(state_dim, action_dim)
    checkpoint_file = 'actor_critic_checkpoint.pkl'

    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        print("Loading checkpoint...")
        agent.load_checkpoint(checkpoint_file)
    
    # Training parameters
    num_episodes = 2000
    max_steps = 500
    
    # Tracking metrics
    episode_rewards = []
    
    # Đo thời gian
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        # Lists to store trajectory
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        # Train on the collected trajectory
        if len(states) > 0:
            agent.train(states, actions, rewards, next_states, dones)
        
        # Store metrics
        episode_rewards.append(total_reward)
        
        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"Saving checkpoint at episode {episode + 1}...")
            agent.save_checkpoint(checkpoint_file)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
    # Tính toán thời gian
    total_time = time.time() - start_time
    
    # Close environment
    env.close()
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Actor-Critic: Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    
    print(f"Actor-Critic Training Time: {total_time:.2f} seconds")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards)}")
    
    return episode_rewards, total_time

# Run training
if __name__ == "__main__":
    rewards, training_time = train_actor_critic()

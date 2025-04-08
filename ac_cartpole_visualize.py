import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

# Visualization function
def plot_training_metrics(rewards, losses, episodes_to_plot=None):
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

# Training function
def train_actor_critic():
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = ActorCriticAgent(state_dim, action_dim)
    
    # Training parameters
    num_episodes = 500
    max_steps = 500
    
    # Tracking metrics
    episode_rewards = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        
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
            loss = agent.train(states, actions, rewards, next_states, dones)
            episode_loss += loss
            loss_count += 1
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_losses.append(episode_loss / loss_count if loss_count > 0 else 0)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Avg Loss: {episode_losses[-1]:.4f}")
    
    # Close environment
    env.close()
    
    # Plot training metrics
    plot_training_metrics(episode_rewards, episode_losses)
    
    return episode_rewards, episode_losses

# Run training
if __name__ == "__main__":
    rewards, losses = train_actor_critic()
    
    
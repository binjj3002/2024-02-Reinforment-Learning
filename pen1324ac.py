import os
# Resolve OMP error by setting environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action

# Define the DDQN Agent
class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Copy weights from Q-network to target Q-network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 64
        self.replay_buffer = []
        self.buffer_size = 100000
        self.update_target_every = 10

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.uniform(-2.0, 2.0, size=(1,))  # Random action (Pendulum-v1 specific)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()  # Action with highest Q-value

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)  # Remove the oldest transition
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough transitions to train

        batch = self.sample_batch()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # Actions must be in shape (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q-values for the current states
        q_values = self.q_network(states).gather(1, actions)

        # Get the best action from the next state using the Q-network
        next_q_values = self.q_network(next_states).detach()
        next_actions = next_q_values.argmax(1).unsqueeze(1)

        # Compute the target using the target Q-network
        target_q_values = self.target_q_network(next_states).gather(1, next_actions)
        target_q_values = rewards + (self.gamma * target_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

# Main training function
def train_ddqn():
    env = gym.make('Pendulum-v1',render_mode = 'human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDQNAgent(state_dim, action_dim)

    num_episodes = 1000
    max_steps = 200
    all_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Store the transition in the replay buffer
            agent.store_transition(state, action, reward, next_state, done or truncated)

            # Train the agent
            agent.train()

            # Update the target network every few episodes
            if episode % agent.update_target_every == 0:
                agent.update_target_network()

            state = next_state

            if done or truncated:
                break

        all_rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

    env.close()

    plt.plot(all_rewards)
    plt.title("DDQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    return agent

def test_ddqn(agent, env, num_episodes=10):
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            env.render()

            if done or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()
    print(f"Average Reward over {num_episodes} test episodes: {np.mean(episode_rewards)}")
    return episode_rewards

if __name__ == "__main__":
    trained_agent = train_ddqn()

    print("\n--- Testing the Trained DDQN Agent ---")
    test_env = gym.make('Pendulum-v1', render_mode='human')
    test_rewards = test_ddqn(trained_agent, test_env, num_episodes=5)

    plt.plot(test_rewards)
    plt.title("DDQN Testing Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

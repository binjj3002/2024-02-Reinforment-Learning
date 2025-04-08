import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Định nghĩa mạng Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Tanh để đảm bảo hành động trong khoảng [-1, 1]

# Định nghĩa mạng Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Định nghĩa DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)

        # Mạng mục tiêu
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005  # Hệ số cập nhật mục tiêu
        self.exploration_noise = 0.1  # Thêm noise để khám phá

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += np.random.normal(0, self.exploration_noise, size=self.action_dim)  # Thêm noise
        return np.clip(action, -1, 1)  # Đảm bảo hành động trong khoảng [-1, 1]

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experience(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.sample_experience()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Cập nhật Critic
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions)
        target_q_values = rewards + self.gamma * next_q_values.squeeze(1) * (1 - dones)

        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values.squeeze(1), target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Cập nhật Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Cập nhật mục tiêu (target)
        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# Hàm huấn luyện DDPG
def train_ddpg():
    env = gym.make('Pendulum-v1')
    #env = gym.make('Pendulum-v1',render_mode = 'human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(state_dim, action_dim)

    num_episodes = 1000
    max_steps = 200
    all_rewards = []

    for episode in range(num_episodes):
        #state, _ = env.reset()
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            agent.store_experience(state, action, reward, next_state, done)
            agent.update()

            state = next_state

            if done or truncated:
                break

        all_rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

    env.close()

    # Hiển thị kết quả huấn luyện
    plt.plot(all_rewards)
    plt.title("DDPG Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    return agent

# Hàm kiểm tra DDPG
def test_ddpg(agent):
    env = gym.make('Pendulum-v1',render_mode = 'human')

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done or truncated:
            break

    env.close()
    print(f"Test Total Reward = {total_reward}")

if __name__ == "__main__":
    trained_agent = train_ddpg()
    test_ddpg(trained_agent)

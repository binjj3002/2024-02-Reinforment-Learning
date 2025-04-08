import os
# Resolve OMP error by setting environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Actor-Critic Neural Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        # Actor output
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic output
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.actor_mean(x)
        log_std = self.actor_log_std.expand_as(mean)
        value = self.critic(x)
        return mean, log_std, value

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.3
        self.entropy_coeff = 0.01
        self.epochs = 10
        self.batch_size = 64

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std, _ = self.model(state_tensor)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(axis=-1)
        return action.squeeze(0).numpy(), action_log_prob.item()

    def compute_advantages(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def train(self, trajectories):
        states = torch.FloatTensor(np.concatenate([t['states'] for t in trajectories]))
        actions = torch.FloatTensor(np.concatenate([t['actions'] for t in trajectories]))
        log_probs = torch.FloatTensor(np.concatenate([t['log_probs'] for t in trajectories]))
        returns = torch.FloatTensor(np.concatenate([t['returns'] for t in trajectories]))
        advantages = torch.FloatTensor(np.concatenate([t['advantages'] for t in trajectories]))

        for _ in range(self.epochs):
            indices = np.arange(states.size(0))
            np.random.shuffle(indices)

            for start in range(0, states.size(0), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                mean, log_std, values = self.model(batch_states)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)

                new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()

                ratio = torch.exp(new_log_probs - batch_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages

                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = (batch_returns - values).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# Main training function
def train_ppo():
    #env = gym.make('Pendulum-v1', render_mode='human', g=5.0)  # Reduced gravity
    env = gym.make('Pendulum-v1', g=5.0)  # Reduced gravity
    env.env.mass = 0.75  # Reduced mass

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)

    num_episodes = 3000
    max_steps = 200

    all_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()

        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        total_reward = 0

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, _, value = agent.model(state_tensor)

            # Custom reward function based on pole angle
            angle = np.arctan2(state[1], state[0])
            reward += -1 * (angle ** 2)  # Penalize large angles

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.item())
            dones.append(done or truncated)

            state = next_state

            if done or truncated:
                break

        # Compute advantages and returns
        next_value = 0 if done else agent.model(torch.FloatTensor(state).unsqueeze(0))[2].item()
        advantages = agent.compute_advantages(rewards, values, dones, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Store trajectory
        trajectory = {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'returns': returns,
            'advantages': advantages,
        }

        # Train agent
        agent.train([trajectory])

        all_rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

    env.close()

    plt.plot(all_rewards)
    plt.title("PPO Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    return agent

def test_ppo(agent, env, num_episodes=10):
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean, _, _ = agent.model(state_tensor)

            action = action_mean.squeeze(0).numpy()
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
    trained_agent = train_ppo()

    print("\n--- Testing the Trained PPO Agent ---")
    test_env = gym.make('Pendulum-v1', render_mode='human', g=5.0)  # Consistent reduced gravity
    test_env.env.mass = 0.75  # Consistent reduced mass
    test_rewards = test_ppo(trained_agent, test_env, num_episodes=5)

    plt.plot(test_rewards)
    plt.title("PPO Testing Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

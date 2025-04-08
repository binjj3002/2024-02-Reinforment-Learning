import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Environment creation
env = gym.make('CartPole-v1', render_mode="human")
# State discretization function
def discretize_state(state):
    # Define bins for each observation dimension
    cart_pos_bins = np.linspace(-2.4, 2.4, 10)
    cart_vel_bins = np.linspace(-3.0, 3.0, 10)
    pole_angle_bins = np.linspace(-0.2, 0.2, 10)
    pole_vel_bins = np.linspace(-2.5, 2.5, 10)
    
    # Convert each observation to discretized index (0 to 9)
    cart_pos_disc = np.clip(np.digitize(state[0], cart_pos_bins) - 1, 0, 9)
    cart_vel_disc = np.clip(np.digitize(state[1], cart_vel_bins) - 1, 0, 9)
    pole_angle_disc = np.clip(np.digitize(state[2], pole_angle_bins) - 1, 0, 9)
    pole_vel_disc = np.clip(np.digitize(state[3], pole_vel_bins) - 1, 0, 9)
    
    return (cart_pos_disc, cart_vel_disc, pole_angle_disc, pole_vel_disc)

# Hyperparameters
learning_rate = 0.1618
discount_factor = 0.9
epsilon = 0.6
epsilon_decay = 0.99
episodes = 5000

# Q-table initialization
q_table = np.zeros((10, 10, 10, 10, env.action_space.n))

# Tracking rewards for visualization
episode_rewards = []

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_episode_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Take action and observe results
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # Track total episode reward
        total_episode_reward += reward
        
        # Q-value update
        current_q = q_table[state + (action,)]
        next_max_q = np.max(q_table[next_state])
        new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
        q_table[state + (action,)] = new_q
        
        state = next_state
    
    # Store episode reward for analysis
    episode_rewards.append(total_episode_reward)
    
    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, 0.01)
    
    # Print progress
    if episode % 500 == 0:
        print(f"Episode {episode} completed, Reward: {total_episode_reward}")

env.close()

# Visualization of rewards
plt.figure(figsize=(12, 6))

# Plot 1: Raw rewards
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Plot 2: Moving average of rewards
plt.subplot(1, 2, 2)
window_size = 100
rewards_moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(rewards_moving_avg)
plt.title(f'Moving Average of Rewards (Window={window_size})')
plt.xlabel('Episode')
plt.ylabel('Average Reward')

plt.tight_layout()
plt.show()

# Performance statistics
print("\nReward Analysis:")
print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
print(f"Max Reward: {np.max(episode_rewards)}")
print(f"Min Reward: {np.min(episode_rewards)}")
print(f"Reward Standard Deviation: {np.std(episode_rewards):.2f}")

# Testing phase with rendering
env = gym.make('CartPole-v1', render_mode="human")
state, _ = env.reset()
state = discretize_state(state)
done = False
total_reward = 0

print("\nTesting trained agent...")
while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    total_reward += reward
    state = next_state
    
    # Additional termination condition
    if abs(state[2]) > 0.15:  # pole angle check
        done = True
        print("The pole has fallen!")
    
    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    env.render()

print(f"Total test reward: {total_reward}")
env.close()
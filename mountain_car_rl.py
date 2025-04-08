import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('MountainCar-v0', render_mode="human")

# Discretize the state space
def discretize_state(state):
    pos_bins = np.linspace(-1.2, 0.6, 20)
    vel_bins = np.linspace(-0.07, 0.07, 20)
    pos_disc = np.digitize(state[0], pos_bins)
    vel_disc = np.digitize(state[1], vel_bins)
    return (pos_disc, vel_disc)

# Q-learning parameters
learning_rate = 0.314
discount_factor = 0.50
epsilon = 0.25
episodes = 100

# Initialize Q-table
q_table = np.zeros((20, 20, 3))

# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    state = discretize_state(state)
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Take action and observe result
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # Update Q-value
        current_q = q_table[state + (action,)]
        next_max_q = np.max(q_table[next_state])
        new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
        q_table[state + (action,)] = new_q
        
        state = next_state
        
        # Render the environment (comment out for faster training)
        env.render()
    
    if episode % 100 == 0:
        print(f"Episode {episode} completed")

# Close the environment
env.close()

# Test the trained agent
env = gym.make('MountainCar-v0', render_mode="human")
state = env.reset()[0]
state = discretize_state(state)
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    total_reward += reward
    state = next_state
    env.render()

print(f"Total reward: {total_reward}")
env.close()
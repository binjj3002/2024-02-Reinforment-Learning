import gymnasium as gym
import numpy as np

# Create the environment (no rendering during training)
env = gym.make('MountainCar-v0')

# Discretize the state space
def discretize_state(state):
    pos_bins = np.linspace(-1.2, 0.6, 20)
    vel_bins = np.linspace(-0.07, 0.07, 20)
    pos_disc = np.digitize(state[0], pos_bins)
    vel_disc = np.digitize(state[1], vel_bins)
    return (pos_disc, vel_disc)

# Q-learning parameters
learning_rate = 0.4
discount_factor = 0.9
epsilon = 0.25
epsilon_decay = 0.99
episodes = 1000

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
    
    # Decay epsilon after each episode
    epsilon = max(epsilon * epsilon_decay, 0.01)

    # Optional print for progress monitoring
    if episode % 100 == 0:
        print(f"Episode {episode} completed")

# Close environment after training
env.close()

# Testing the trained agent with rendering
env = gym.make('MountainCar-v0', render_mode="human")
state = env.reset()[0]
state = discretize_state(state)
done = False
total_reward = 0

print("Testing trained agent...")

while not done:
    action = np.argmax(q_table[state])  # Choose the best action from the Q-table
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    total_reward += reward
    state = next_state
    env.render()  # Render each step for visualization

print(f"Total reward: {total_reward}")
env.close()  # Close environment after testing

import gymnasium as gym
import numpy as np
import pickle  # Thư viện để tải bảng Q từ pickle
import pygame  # Thư viện để theo dõi sự kiện phím và cửa sổ

# Function to discretize the state for indexing in the Q-table
def discretize_state(state):
    cart_pos_bins = np.linspace(-2.4, 2.4, 10)
    cart_vel_bins = np.linspace(-3.0, 3.0, 10)
    pole_angle_bins = np.linspace(-0.2, 0.2, 10)
    pole_vel_bins = np.linspace(-2.5, 2.5, 10)
    
    cart_pos_disc = np.clip(np.digitize(state[0], cart_pos_bins) - 1, 0, 9)
    cart_vel_disc = np.clip(np.digitize(state[1], cart_vel_bins) - 1, 0, 9)
    pole_angle_disc = np.clip(np.digitize(state[2], pole_angle_bins) - 1, 0, 9)
    pole_vel_disc = np.clip(np.digitize(state[3], pole_vel_bins) - 1, 0, 9)
    
    return (cart_pos_disc, cart_vel_disc, pole_angle_disc, pole_vel_disc)

# Load the trained Q-table from pickle file
with open("Cart_Pole_q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Create the environment in render mode
env = gym.make('CartPole-v1', render_mode="human")

# Initialize pygame for event handling
pygame.init()

# Reset the environment and initialize variables
state = env.reset()[0]
state = discretize_state(state)
done = False
total_reward = 0

# Set up a flag for quitting
quit_game = False

print("Rendering test run until pole falls or window is closed...")

# Test loop - Render until done or quit
while not done and not quit_game:
    # Poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Check if the window was closed
            quit_game = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # Check if 'q' is pressed
                quit_game = True

    # Choose the action based on the trained Q-table
    action = np.argmax(q_table[state])  
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    total_reward += reward
    state = next_state
    
    # Render each step
    env.render()

# Print total reward indicating duration of balance
print(f"Total reward (balance duration): {total_reward}")
env.close()
pygame.quit()  # Quit pygame when done

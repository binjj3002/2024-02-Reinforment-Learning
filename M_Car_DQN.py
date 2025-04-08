import gymnasium as gym
import numpy as np
import pygame
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Tạo môi trường Mountain Car
env = gym.make('MountainCar-v0', render_mode="human")
env.unwrapped.frame_skip = 4

# Định nghĩa mạng neural network cho DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hàm reward tùy chỉnh
def custom_reward(state, action, next_state, done):
    position = next_state[0]
    velocity = next_state[1]
    
    # Khởi tạo reward
    reward = 0
    
    # Khuyến khích di chuyển về phía bên phải và tăng tốc
    reward += position + abs(velocity)
    
    # Khuyến khích tăng tốc về phía bên trái khi ở nửa bên trái của môi trường
    if position < -0.5 and velocity < 0:
        reward += abs(velocity)
    
    # Khuyến khích tăng tốc về phía bên phải khi ở nửa bên phải của môi trường
    if position > -0.5 and velocity > 0:
        reward += velocity
    
    # Thưởng lớn nếu đạt đến đích
    if position >= 0.5:
        reward += 100
    
    # Phạt nhẹ cho mỗi bước để khuyến khích hoàn thành nhanh
    reward -= 1
    
    return reward

# Các tham số cho DQN
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 0.01618

# Khởi tạo DQN và memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

# Hàm chọn action
def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(torch.FloatTensor(state).to(device)).max(0)[1].view(1, 1).item()
    else:
        return env.action_space.sample()

# Hàm huấn luyện
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.FloatTensor(batch[0]).to(device)
    action_batch = torch.LongTensor(batch[1]).view(-1, 1).to(device)
    reward_batch = torch.FloatTensor(batch[2]).to(device)
    next_state_batch = torch.FloatTensor(batch[3]).to(device)
    done_batch = torch.FloatTensor(batch[4]).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

    loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Vòng lặp huấn luyện
episodes = 1000
epsilon = EPS_START

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                sys.exit()

        action = select_action(state, epsilon)
        next_state, _, done, _, _ = env.step(action)
        reward = custom_reward(state, action, next_state, done)
        episode_reward += reward
        steps += 1

        memory.append((state, action, reward, next_state, done))
        state = next_state

        optimize_model()

        if steps % 10 == 0:
            env.render()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}:")
    print(f"  Reward: {episode_reward:.2f}")
    print(f"  Steps: {steps}")
    print(f"  Epsilon: {epsilon:.4f}")
    print("--------------------")

    if episode % 100 == 0:
        print(f"Đã hoàn thành {episode} episodes")

env.close()

# Kiểm tra agent đã được huấn luyện
env = gym.make('MountainCar-v0', render_mode="human")
env.unwrapped.frame_skip = 2
state = env.reset()[0]
done = False
test_reward = 0
test_steps = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            env.close()
            sys.exit()

    with torch.no_grad():
        action = policy_net(torch.FloatTensor(state).to(device)).max(0)[1].view(1, 1).item()
    next_state, _, done, _, _ = env.step(action)
    reward = custom_reward(state, action, next_state, done)
    test_reward += reward
    test_steps += 1
    state = next_state

    if test_steps % 5 == 0:
        env.render()

    if next_state[0] >= 0.5:
        print("Xe đã đạt đến đích trong quá trình kiểm tra!")
        env.render()
        time.sleep(1)
        break

print(f"Kết quả kiểm tra cuối cùng:")
print(f"  Phần thưởng: {test_reward:.2f}")
print(f"  Số bước: {test_steps}")

env.close()
pygame.quit()
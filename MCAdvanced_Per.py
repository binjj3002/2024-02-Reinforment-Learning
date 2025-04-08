import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import time
import pickle
import os

class CustomMountainCarEnv(gym.Env):
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 2.4  # Mở rộng khoảng vị trí để chứa cả hai đỉnh
        self.max_speed = 0.07
        self.goal_position = 2.4  # Đặt cờ đích tại đỉnh thứ hai
        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action} invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force - np.cos(3 * position) * self.gravity
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if position == self.min_position and velocity < 0:
            velocity = 0

        done = position >= self.goal_position
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 400))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((600, 400))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]
        self._draw_hills()
        
        # Draw car
        pos_px = (pos - self.min_position) / (self.max_position - self.min_position) * 600
        car_y = int(400 - self._height(pos) * 40)  # Scale height by 40 to fit in screen
        car_image = pygame.Surface((20, 10))
        car_image.fill((255, 0, 0))
        self.surf.blit(car_image, (int(pos_px) - 10, car_y - 10))

        # Draw flag
        flag_x = (self.goal_position - self.min_position) / (self.max_position - self.min_position) * 600
        flag_y = int(400 - self._height(self.goal_position) * 40)
        pygame.draw.polygon(self.surf, (0, 255, 0), [(flag_x, flag_y), (flag_x, flag_y - 30), (flag_x + 20, flag_y - 15)])

        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(30)

    def _height(self, xs):
        # Tạo hai đỉnh núi liên tiếp
        if xs < 0.5:
            return max(0, 5 * (1 - np.cos(np.pi * xs)))  # Đỉnh thứ nhất cao 5m
        else:
            return 5 + 5 * (1 - np.cos(np.pi * (xs - 0.5)))  # Đỉnh thứ hai cao 10m

    def _draw_hills(self):
        xs = np.linspace(self.min_position, self.max_position, 600)
        ys = self._height(xs)
        scaled_ys = 400 - ys * 40  # Scale height by 40 to fit in screen
        points = list(zip(range(600), scaled_ys.astype(int)))
        pygame.draw.lines(self.surf, (0, 0, 0), False, points)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

# Khởi tạo Pygame
pygame.init()

# Tạo môi trường Mountain Car với tốc độ cao hơn
env = CustomMountainCarEnv()
env.unwrapped.frame_skip = 4  # Bỏ qua 4 frame để tăng tốc

# Hàm rời rạc hóa không gian trạng thái
def discretize_state(state):
    pos_bins = np.linspace(-1.2, 1.8, 30)
    vel_bins = np.linspace(-0.07, 0.07, 20)
    pos_disc = np.digitize(state[0], pos_bins)
    vel_disc = np.digitize(state[1], vel_bins)
    return (pos_disc, vel_disc)

# Hàm reward tùy chỉnh
def custom_reward(state, action, next_state, done):
    position = next_state[0]
    velocity = next_state[1]
    
    reward = 0
    reward += position + abs(velocity)
    
    if position < 0 and velocity < 0:
        reward += abs(velocity)
    
    if position > 0 and velocity > 0:
        reward += velocity
    
    if position <= -1.2 and velocity == 0:
        reward -= 20
    
    if position >= 1.5:
        reward += 100
    
    reward -= 1
    
    return reward

# Các tham số cho thuật toán Dyna-Q
learning_rate = 0.16188
discount_factor = 0.99
epsilon = 0.12
episodes = 1000
planning_steps = 5

# Khởi tạo Q-table và model
q_table = np.zeros((30, 20, 3))
model = {}

# Biến theo dõi
total_cumulative_reward = 0
episode_durations = []
successful_episodes = 0
epsilon_decay = 0.99

# Hàm lưu trạng thái
def save_state():
    state = {
        'q_table': q_table,
        'model': model,
        'total_cumulative_reward': total_cumulative_reward,
        'episode_durations': episode_durations,
        'successful_episodes': successful_episodes,
        'epsilon': epsilon,
        'episode': episode
    }
    with open('training_state.pickle', 'wb') as f:
        pickle.dump(state, f)
    print("Đã lưu trạng thái training")

# Hàm tải trạng thái
def load_state():
    global q_table, model, total_cumulative_reward, episode_durations, successful_episodes, epsilon, episode
    if os.path.exists('training_state.pickle'):
        with open('training_state.pickle', 'rb') as f:
            state = pickle.load(f)
        q_table = state['q_table']
        model = state['model']
        total_cumulative_reward = state['total_cumulative_reward']
        episode_durations = state['episode_durations']
        successful_episodes = state['successful_episodes']
        epsilon = state['epsilon']
        episode = state['episode']
        print(f"Đã tải trạng thái training, tiếp tục từ episode {episode}")
        return True
    return False

# Tải trạng thái nếu có
if load_state():
    start_episode = episode + 1
else:
    start_episode = 0

# Vòng lặp huấn luyện Dyna-Q
for episode in range(start_episode, episodes):
    state, _ = env.reset()
    state_disc = discretize_state(state)
    done = False
    episode_reward = 0
    steps = 0
    start_time = time.time()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_state()
                env.close()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_state()
                    print("Trạng thái đã được lưu. Bạn có thể thoát an toàn.")

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_disc])

        next_state, reward, done, _, _ = env.step(action)
        next_state_disc = discretize_state(next_state)
        reward = custom_reward(state, action, next_state, done)
        episode_reward += reward
        steps += 1
        
        # Cập nhật Q-table
        current_q = q_table[state_disc + (action,)]
        next_max_q = np.max(q_table[next_state_disc])
        new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
        q_table[state_disc + (action,)] = new_q
        
        # Cập nhật model
        if state_disc not in model:
            model[state_disc] = {}
        model[state_disc][action] = (next_state_disc, reward)
        
        # Lập kế hoạch (planning)
        for _ in range(planning_steps):
            plan_state = tuple(np.random.randint(30, size=2))
            plan_action = np.random.randint(3)
            
            if plan_state in model and plan_action in model[plan_state]:
                plan_next_state, plan_reward = model[plan_state][plan_action]
                plan_current_q = q_table[plan_state + (plan_action,)]
                plan_next_max_q = np.max(q_table[plan_next_state])
                plan_new_q = plan_current_q + learning_rate * (plan_reward + discount_factor * plan_next_max_q - plan_current_q)
                q_table[plan_state + (plan_action,)] = plan_new_q
        
        state = next_state
        state_disc = next_state_disc
        
        # Chỉ render mỗi 10 bước để tăng tốc
        if steps % 10 == 0:
            env.render()

        # Kiểm tra nếu xe đạt đến đích
        if next_state[0] >= 1.5:
            print("Xe đã đạt đến đích!")
            env.render()
            time.sleep(1)
            done = True

    episode_duration = time.time() - start_time
    total_cumulative_reward += episode_reward
    episode_durations.append(episode_duration)
    if episode_reward > -200:
        successful_episodes += 1

    epsilon *= epsilon_decay

    # In thông tin chi tiết sau mỗi episode
    print(f"Episode {episode}:")
    print(f"  Reward: {episode_reward:.2f}")
    print(f"  Total Cumulative Reward: {total_cumulative_reward:.2f}")
    print(f"  Steps: {steps}")
    print(f"  Duration: {episode_duration:.2f} seconds")
    print(f"  Epsilon: {epsilon:.4f}")
    print(f"  Success Rate: {successful_episodes / (episode + 1):.2%}")
    print(f"  Average Duration: {np.mean(episode_durations):.2f} seconds")
    print(f"  Q-table max value: {np.max(q_table):.2f}")
    print(f"  Q-table min value: {np.min(q_table):.2f}")
    print(f"  Model size: {sum(len(actions) for actions in model.values())}")
    print("--------------------")

    if episode % 100 == 0:
        print(f"Đã hoàn thành {episode} episodes")
        save_state()

env.close()

# Kiểm tra agent đã được huấn luyện
env = CustomMountainCarEnv()
state, _ = env.reset()
state_disc = discretize_state(state)
done = False
test_reward = 0
test_steps = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            sys.exit()

    action = np.argmax(q_table[state_disc])
    next_state, reward, done, _, _ = env.step(action)
    next_state_disc = discretize_state(next_state)
    reward = custom_reward(state, action, next_state, done)
    test_reward += reward
    test_steps += 1
    state = next_state
    state_disc = next_state_disc
    
    env.render()

    if next_state[0] >= 1.5:
        print("Xe đã đạt đến đích trong quá trình kiểm tra!")
        env.render()
        time.sleep(1)
        break

print(f"Kết quả kiểm tra cuối cùng:")
print(f"  Phần thưởng: {test_reward:.2f}")
print(f"  Số bước: {test_steps}")

env.close()
pygame.quit()

# Tạo môi trường Mountain Car với tốc độ cao hơn
env = CustomMountainCarEnv()

# Hàm rời rạc hóa không gian trạng thái
def discretize_state(state):
    pos_bins = np.linspace(-1.2, 1.8, 30)
    vel_bins = np.linspace(-0.07, 0.07, 20)
    pos_disc = np.digitize(state[0], pos_bins)
    vel_disc = np.digitize(state[1], vel_bins)
    return (pos_disc, vel_disc)

# Hàm reward tùy chỉnh
def custom_reward(state, action, next_state, done):
    position = next_state[0]
    velocity = next_state[1]
    
    reward = 0
    reward += position + abs(velocity)
    
    if position < 0 and velocity < 0:
        reward += abs(velocity)
    
    if position > 0 and velocity > 0:
        reward += velocity
    
    if position <= -1.2 and velocity == 0:
        reward -= 20
    
    if position >= 1.5:
        reward += 100
    
    reward -= 1
    
    return reward

# Các tham số cho thuật toán Dyna-Q
learning_rate = 0.16188
discount_factor = 0.99
epsilon = 0.12
episodes = 1000
planning_steps = 5

# Khởi tạo Q-table và model
q_table = np.zeros((30, 20, 3))
model = {}

# Biến theo dõi
total_cumulative_reward = 0
episode_durations = []
successful_episodes = 0
epsilon_decay = 0.99

# Hàm lưu trạng thái
def save_state():
    state = {
        'q_table': q_table,
        'model': model,
        'total_cumulative_reward': total_cumulative_reward,
        'episode_durations': episode_durations,
        'successful_episodes': successful_episodes,
        'epsilon': epsilon,
        'episode': episode
    }
    with open('training_state.pickle', 'wb') as f:
        pickle.dump(state, f)
    print("Đã lưu trạng thái training")

# Hàm tải trạng thái
def load_state():
    global q_table, model, total_cumulative_reward, episode_durations, successful_episodes, epsilon, episode
    if os.path.exists('training_state.pickle'):
        with open('training_state.pickle', 'rb') as f:
            state = pickle.load(f)
        q_table = state['q_table']
        model = state['model']
        total_cumulative_reward = state['total_cumulative_reward']
        episode_durations = state['episode_durations']
        successful_episodes = state['successful_episodes']
        epsilon = state['epsilon']
        episode = state['episode']
        print(f"Đã tải trạng thái training, tiếp tục từ episode {episode}")
        return True
    return False

# Tải trạng thái nếu có
if load_state():
    start_episode = episode + 1
else:
    start_episode = 0

# Vòng lặp huấn luyện Dyna-Q
for episode in range(start_episode, episodes):
    state, _ = env.reset()
    state_disc = discretize_state(state)
    done = False
    episode_reward = 0
    steps = 0
    start_time = time.time()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_state()
                env.close()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_state()
                    print("Trạng thái đã được lưu. Bạn có thể thoát an toàn.")

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_disc])

        next_state, reward, done, _, _ = env.step(action)
        next_state_disc = discretize_state(next_state)
        reward = custom_reward(state, action, next_state, done)
        episode_reward += reward
        steps += 1
        
        # Cập nhật Q-table
        current_q = q_table[state_disc + (action,)]
        next_max_q = np.max(q_table[next_state_disc])
        new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
        q_table[state_disc + (action,)] = new_q
        
        # Cập nhật model
        if state_disc not in model:
            model[state_disc] = {}
        model[state_disc][action] = (next_state_disc, reward)
        
        # Lập kế hoạch (planning)
        for _ in range(planning_steps):
            plan_state = tuple(np.random.randint(30, size=2))
            plan_action = np.random.randint(3)
            
            if plan_state in model and plan_action in model[plan_state]:
                plan_next_state, plan_reward = model[plan_state][plan_action]
                plan_current_q = q_table[plan_state + (plan_action,)]
                plan_next_max_q = np.max(q_table[plan_next_state])
                plan_new_q = plan_current_q + learning_rate * (plan_reward + discount_factor * plan_next_max_q - plan_current_q)
                q_table[plan_state + (plan_action,)] = plan_new_q
        
        state = next_state
        state_disc = next_state_disc
        
        env.render()  # Render mỗi bước để thấy rõ chuyển động

        if next_state[0] >= 1.5:
            print("Xe đã đạt đến đích!")
            env.render()
            time.sleep(1)
            done = True

    episode_duration = time.time() - start_time
    total_cumulative_reward += episode_reward
    episode_durations.append(episode_duration)
    if episode_reward > -200:
        successful_episodes += 1

    epsilon *= epsilon_decay

    # In thông tin chi tiết sau mỗi episode
    print(f"Episode {episode}:")
    print(f"  Reward: {episode_reward:.2f}")
    print(f"  Total Cumulative Reward: {total_cumulative_reward:.2f}")
    print(f"  Steps: {steps}")
    print(f"  Duration: {episode_duration:.2f} seconds")
    print(f"  Epsilon: {epsilon:.4f}")
    print(f"  Success Rate: {successful_episodes / (episode + 1):.2%}")
    print(f"  Average Duration: {np.mean(episode_durations):.2f} seconds")
    print(f"  Q-table max value: {np.max(q_table):.2f}")
    print(f"  Q-table min value: {np.min(q_table):.2f}")
    print(f"  Model size: {sum(len(actions) for actions in model.values())}")
    print("--------------------")

    if episode % 100 == 0:
        print(f"Đã hoàn thành {episode} episodes")
        save_state()

env.close()

# Kiểm tra agent đã được huấn luyện
env = CustomMountainCarEnv()
state, _ = env.reset()
state_disc = discretize_state(state)
done = False
test_reward = 0
test_steps = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            sys.exit()

    action = np.argmax(q_table[state_disc])
    next_state, reward, done, _, _ = env.step(action)
    next_state_disc = discretize_state(next_state)
    reward = custom_reward(state, action, next_state, done)
    test_reward += reward
    test_steps += 1
    state = next_state
    state_disc = next_state_disc
    
    env.render()

    if next_state[0] >= 1.5:
        print("Xe đã đạt đến đích trong quá trình kiểm tra!")
        env.render()
        time.sleep(1)
        break

print(f"Kết quả kiểm tra cuối cùng:")
print(f"  Phần thưởng: {test_reward:.2f}")
print(f"  Số bước: {test_steps}")

env.close()
pygame.quit()
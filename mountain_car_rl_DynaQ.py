import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import time
import pickle
import os
from numba import jit, float64

pygame.init()

class CustomMountainCarEnv(gym.Env):
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 2.4
        self.max_speed = 0.14
        self.goal_position = 2.0
        self.force = 0.002
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float64)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float64)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)

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

        self.state = np.array([position, velocity], dtype=np.float64)
        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0], dtype=np.float64)
        return self.state, {}

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((800, 400))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((800, 400))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]
        self._draw_hills()
        
        # Draw car
        pos_px = int((pos - self.min_position) / (self.max_position - self.min_position) * 800)
        car_y = int(350 - self._height(pos) * 200)
        car_image = pygame.Surface((20, 10))
        car_image.fill((255, 0, 0))
        self.surf.blit(car_image, (pos_px - 10, car_y - 10))

        # Draw flag
        flag_x = int((self.goal_position - self.min_position) / (self.max_position - self.min_position) * 800)
        flag_y = int(350 - self._height(self.goal_position) * 200)
        pygame.draw.polygon(self.surf, (0, 255, 0), [(flag_x, flag_y), (flag_x, flag_y - 30), (flag_x + 20, flag_y - 15)])

        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)

    @staticmethod
    @jit(nopython=True)
    def _height(x):
        if x < 0.5:
            return 0.45 * (1 - np.cos(np.pi * x * 1.5))
        else:
            return 0.45 + 0.55 * (1 - np.cos(np.pi * (x - 0.5) * 1.2))

    def _draw_hills(self):
        xs = np.linspace(self.min_position, self.max_position, 800)
        ys = np.array([self._height(x) for x in xs])
        scaled_ys = 350 - ys * 200
        points = list(zip(range(800), scaled_ys.astype(int)))
        pygame.draw.lines(self.surf, (0, 0, 0), False, points)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
        if self.clock is not None:
            self.clock = None

@jit(nopython=True)
def discretize_state(state, pos_bins, vel_bins):
    pos_disc = np.searchsorted(pos_bins, state[0], side='right')
    vel_disc = np.searchsorted(vel_bins, state[1], side='right')
    return pos_disc, vel_disc

@jit(nopython=True)
def custom_reward(state, action, next_state, done):
    position = next_state[0]
    velocity = next_state[1]
    
    reward = 0.0
    reward += position + abs(velocity)
    
    if position < 0 and velocity < 0:
        reward += abs(velocity)
    
    if position > 0 and velocity > 0:
        reward += velocity
    
    if position <= -1.2 and velocity == 0:
        reward -= 20.0
    
    if position >= 2.0:
        reward += 100.0
    
    reward -= 1.0
    
    return reward

# Khởi tạo môi trường và các tham số
env = CustomMountainCarEnv()
pos_bins = np.linspace(-1.2, 2.4, 30, dtype=np.float64)
vel_bins = np.linspace(-0.14, 0.14, 20, dtype=np.float64)

learning_rate = 0.16188
discount_factor = 0.99
epsilon = 0.12
episodes = 1000
planning_steps = 5

q_table = np.zeros((30, 20, 3), dtype=np.float64)
model = {}

total_cumulative_reward = 0.0
episode_durations = []
successful_episodes = 0
epsilon_decay = 0.99

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

if load_state():
    start_episode = episode + 1
else:
    start_episode = 0

running = True
for episode in range(start_episode, episodes):
    if not running:
        break
    state, _ = env.reset()
    state_disc = discretize_state(state, pos_bins, vel_bins)
    done = False
    episode_reward = 0.0
    steps = 0
    start_time = time.time()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_state()
                running = False
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_state()
                    print("Trạng thái đã được lưu. Bạn có thể thoát an toàn.")
        
        if not running:
            break

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_disc])

        next_state, reward, done, _, _ = env.step(action)
        next_state_disc = discretize_state(next_state, pos_bins, vel_bins)
        reward = custom_reward(state, action, next_state, done)
        episode_reward += reward
        steps += 1
        
        current_q = q_table[state_disc + (action,)]
        next_max_q = np.max(q_table[next_state_disc])
        new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
        q_table[state_disc + (action,)] = new_q
        
        if state_disc not in model:
            model[state_disc] = {}
        model[state_disc][action] = (next_state_disc, reward)
        
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
        
        if steps % 5 == 0:
            env.render()

        if next_state[0] >= 2.0:
            print("Xe đã đạt đến đích!")
            env.render()
            time.sleep(0.5)
            done = True

    if not running:
        break

    episode_duration = time.time() - start_time
    total_cumulative_reward += episode_reward
    episode_durations.append(episode_duration)
    if episode_reward > -200:
        successful_episodes += 1

    epsilon *= epsilon_decay

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
if running:
    env = CustomMountainCarEnv()
    state, _ = env.reset()
    state_disc = discretize_state(state, pos_bins, vel_bins)
    done = False
    test_reward = 0.0
    test_steps = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        if done:
            break

        action = np.argmax(q_table[state_disc])
        next_state, reward, done, _, _ = env.step(action)
        next_state_disc = discretize_state(next_state, pos_bins, vel_bins)
        reward = custom_reward(state, action, next_state, done)
        test_reward += reward
        test_steps += 1
        state = next_state
        state_disc = next_state_disc
        
        env.render()

        if next_state[0] >= 2.0:
            print("Xe đã đạt đến đích trong quá trình kiểm tra!")
            env.render()
            time.sleep(0.5)
            break

    print(f"Kết quả kiểm tra cuối cùng:")
    print(f"  Phần thưởng: {test_reward:.2f}")
    print(f"  Số bước: {test_steps}")

    env.close()

pygame.quit()
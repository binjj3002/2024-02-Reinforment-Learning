import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import time
import pickle

# Khởi tạo Pygame
pygame.init()

class ExtendedMountainCarEnv(gym.Env):
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 1.8
        self.max_speed = 0.07
        self.goal_position = 1.5
        self.goal_velocity = 0

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = self.calculate_reward(position, velocity, action)

        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def calculate_reward(self, position, velocity, action):
        reward = 0
        reward += position + abs(velocity)
        
        if position < 0 and velocity < 0:
            reward += abs(velocity)
        
        if position > 0 and velocity > 0:
            reward += velocity
        
        if position <= self.min_position and velocity == 0:
            reward -= 20
        
        if position >= self.goal_position:
            reward += 100
        
        reward -= 1
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((600, 400))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = 600 / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((600, 400))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        # Vẽ đường dốc mới
        def new_height(x):
            if x < 0:
                return np.sin(3 * x) * 0.3 + 0.55
            else:
                return np.sin(3 * x) * 0.4 + 0.8

        pygame.draw.lines(
            self.surf,
            (0, 0, 0),
            False,
            [
                (scale * (i / 100.0 + self.min_position), 400 - scale * new_height(i / 100.0 - 0.5) * 0.8)
                for i in range(600)
            ],
            2,
        )

        # Vẽ cờ đích
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(400 - scale * new_height(self.goal_position - 0.5) * 0.8)
        flagy2 = flagy1 + 50
        pygame.draw.line(self.surf, (0, 0, 0), (flagx, flagy1), (flagx, flagy2), 2)
        pygame.draw.polygon(self.surf, (255, 0, 0), [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])

        # Vẽ xe
        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(np.arctan(np.cos(3 * pos) * 0.3))
            coords.append(
                (
                    int(c[0] + scale * (pos - self.min_position)),
                    int(c[1] + clearance + scale * new_height(pos - 0.5) * 0.8),
                )
            )

        # Vẽ thân xe
        pygame.draw.polygon(self.surf, (0, 0, 255), coords)
        
        # Vẽ bánh xe
        wheel_radius = 5
        left_wheel = (int(coords[0][0]), int(coords[0][1]))
        right_wheel = (int(coords[3][0]), int(coords[3][1]))
        pygame.draw.circle(self.surf, (0, 0, 0), left_wheel, wheel_radius)
        pygame.draw.circle(self.surf, (0, 0, 0), right_wheel, wheel_radius)

        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(30)

    def height(self, pos):
        return np.sin(3 * pos) * 0.45 + 0.55

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

# Phần còn lại của mã giữ nguyên
# ...

# Tạo môi trường mới
env = ExtendedMountainCarEnv()

# Hàm rời rạc hóa không gian trạng thái
def discretize_state(state):
    pos_bins = np.linspace(env.min_position, env.max_position, 40)
    vel_bins = np.linspace(-env.max_speed, env.max_speed, 40)
    pos_disc = np.digitize(state[0], pos_bins)
    vel_disc = np.digitize(state[1], vel_bins)
    return (pos_disc, vel_disc)

# Các tham số cho thuật toán Dyna-Q
learning_rate = 0.16188
discount_factor = 0.99
epsilon = 0.12
episodes = 1000
planning_steps = 5

# Khởi tạo Q-table và model
q_table = np.zeros((40, 40, 3))
model = {}

# Biến theo dõi
total_cumulative_reward = 0
episode_durations = []
successful_episodes = 0
epsilon_decay = 0.99

# Hàm lưu trạng thái
def save_state(filename):
    state = {
        'q_table': q_table,
        'model': model,
        'episode': episode,
        'total_cumulative_reward': total_cumulative_reward,
        'episode_durations': episode_durations,
        'successful_episodes': successful_episodes,
        'epsilon': epsilon
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"Đã lưu trạng thái vào {filename}")

# Hàm tải trạng thái
def load_state(filename):
    global q_table, model, episode, total_cumulative_reward, episode_durations, successful_episodes, epsilon
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    q_table = state['q_table']
    model = state['model']
    episode = state['episode']
    total_cumulative_reward = state['total_cumulative_reward']
    episode_durations = state['episode_durations']
    successful_episodes = state['successful_episodes']
    epsilon = state['epsilon']
    print(f"Đã tải trạng thái từ {filename}")

# Vòng lặp huấn luyện Dyna-Q
for episode in range(episodes):
    state = env.reset()[0]
    state_disc = discretize_state(state)
    done = False
    episode_reward = 0
    steps = 0
    start_time = time.time()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Nhấn 'S' để lưu
                    save_state('mountain_car_state.pkl')
                elif event.key == pygame.K_l:  # Nhấn 'L' để tải
                    load_state('mountain_car_state.pkl')

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_disc])

        next_state, reward, done, _, _ = env.step(action)
        next_state_disc = discretize_state(next_state)
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
            plan_state = tuple(np.random.randint(40, size=2))
            plan_action = np.random.randint(3)
            
            if plan_state in model and plan_action in model[plan_state]:
                plan_next_state, plan_reward = model[plan_state][plan_action]
                plan_current_q = q_table[plan_state + (plan_action,)]
                plan_next_max_q = np.max(q_table[plan_next_state])
                plan_new_q = plan_current_q + learning_rate * (plan_reward + discount_factor * plan_next_max_q - plan_current_q)
                q_table[plan_state + (plan_action,)] = plan_new_q
        
        state = next_state
        state_disc = next_state_disc
        
        # Render mỗi 10 bước để tăng tốc
        if steps % 10 == 0:
            env.render()

        # Kiểm tra nếu xe đạt đến đích
        if next_state[0] >= env.goal_position:
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

env.close()

# Kiểm tra agent đã được huấn luyện
env = ExtendedMountainCarEnv()
state = env.reset()[0]
state_disc = discretize_state(state)
done = False
test_reward = 0
test_steps = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            env.close()
            sys.exit()

    action = np.argmax(q_table[state_disc])
    next_state, reward, done, _, _ = env.step(action)
    next_state_disc = discretize_state(next_state)
    test_reward += reward
    test_steps += 1
    state = next_state
    state_disc = next_state_disc
    
    # Render mỗi 5 bước trong quá trình kiểm tra
    if test_steps % 5 == 0:
        env.render()

    if next_state[0] >= env.goal_position:
        print("Xe đã đạt đến đích trong quá trình kiểm tra!")
        env.render()
        time.sleep(1)
        break

print(f"Kết quả kiểm tra cuối cùng:")
print(f"  Phần thưởng: {test_reward:.2f}")
print(f"  Số bước: {test_steps}")

env.close()
pygame.quit()
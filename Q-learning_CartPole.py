import gymnasium as gym
import numpy as np

# 환경 생성 (훈련 중에는 렌더링하지 않음)
env = gym.make('CartPole-v1')

# Q-learning을 위한 상태 공간 이산화
def discretize_state(state):
    # 각 관찰 차원에 대해 이산화할 구간 정의
    cart_pos_bins = np.linspace(-2.4, 2.4, 10)  # 카트 위치의 구간
    cart_vel_bins = np.linspace(-3.0, 3.0, 10)  # 카트 속도의 구간
    pole_angle_bins = np.linspace(-0.2, 0.2, 10)  # 기둥의 각도 구간
    pole_vel_bins = np.linspace(-2.5, 2.5, 10)  # 기둥의 속도 구간
    
    # 각 관찰값을 이산화된 값으로 변환 (0부터 9까지의 인덱스로)
    cart_pos_disc = np.clip(np.digitize(state[0], cart_pos_bins) - 1, 0, 9)
    cart_vel_disc = np.clip(np.digitize(state[1], cart_vel_bins) - 1, 0, 9)
    pole_angle_disc = np.clip(np.digitize(state[2], pole_angle_bins) - 1, 0, 9)
    pole_vel_disc = np.clip(np.digitize(state[3], pole_vel_bins) - 1, 0, 9)
    
    return (cart_pos_disc, cart_vel_disc, pole_angle_disc, pole_vel_disc)


# Q-learning 파라미터 설정
learning_rate = 0.1618  # 학습률
discount_factor = 0.9  # 할인율
epsilon = 0.6  # 탐사율 (epsilon-greedy)
epsilon_decay = 0.99  # epsilon 감소율
episodes = 5000  # 에피소드 수

# Q-table 초기화 (상태 공간 크기: 10x10x10x10, 액션 공간 크기: 2)
q_table = np.zeros((10, 10, 10, 10, env.action_space.n))

# 훈련 루프
for episode in range(episodes):
    state = env.reset()[0]  # 환경 리셋
    state = discretize_state(state)  # 상태 이산화
    done = False
    
    while not done:
        # Epsilon-greedy 방식으로 액션 선택
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # 랜덤 액션
        else:
            action = np.argmax(q_table[state])  # 최적 액션
        
        # 액션을 취하고 결과를 관찰
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)  # 다음 상태 이산화
        
        # Q-value 업데이트
        current_q = q_table[state + (action,)]  # 현재 상태와 액션에 대한 Q값
        next_max_q = np.max(q_table[next_state])  # 다음 상태에서의 최대 Q값
        new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)  # 새로운 Q값 계산
        q_table[state + (action,)] = new_q  # Q-table에 새로운 Q값 저장
        
        state = next_state  # 상태 갱신
    
    # 매 에피소드가 끝날 때마다 epsilon을 감소시킴
    epsilon = max(epsilon * epsilon_decay, 0.01)

    # 진행 상황 출력 (500번째 에피소드마다 출력)
    if episode % 500 == 0:
        print(f"Episode {episode} completed")

# 훈련이 끝난 후 환경을 닫음
env.close()


# 훈련된 에이전트를 테스트하고 렌더링
env = gym.make('CartPole-v1', render_mode="human")
state = env.reset()[0]
state = discretize_state(state)
done = False
total_reward = 0

print("Testing trained agent...")

while not done:
    action = np.argmax(q_table[state])  # Q-table에서 최적 액션을 선택
    next_state, reward, done, _, _ = env.step(action)  # 액션을 취하고 결과를 관찰
    next_state = discretize_state(next_state)  # 다음 상태 이산화
    total_reward += reward  # 총 보상 계산
    state = next_state  # 상태 갱신

    # 기둥이 너무 많이 넘어지면 종료
    if abs(state[2]) > 0.15:  # state[2]는 기둥의 각도 (pole angle)
        done = True
        print("The pole has fallen!")
    
    # 현재 상태 출력
    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    
    env.render()  # 매 스텝마다 렌더링하여 시각화

print(f"Total reward: {total_reward}")  # 총 보상 출력
env.close()  # 테스트 종료 후 환경을 닫음

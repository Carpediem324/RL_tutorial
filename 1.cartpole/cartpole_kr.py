import os
import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TensorBoard 로깅을 위한 모듈 임포트
from torch.utils.tensorboard import SummaryWriter

# 현재 스크립트 파일의 위치를 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))

# Gym 환경 생성 (CartPole-v1)
env = gym.make("CartPole-v1")

# 디바이스 설정 (GPU, Apple MPS, CPU 순)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# transition 데이터를 저장하기 위한 namedtuple 생성
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 경험 재생 메모리 클래스 정의
class ReplayMemory(object):
    def __init__(self, capacity):
        # 최대 capacity 만큼 저장되는 deque
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """배치 사이즈만큼 랜덤 샘플링"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """저장된 transition의 개수 반환"""
        return len(self.memory)

# DQN 모델 클래스 정의
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # 입력: 환경의 상태 관측값, 출력: 행동별 Q값
        self.layer1 = nn.Linear(n_observations, 128)   # 첫 번째 은닉층 (128개 뉴런)
        self.layer2 = nn.Linear(128, 128)                # 두 번째 은닉층 (128개 뉴런)
        self.layer3 = nn.Linear(128, n_actions)          # 출력층 (행동 수 만큼 출력)

    def forward(self, x):
        # 각 층에 ReLU 활성화 함수 적용 후 순차적으로 통과
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 하이퍼파라미터 설정
BATCH_SIZE = 128        # 리플레이 메모리에서 샘플링할 배치 크기
GAMMA = 0.99            # 할인율 (미래 보상 할인)
EPS_START = 0.9         # 초기 epsilon 값 (탐험 비율)
EPS_END = 0.05          # 최소 epsilon 값
EPS_DECAY = 1000        # epsilon 감소 속도
TAU = 0.005             # 타겟 네트워크 업데이트 속도 (Soft update)
LR = 1e-4               # 학습률

# 환경에서 행동(action) 수와 상태(state) 관측값 개수 획득
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# 정책 네트워크와 타겟 네트워크 생성 후 device로 이동
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# AdamW 옵티마이저 생성
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# 경험 재생 메모리 생성 (최대 10,000개 transition 저장)
memory = ReplayMemory(10000)

# TensorBoard SummaryWriter 생성: 현재 스크립트와 같은 디렉터리 아래 "runs/cartpole_experiment_1" 생성
log_dir = os.path.join(base_dir, "runs", "cartpole_experiment_1")
writer = SummaryWriter(log_dir)

# 현재까지 진행된 스텝 수 (탐험 정책 계산에 사용)
steps_done = 0

def select_action(state):
    """
    주어진 상태에 대해 epsilon-greedy 정책으로 행동 선택
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # 탐사(exploitation): 정책 네트워크의 Q값이 가장 큰 행동 선택
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # 탐험(exploration): 무작위 행동 선택
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    """
    경험 재생 메모리에서 샘플링한 데이터를 이용해 정책 네트워크 업데이트
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 여러 transition 데이터를 묶어서 배치 생성
    batch = Transition(*zip(*transitions))

    # 최종 상태가 아닌 transition들의 mask 생성
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                    device=device, dtype=torch.bool)
    # 최종 상태가 아닌 next_state들을 concatenate
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 현재 상태(state)에서 취한 행동(action)에 해당하는 Q값 선택
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 다음 상태(next state)에 대해 타겟 네트워크로 최대 Q값 계산
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # 벨만 방정식을 이용해 target Q값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber Loss (Smooth L1 Loss) 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 네트워크 파라미터 업데이트
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping (최대 100까지)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# 모델 저장 파일 경로: 현재 스크립트 파일이 있는 디렉터리 내에 "cartpole_dqn.pth" 생성
MODEL_FILENAME = os.path.join(base_dir, "cartpole_dqn.pth")

# 학습 모드 진행 (즉각적인 평가: 에피소드마다 진행 상황 출력)
# GPU나 MPS 사용 가능하면 에피소드 수를 600, 아니면 50으로 설정
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # 환경 초기화 및 상태 획득
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # 에피소드 내부에서 각 타임스텝 실행
    for t in count():
        # 현재 상태에 대한 행동 선택
        action = select_action(state)
        # 선택한 행동을 환경에 적용하여 다음 상태, 보상, 종료 여부 확인
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        # 에피소드 종료 시 next_state는 None 처리
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # transition (상태, 행동, 다음 상태, 보상) 저장
        memory.push(state, action, next_state, reward)

        # 상태 업데이트
        state = next_state

        # 정책 네트워크 최적화 (한 스텝 update)
        optimize_model()

        # 타겟 네트워크 소프트 업데이트: θ' ← τ * θ + (1-τ) * θ'
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        # 에피소드 종료 시 즉각적인 평가 출력 및 TensorBoard 로깅
        if done:
            episode_duration = t + 1  # 에피소드 동안 진행된 스텝 수
            writer.add_scalar("Episode/Duration", episode_duration, i_episode)
            print(f"에피소드: {i_episode + 1}, 지속 시간: {episode_duration} 스텝")
            break

print('학습 완료')

# 학습 완료 후, 정책 네트워크 파라미터를 pth 파일로 저장
torch.save(policy_net.state_dict(), MODEL_FILENAME)
print(f"모델이 {MODEL_FILENAME} 파일로 저장되었습니다.")

writer.close()
env.close()

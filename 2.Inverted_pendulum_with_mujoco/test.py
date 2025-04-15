import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# =====================
# 하이퍼파라미터 설정
# =====================
ENV_NAME = "InvertedPendulum-v4"  # 최신 Gym 환경 사용 (연속 액션)
LEARNING_RATE = 3e-4              # 학습률
GAMMA = 0.99                    # 할인율
LAMBDA = 0.95                   # GAE(lambda) 계수
EPS_CLIP = 0.2                  # PPO 클리핑 파라미터
K_EPOCH = 4                     # 업데이트 시 반복 횟수
ROLLOUT_LENGTH = 2048           # 한 업데이트 당 수집하는 timestep 수
BATCH_SIZE = 64                 # 미니배치 크기

# GPU 사용 가능 시 GPU 설정 (가능하면 GPU로 학습)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Actor-Critic 네트워크 정의
# - 연속 액션의 경우, Actor는 액션 평균 (mu) 값을 출력하며, 학습 가능한 log_std 파라미터를 이용해 표준편차를 구성합니다.
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        # 공유 네트워크
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        # Actor 네트워크: 평균(mu)을 출력 (마지막에 활성화 함수가 없음)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        # 학습 가능한 표준편차(log_std)를 Parameter로 정의 (상태 독립적이라고 가정)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic 네트워크: 상태 가치를 출력
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        mu = self.actor(x)
        # 표준편차는 exp(log_std)로 계산하며 배치 크기에 맞게 확장
        std = torch.exp(self.log_std).expand_as(mu)
        state_value = self.critic(x)
        return (mu, std), state_value

# =====================
# 경험 저장용 버퍼 클래스
# =====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.state_values.clear()

# =====================
# Generalized Advantage Estimation (GAE)
# =====================
def compute_gae(rewards, dones, state_values, next_value, gamma=GAMMA, lmbda=LAMBDA):
    advantages = []
    gae = 0
    # 마지막 timestep부터 역순 계산
    for i in reversed(range(len(rewards))):
        mask = 0 if dones[i] else 1  # done이면 mask=0
        delta = rewards[i] + gamma * next_value * mask - state_values[i]
        gae = delta + gamma * lmbda * gae * mask
        advantages.insert(0, gae)
        next_value = state_values[i]
    returns = [adv + sv for adv, sv in zip(advantages, state_values)]
    return advantages, returns

# =====================
# 현재 상태로부터 행동 선택 함수 (연속 액션)
# =====================
def select_action(model, state):
    # state가 numpy array가 아니라면 변환
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    # 1차원 state는 (1, state_dim)으로 변환
    if state.ndim == 1:
        state = state.reshape(1, -1)
    state_tensor = torch.from_numpy(state).float().to(device)
    (mu, std), state_value = model(state_tensor)
    # 정규분포를 생성 후 행동 샘플링
    dist = Normal(mu, std)
    action = dist.sample()
    # log probability 계산
    log_prob = dist.log_prob(action).sum(dim=-1)  # 다차원일 경우 합산
    return action.cpu().detach().numpy()[0], log_prob.item(), state_value.item()

# =====================
# 메인 학습 함수
# =====================
def main():
    # Gym 환경 생성 (연속 액션 Box, reset시 튜플이면 observation만 사용)
    env = gym.make(ENV_NAME)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    # 상태 및 행동 차원 (연속 액션이면 action_dim은 Box의 shape[0])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = RolloutBuffer()

    episode = 0
    timestep = 0
    max_episodes = 1000

    while episode < max_episodes:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        ep_reward = 0

        while not done:
            action, logprob, state_val = select_action(model, state)
            result = env.step(action)

            # Gym 반환값 처리: (observation, reward, done, info) 또는 (obs, reward, terminated, truncated, info)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # 버퍼에 경험 데이터 저장
            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(logprob)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.state_values.append(state_val)

            state = next_state
            ep_reward += reward
            timestep += 1

            # 일정 timestep마다 PPO 업데이트
            if timestep % ROLLOUT_LENGTH == 0:
                state_np = np.array(state) if not isinstance(state, np.ndarray) else state
                if state_np.ndim == 1:
                    state_np = state_np.reshape(1, -1)
                state_tensor = torch.from_numpy(state_np).float().to(device)
                with torch.no_grad():
                    _, next_state_value = model(state_tensor)
                    next_state_value = next_state_value.item()

                advantages, returns = compute_gae(buffer.rewards, buffer.dones, buffer.state_values, next_state_value)

                # 저장된 버퍼 데이터를 tensor로 변환
                states = torch.from_numpy(np.array(buffer.states)).float().to(device)
                actions = torch.from_numpy(np.array(buffer.actions)).float().to(device)
                old_logprobs = torch.FloatTensor(buffer.logprobs).to(device)
                returns = torch.FloatTensor(returns).to(device)
                advantages = torch.FloatTensor(advantages).to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dataset_size = states.size(0)
                for _ in range(K_EPOCH):
                    for i in range(0, dataset_size, BATCH_SIZE):
                        idx = slice(i, min(i + BATCH_SIZE, dataset_size))
                        batch_states = states[idx]
                        batch_actions = actions[idx]
                        batch_old_logprobs = old_logprobs[idx]
                        batch_returns = returns[idx]
                        batch_advantages = advantages[idx]

                        (mu, std), state_values = model(batch_states)
                        dist = Normal(mu, std)
                        new_logprobs = dist.log_prob(batch_actions).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()

                        ratio = torch.exp(new_logprobs - batch_old_logprobs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(state_values.flatten(), batch_returns)
                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                buffer.clear()

            if done:
                episode += 1
                print(f"Episode {episode} Reward: {ep_reward}")
                break

    env.close()

if __name__ == "__main__":
    main()

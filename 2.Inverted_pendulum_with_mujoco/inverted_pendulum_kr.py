import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import os
from collections import deque

# =====================
# 하이퍼파라미터 설정 및 설명
# =====================
ENV_NAME       = "InvertedPendulum-v4"  # 학습할 Gym 환경 이름
# LEARNING_RATE: 네트워크 가중치 업데이트 크기. 값이 크면 빠르게 학습하지만 불안정할 수 있고,
# 작으면 안정적이지만 학습 속도가 느려집니다. 필요 시 1e-3 ~ 1e-5 사이로 조정하세요.
LEARNING_RATE  = 1e-3   
# GAMMA: 할인율(discount factor). 0~1 사이 값으로 미래 보상에 대한 중요도.
# 1에 가까울수록 먼 미래 보상을 중시, 작을수록 단기 보상에 집중합니다.
GAMMA          = 0.5
# LAMBDA: GAE(lambda) 계수. 부트스트랩과 몬테카를로 사이의 절충.
# 낮추면 편향은 줄지만 분산이 커지고, 높이면 분산은 줄지만 편향이 증가합니다.
LAMBDA         = 0.95
# EPS_CLIP: PPO 클리핑 범위. 작게 설정(예:0.1)이면 정책 업데이트가 더 보수적,
# 크게 설정(예:0.3)이면 공격적인 업데이트가 이루어집니다.
EPS_CLIP       = 0.3
# K_EPOCH: 한 번 수집한 데이터를 반복 학습하는 횟수.
# 많으면 정책이 더 많이 수렴하나 과적합 위험이 있으며,
# 적으면 학습 효율이 떨어질 수 있습니다.
K_EPOCH        = 4         
# ROLLOUT_LENGTH: 한 업데이트 당 수집할 timestep 수.
# 길게 설정하면 보상 추정이 안정적이지만 메모리 사용량과 업데이트 간격이 늘어나고,
# 짧게 설정하면 자주 업데이트하지만 불안정할 수 있습니다.
ROLLOUT_LENGTH = 1024      
# BATCH_SIZE: 미니배치 크기. 작으면 잡음이 많지만 업데이트가 자주 이루어지고,
# 크면 안정적이지만 계산 비용이 증가합니다.
BATCH_SIZE     = 128       
# MAX_EPISODES: 총 학습 에피소드 수. 목표 성능에 따라 늘리거나 줄이세요.
MAX_EPISODES   = 1000

# 모델 및 결과 저장 경로
SAVE_DIR         = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_NAME = "ppo_invertedpendulum_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Actor-Critic 네트워크 정의
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        # shared: 상태 입력을 공통 표현으로 변환
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        # actor: 정책(평균값)을 출력, Tanh로 액션 범위 -1~1로 제한
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        # log_std: 액션 분산(log 표준편차), 학습 가능 파라미터
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # critic: 상태가치(value) 출력
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        mu = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mu)
        value = self.critic(x)
        return (mu, std), value

# =====================
# 경험 저장 버퍼
# =====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

# =====================
# GAE 계산 함수
# =====================
def compute_gae(rewards, dones, values, next_value, gamma=GAMMA, lmbda=LAMBDA):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        mask = 0 if dones[i] else 1
        delta = rewards[i] + gamma * next_value * mask - values[i]
        gae = delta + gamma * lmbda * gae * mask
        advantages.insert(0, gae)
        next_value = values[i]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

# =====================
# 액션 선택 함수 (클리핑 포함)
# =====================
ACTION_LOW = None  # 전역으로 환경 액션 범위 저장
ACTION_HIGH = None

def select_action(model, state):
    state = np.array(state, ndmin=2)
    state_tensor = torch.from_numpy(state).float().to(device)
    (mu, std), value = model(state_tensor)
    dist = Normal(mu, std)
    raw_action = dist.sample()
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    action = raw_action.cpu().detach().numpy()[0]
    # 환경이 허용하는 최소/최대 값 사이로 클리핑
    action = np.clip(action, ACTION_LOW, ACTION_HIGH)
    return action, log_prob.item(), value.item()

# =====================
# 메인 학습 루프
# =====================
def main():
    global ACTION_LOW, ACTION_HIGH

    env = gym.make(ENV_NAME)
    ACTION_LOW  = env.action_space.low
    ACTION_HIGH = env.action_space.high

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model     = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer    = RolloutBuffer()

    reward_history = deque(maxlen=100)
    episode = 0
    timestep = 0

    while episode < MAX_EPISODES:
        # Gym reset 구조 차이 처리
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        done = False
        ep_reward = 0

        while not done:
            action, logp, value = select_action(model, state)
            step_ret = env.step(action)
            if len(step_ret) == 4:
                next_state, reward, done, _ = step_ret
            else:
                next_state, reward, term, trunc, _ = step_ret
                done = term or trunc

            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(logp)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value)

            state = next_state
            ep_reward += reward
            timestep += 1

            # 지정된 Rollout 수집 후 업데이트
            if timestep % ROLLOUT_LENGTH == 0:
                # 다음 상태 가치 계산
                state_tensor = torch.from_numpy(np.array(state, ndmin=2)).float().to(device)
                _, next_val = model(state_tensor)
                next_val = next_val.item()

                advs, returns = compute_gae(
                    buffer.rewards, buffer.dones, buffer.values, next_val
                )

                # 텐서 변환
                states  = torch.FloatTensor(buffer.states).to(device)
                actions = torch.FloatTensor(buffer.actions).to(device)
                oldlogp = torch.FloatTensor(buffer.logprobs).to(device)
                returns = torch.FloatTensor(returns).to(device)
                advs    = torch.FloatTensor(advs).to(device)
                advs    = (advs - advs.mean()) / (advs.std() + 1e-8)

                dataset_size = states.size(0)
                for _ in range(K_EPOCH):
                    for start in range(0, dataset_size, BATCH_SIZE):
                        end = start + BATCH_SIZE
                        mb_idx = slice(start, end)

                        (mu, std), vals = model(states[mb_idx])
                        dist = Normal(mu, std)
                        newlogp  = dist.log_prob(actions[mb_idx]).sum(dim=-1)
                        entropy  = dist.entropy().sum(dim=-1).mean()

                        ratio = torch.exp(newlogp - oldlogp[mb_idx])
                        surr1 = ratio * advs[mb_idx]
                        surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advs[mb_idx]
                        actor_loss  = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(vals.flatten(), returns[mb_idx])
                        loss = actor_loss + 0.5*critic_loss - 0.01*entropy

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                buffer.clear()

            if done:
                episode += 1
                reward_history.append(ep_reward)
                avg_reward = np.mean(reward_history)
                print(f"Episode {episode}  Reward: {ep_reward:.1f}  Avg(100): {avg_reward:.1f}")
                break

    # 모델 저장
    save_path = os.path.join(SAVE_DIR, FINAL_MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print(f"최종 모델 저장 완료: {save_path}")
    env.close()

if __name__ == "__main__":
    main()

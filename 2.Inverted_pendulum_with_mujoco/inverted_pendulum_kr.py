import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import os

# =====================
# 하이퍼파라미터 설정
# =====================
ENV_NAME = "InvertedPendulum-v4"  # 최신 Gym 환경 사용 (연속 액션 환경)
LEARNING_RATE = 3e-4              # 학습률
GAMMA = 0.99                    # 할인율
LAMBDA = 0.95                   # GAE(lambda) 계수
EPS_CLIP = 0.2                  # PPO 클리핑 파라미터
K_EPOCH = 4                     # 업데이트 시 반복 횟수
ROLLOUT_LENGTH = 2048           # 한 업데이트 당 수집 timestep 수
BATCH_SIZE = 64                 # 미니배치 크기
MAX_EPISODES = 1000             # 총 학습 에피소드 수

# =====================
# 저장할 경로 설정: 현재 파이썬 코드 파일이 있는 경로
# =====================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_NAME = "ppo_invertedpendulum_final.pth"

# GPU 사용 가능한 경우 GPU 설정 (가능하면 GPU로 학습)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Actor-Critic 네트워크 정의 (연속 액션)
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        # 학습 가능한 log_std 파라미터를 정의 (상태 독립적)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        mu = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mu)
        state_value = self.critic(x)
        return (mu, std), state_value

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
    for i in reversed(range(len(rewards))):
        mask = 0 if dones[i] else 1
        delta = rewards[i] + gamma * next_value * mask - state_values[i]
        gae = delta + gamma * lmbda * gae * mask
        advantages.insert(0, gae)
        next_value = state_values[i]
    returns = [adv + sv for adv, sv in zip(advantages, state_values)]
    return advantages, returns

# =====================
# 행동 선택 함수 (연속 액션)
# =====================
def select_action(model, state):
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    state_tensor = torch.from_numpy(state).float().to(device)
    (mu, std), state_value = model(state_tensor)
    dist = Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    return action.cpu().detach().numpy()[0], log_prob.item(), state_value.item()

# =====================
# 메인 학습 함수
# =====================
def main():
    env = gym.make(ENV_NAME)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = RolloutBuffer()

    episode = 0
    timestep = 0

    while episode < MAX_EPISODES:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        ep_reward = 0

        while not done:
            action, logprob, state_val = select_action(model, state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(logprob)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.state_values.append(state_val)

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep % ROLLOUT_LENGTH == 0:
                state_np = np.array(state) if not isinstance(state, np.ndarray) else state
                if state_np.ndim == 1:
                    state_np = state_np.reshape(1, -1)
                state_tensor = torch.from_numpy(state_np).float().to(device)
                with torch.no_grad():
                    _, next_state_value = model(state_tensor)
                    next_state_value = next_state_value.item()

                advantages, returns = compute_gae(buffer.rewards, buffer.dones, buffer.state_values, next_state_value)

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

    # 학습 종료 후 최종 모델 저장 (파이썬 코드가 있는 폴더에 저장)
    final_save_path = os.path.join(SAVE_DIR, FINAL_MODEL_NAME)
    torch.save(model.state_dict(), final_save_path)
    print(f"최종 모델 저장 완료: {final_save_path}")

    env.close()

if __name__ == "__main__":
    main()

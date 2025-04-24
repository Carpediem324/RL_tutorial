import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import os
from collections import deque

# =====================
# 하이퍼파라미터
# =====================

ENV_NAME       = "Pendulum-v1"                # 어떤 게임(환경)을 학습할지 정해요
LEARNING_RATE  = 3e-6                         # 학습률: 올리면 빠르게 배우지만 불안정해지고, 내리면 느리지만 안정적이에요
GAMMA          = 0.99                         # 할인율: 올리면 먼 미래 보상 중시해 장기 전략을 배우지만 불안정해지고, 내리면 단기 집중해 빠르게 반응해요
LAMBDA         = 0.95                         # GAE λ: 올리면 보상 추정이 부드러워지고 분산이 줄지만 편향이 늘어나고, 내리면 편향은 줄지만 분산이 커져요
EPS_CLIP       = 0.2                          # 클리핑 범위: 올리면 정책 변화를 크게 허용해 공격적이고 불안정해지며, 내리면 보수적이고 느려져요
K_EPOCH        = 10                           # 반복 학습 횟수: 올리면 같은 데이터를 많이 학습해 과적합 위험이 있고 계산 비용이 커지며, 내리면 학습 효율이 떨어져요
ROLLOUT_LENGTH = 2048                         # 롤아웃 길이: 올리면 보상 추정이 안정적이지만 업데이트 간격이 길어지고, 내리면 자주 업데이트하지만 불안정해져요
BATCH_SIZE     = 256                          # 배치 크기: 올리면 업데이트가 안정적이지만 메모리·연산 비용이 커지고, 내리면 빠르지만 그라디언트 노이즈가 커져요
MAX_EPISODES   = 1000                         # 전체 에피소드 수: 올리면 더 오래 학습해 성능 향상의 여지가 있지만 시간·자원이 많이 들고, 내리면 빠르지만 학습이 부족해질 수 있어요


SAVE_DIR         = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_NAME = "ppo_pendulum-v1_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Actor-Critic 네트워크
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim),   nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x   = self.shared(x)
        mu  = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mu)
        val = self.critic(x)
        return (mu, std), val


# =====================
# GAE 계산
# =====================
def compute_gae(rewards, dones, values, next_value):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        mask  = 0 if dones[i] else 1
        delta = rewards[i] + GAMMA * next_value * mask - values[i]
        gae   = delta + GAMMA * LAMBDA * gae * mask
        advantages.insert(0, gae)
        next_value = values[i]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# =====================
# 액션 선택
# =====================
ACTION_LOW  = None
ACTION_HIGH = None

def select_action(model, state):
    state = np.array(state, ndmin=2)
    st_t  = torch.from_numpy(state).float().to(device)
    (mu, std), val = model(st_t)
    dist   = Normal(mu, std)
    raw    = dist.sample()
    logp   = dist.log_prob(raw).sum(dim=-1)
    action = raw.cpu().detach().numpy()[0]
    action = np.clip(action, ACTION_LOW, ACTION_HIGH)
    return action, logp.item(), val.item()


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

    model     = ActorCritic(state_dim, action_dim, hidden_size=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    buffer = {
        'states': [], 'actions': [], 'logprobs': [],
        'rewards': [], 'dones': [], 'values': []
    }
    reward_history = deque(maxlen=100)
    episode = 0
    timestep = 0

    while episode < MAX_EPISODES:
        raw = env.reset()
        state = raw[0] if isinstance(raw, tuple) else raw
        done = False
        ep_reward = 0

        while not done:
            action, logp, val = select_action(model, state)
            step = env.step(action)
            if len(step) == 4:
                next_state, reward, done, _ = step
            else:
                next_state, reward, term, trunc, _ = step
                done = term or trunc

            # 버퍼에 저장
            buffer['states'].append(state)
            buffer['actions'].append(action)
            buffer['logprobs'].append(logp)
            buffer['rewards'].append(reward)
            buffer['dones'].append(done)
            buffer['values'].append(val)

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep % ROLLOUT_LENGTH == 0:
                # 다음 상태 가치
                st_t = torch.from_numpy(
                    np.array(state, ndmin=2).astype(np.float32)
                ).to(device)
                _, next_val = model(st_t)
                next_val = next_val.item()

                advs, rets = compute_gae(
                    buffer['rewards'], buffer['dones'],
                    buffer['values'], next_val
                )

                # ⚡️ 여기부터 변경된 부분
                states_np   = np.array(buffer['states'], dtype=np.float32)
                actions_np  = np.array(buffer['actions'], dtype=np.float32)
                logp_np     = np.array(buffer['logprobs'], dtype=np.float32)
                returns_np  = np.array(rets, dtype=np.float32)
                advs_np     = np.array(advs, dtype=np.float32)

                states  = torch.from_numpy(states_np).to(device)
                actions = torch.from_numpy(actions_np).to(device)
                oldlogp = torch.from_numpy(logp_np).to(device)
                returns = torch.from_numpy(returns_np).to(device)
                advs    = torch.from_numpy(advs_np).to(device)
                advs    = (advs - advs.mean()) / (advs.std() + 1e-8)
                # ⚡️ 여기까지

                dataset_size = states.size(0)
                for _ in range(K_EPOCH):
                    for start in range(0, dataset_size, BATCH_SIZE):
                        end = start + BATCH_SIZE
                        mb = slice(start, end)

                        (mu, std), vals = model(states[mb])
                        dist   = Normal(mu, std)
                        newlogp  = dist.log_prob(actions[mb]).sum(dim=-1)
                        entropy  = dist.entropy().sum(dim=-1).mean()

                        ratio = torch.exp(newlogp - oldlogp[mb])
                        surr1 = ratio * advs[mb]
                        surr2 = torch.clamp(
                            ratio, 1-EPS_CLIP, 1+EPS_CLIP
                        ) * advs[mb]
                        actor_loss  = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(
                            vals.flatten(), returns[mb])
                        loss = actor_loss + 0.5*critic_loss - 0.01*entropy

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()

                # 버퍼 초기화
                for k in buffer: buffer[k].clear()

            if done:
                episode += 1
                reward_history.append(ep_reward)
                avg = np.mean(reward_history)
                print(f"Episode {episode}  Reward: {ep_reward:.1f}  Avg(100): {avg:.1f}")
                break

    # 모델 저장
    save_path = os.path.join(SAVE_DIR, FINAL_MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print(f"최종 모델 저장 완료: {save_path}")
    env.close()


if __name__ == "__main__":
    main()

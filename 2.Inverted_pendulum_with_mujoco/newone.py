import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import os
from collections import deque

# =====================
# 하이퍼파라미터 (수정된 부분)
# =====================
ENV_NAME       = "Pendulum-v1"
LEARNING_RATE  = 3e-4          # 3e-6 → 3e-4
GAMMA          = 0.99
LAMBDA         = 0.95
EPS_CLIP       = 0.2
K_EPOCH        = 4             # 10 → 4
ROLLOUT_LENGTH = 1024          # 2048 → 1024
BATCH_SIZE     = 64            # 256 → 64
MAX_EPISODES   = 1000

SAVE_DIR         = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_NAME = "ppo_pendulum-v1_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Actor-Critic 네트워크 (은닉 크기 수정)
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):  # 128 → 64
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x   = self.shared(x)
        mu  = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mu)
        val = self.critic(x)
        return (mu, std), val

# =====================
# GAE 계산 (변경 없음)
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
# 액션 선택 (변경 없음)
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
# 메인 학습 루프 (수정된 하이퍼파라미터 반영)
# =====================
def main():
    global ACTION_LOW, ACTION_HIGH

    env = gym.make(ENV_NAME)
    ACTION_LOW  = env.action_space.low
    ACTION_HIGH = env.action_space.high

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model     = ActorCritic(state_dim, action_dim, hidden_size=64).to(device)
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

            buffer['states'].append(state)
            buffer['actions'].append(action)
            buffer['logprobs'].append(logp)
            buffer['rewards'].append(reward)
            buffer['dones'].append(done)
            buffer['values'].append(val)

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep % ROLLOUT_LENGTH == 0:  # 1024 간격으로 업데이트
                st_t = torch.from_numpy(np.array(state, ndmin=2).astype(np.float32)).to(device)
                _, next_val = model(st_t)
                next_val = next_val.item()

                advs, rets = compute_gae(buffer['rewards'], buffer['dones'], buffer['values'], next_val)

                # NumPy → Tensor
                states   = torch.from_numpy(np.array(buffer['states'], dtype=np.float32)).to(device)
                actions  = torch.from_numpy(np.array(buffer['actions'], dtype=np.float32)).to(device)
                oldlogp  = torch.from_numpy(np.array(buffer['logprobs'], dtype=np.float32)).to(device)
                returns  = torch.from_numpy(np.array(rets, dtype=np.float32)).to(device)
                advs_t   = torch.from_numpy(np.array(advs, dtype=np.float32)).to(device)
                advs_t   = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

                dataset_size = states.size(0)
                for _ in range(K_EPOCH):  # 4 에폭
                    for start in range(0, dataset_size, BATCH_SIZE):  # 배치 64
                        mb = slice(start, start + BATCH_SIZE)

                        (mu, std), vals = model(states[mb])
                        dist    = Normal(mu, std)
                        newlogp = dist.log_prob(actions[mb]).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()

                        ratio = torch.exp(newlogp - oldlogp[mb])
                        surr1 = ratio * advs_t[mb]
                        surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advs_t[mb]
                        actor_loss  = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(vals.flatten(), returns[mb])
                        loss = actor_loss + 0.5*critic_loss - 0.01*entropy

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()

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

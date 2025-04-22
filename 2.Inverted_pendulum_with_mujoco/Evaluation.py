#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import numpy as np
import os

# =====================
# 사용할 환경 목록
# =====================
ENV_LIST = [
    "Pendulum-v1",               # 1: 초기각도 랜덤, 수직 세우기 과제
    "InvertedPendulum-v4",       # 2: 이미 수직으로 세워진 채 시작
    "InvertedDoublePendulum-v4", # 3: 두 링크 수직 세우기
    "CartPole-v1"                # 4: 카트 위 막대 수직 유지
]

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
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic  = nn.Sequential(
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
# 평가 함수 (랜덤 시작 포함)
# =====================
def evaluate_model(env_name, model_path, num_episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, render_mode="human" if render else None)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for ep in range(1, num_episodes+1):
        # 1) reset & 랜덤 초기화
        _ = env.reset()
        unwrapped = env.unwrapped
        # qpos/qvel 에 ±0.05 노이즈 추가
        qpos = unwrapped.init_qpos + np.random.uniform(-0.05, 0.05, unwrapped.init_qpos.shape)
        qvel = unwrapped.init_qvel + np.random.uniform(-0.05, 0.05, unwrapped.init_qvel.shape)
        unwrapped.set_state(qpos, qvel)
        state = unwrapped._get_obs()

        done = False
        total_reward = 0.0

        while not done:
            st_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                (mu, _), _ = model(st_t)
            action = mu.cpu().numpy().flatten()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            step_ret = env.step(action)
            if len(step_ret) == 4:
                next_state, reward, done, _ = step_ret
            else:
                next_state, reward, term, trunc, _ = step_ret
                done = term or trunc

            total_reward += reward
            state = next_state

        print(f"[{env_name} Evaluation] Episode {ep} → Reward: {total_reward:.1f}")

    env.close()

# =====================
# 메인: 실행 시 환경 선택
# =====================
if __name__ == "__main__":
    # 1) 환경 리스트 표시
    print("===== 사용할 환경 선택 =====")
    for i, name in enumerate(ENV_LIST, 1):
        print(f"{i}. {name}")
    idx = int(input("환경 번호를 입력하세요: ")) - 1
    env_name = ENV_LIST[idx]

    # 2) 모델 파일 경로 (필요에 따라 파일명 수정)
    #    여기서는 "{env_name.lower()}_final.pth" 형태로 가정
    model_filename = f"ppo_{env_name.lower()}_final.pth"
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    # 3) 평가 실행
    evaluate_model(env_name, model_path, num_episodes=5, render=True)

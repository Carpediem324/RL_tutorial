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
# TD3용 액터 네트워크
# =====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
    
    def forward(self, state):
        return self.max_action * self.net(state)

# =====================
# 평가 함수 (랜덤 시작 포함)
# =====================
def evaluate_model(env_name, model_path, num_episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, render_mode="human" if render else None)

    # 모델 로드
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # TD3 액터 모델 생성
    actor = Actor(state_dim, action_dim, max_action).to(device)
    
    # TD3 체크포인트에서 액터 모델만 로드
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'actor' in checkpoint:
        # TD3 저장 형식인 경우
        actor.load_state_dict(checkpoint['actor'])
    else:
        # 직접 액터 state_dict만 저장된 경우
        actor.load_state_dict(checkpoint)
    
    actor.eval()

    for ep in range(1, num_episodes+1):
        # 1) reset
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

        # 2) MuJoCo 환경이면 init_qpos/init_qvel 을 이용해 랜덤 시작,
        #    아니면 reset() 만으로 충분히 랜덤이므로 obs 그대로 사용
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'init_qpos'):
            # MuJoCo 환경
            qpos = unwrapped.init_qpos + \
                   np.random.uniform(-0.05, 0.05, unwrapped.init_qpos.shape)
            qvel = unwrapped.init_qvel + \
                   np.random.uniform(-0.05, 0.05, unwrapped.init_qvel.shape)
            unwrapped.set_state(qpos, qvel)
            state = unwrapped._get_obs()
        else:
            # Classic 환경
            state = obs

        done = False
        total_reward = 0.0

        while not done:
            st_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(st_t).cpu().numpy().flatten()
            
            action = np.clip(action,
                             env.action_space.low,
                             env.action_space.high)

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
    print("===== 사용할 환경 선택 =====")
    for i, name in enumerate(ENV_LIST, 1):
        print(f"{i}. {name}")
    idx = int(input("환경 번호를 입력하세요: ")) - 1
    env_name = ENV_LIST[idx]

    model_filename = f"td3_pendulum_best.pth"
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    evaluate_model(env_name, model_path, num_episodes=5, render=True)

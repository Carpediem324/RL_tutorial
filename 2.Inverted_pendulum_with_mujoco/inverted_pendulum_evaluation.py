import gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import os
import time

# =====================
# 환경 및 모델 설정
# =====================
ENV_NAME = "InvertedPendulum-v4"  # 평가할 Gym 환경 (MuJoCo 기반)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_invertedpendulum_final.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Actor-Critic 네트워크 정의 (훈련 시 사용한 네트워크와 동일하게)
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        # 공유 네트워크
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        # Actor 네트워크: 평균(mu)을 출력 (연속 액션)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        # 학습 가능한 log_std 파라미터 (상태 독립적)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic 네트워크 (평가용)
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
# 평가 함수: 저장된 모델을 로드하여 환경에서 에이전트를 실행하는 코드
# =====================
def evaluate_model(num_episodes=5):
    # render_mode="human" 옵션을 추가하여 화면에 시뮬레이션 창이 뜨게 함
    env = gym.make(ENV_NAME, render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # 평가 모드

    total_rewards = []
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        ep_reward = 0
        
        while not done:
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            state_tensor = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                # deterministic policy: 평균(mu) 사용
                (mu, _), _ = model(state_tensor)
                action = mu.cpu().numpy()[0]
            
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
                
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            ep_reward += reward
            state = next_state

            env.render()
            time.sleep(0.01)  # 프레임 간 잠시 대기
        
        total_rewards.append(ep_reward)
        print(f"Episode {ep} Reward: {ep_reward}")
    
    env.close()
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"평균 리워드: {avg_reward}")

if __name__ == "__main__":
    evaluate_model(num_episodes=5)

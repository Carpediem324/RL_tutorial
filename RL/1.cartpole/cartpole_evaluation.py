# cartpole_evaluation.py
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import time

# 학습 시 사용한 DQN 모델 구조와 동일하게 정의 (변수명: layer1, layer2, layer3)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 현재 스크립트 파일이 위치한 디렉터리 경로를 얻어, 거기에 "cartpole_dqn.pth" 파일의 절대경로 설정
MODEL_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_dqn.pth")

def evaluate_model(model, env, num_episodes=10):
    """
    저장된 모델을 사용하여 CartPole 환경에서 평가하는 함수.
    GUI 창을 통해 실시간으로 환경 상태를 확인할 수 있습니다.
    """
    model.eval()  # 평가 모드 전환 (드롭아웃, 배치 정규화 등 비활성화)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        # 상태를 tensor로 변환 및 배치 차원 추가
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        
        for t in count():
            env.render()  # GUI로 환경 상태 출력
            with torch.no_grad():
                # 탐험 없이 모델이 출력하는 최대 Q값 행동 선택
                action = model(state).max(1)[1].view(1, 1)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            
            if terminated or truncated:
                print(f"평가 에피소드 {episode + 1}: 총 보상 = {total_reward}, 스텝 수 = {t + 1}")
                break
            
            # 다음 상태 tensor 변환
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            time.sleep(0.02)  # GUI 업데이트를 위한 잠깐의 지연

    env.close()

if __name__ == "__main__":
    # CartPole 환경 생성 시 render_mode="human"로 GUI 창 활성화
    env = gym.make("CartPole-v1", render_mode="human")
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n
    
    # DQN 모델 생성 (학습 시 사용한 구조와 동일)
    model = DQN(n_observations, n_actions)
    # 저장된 모델 파라미터 불러오기 (파일은 스크립트와 같은 위치에 있어야 함)
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=torch.device("cpu")))
    print("저장된 모델을 불러왔습니다.")
    
    # 평가 에피소드 10회 실행 (GUI로 표시)
    evaluate_model(model, env, num_episodes=10)

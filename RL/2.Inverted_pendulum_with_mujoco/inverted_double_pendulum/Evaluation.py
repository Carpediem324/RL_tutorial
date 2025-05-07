#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# =====================
# 환경 설정
# =====================
ENV_NAME = "InvertedDoublePendulum-v4"  # 두 링크 수직 세우기

# =====================
# Actor-Critic 네트워크
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        # 공유 네트워크 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 액터 네트워크 (평균)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh()  # 액션 범위 [-1, 1]로 제한
        )
        
        # 로그 표준편차 (학습 가능한 파라미터)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        
        # 크리틱 네트워크 (상태 가치)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
                
    def forward(self, x):
        x = self.shared(x)
        
        # 액터: 평균과 표준편차
        mu = self.actor_mean(x)
        std = torch.exp(self.log_std.clamp(-20, 2))  # 안정성을 위한 클램핑
        
        # 크리틱: 상태 가치
        val = self.critic(x)
        
        return (mu, std), val

# =====================
# 평가 함수 (개선됨)
# =====================
def evaluate_model(model_path, num_episodes=5, render=True, record=False, random_init=True):
    """
    학습된 모델을 평가하는 함수
    
    Args:
        model_path: 모델 파일 경로
        num_episodes: 평가할 에피소드 수
        render: 화면에 렌더링 여부
        record: 비디오 녹화 여부
        random_init: 랜덤 초기 상태 사용 여부
    """
    print(f"InvertedDoublePendulum 모델 평가 시작")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 환경 생성 (렌더링 모드 설정)
    render_mode = "human" if render else None
    
    # 환경 및 비디오 녹화 설정
    if record:
        video_env = None
        try:
            # 최신 gymnasium 사용 시도
            import gymnasium
            video_env = gymnasium.make(ENV_NAME, render_mode=render_mode)
            from gymnasium.wrappers import RecordVideo
            env = RecordVideo(video_env, f"videos/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print("Gymnasium 환경으로 비디오 녹화 설정됨")
        except (ImportError, AttributeError):
            if video_env:
                video_env.close()
            
            try:
                # gym 사용 시도
                video_env = gym.make(ENV_NAME, render_mode=render_mode)
                from gym.wrappers import RecordVideo
                env = RecordVideo(video_env, f"videos/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                print("Gym 환경으로 비디오 녹화 설정됨")
            except (ImportError, AttributeError):
                print("비디오 녹화 기능을 사용할 수 없습니다. render=True로 설정합니다.")
                env = gym.make(ENV_NAME, render_mode="human")
    else:
        env = gym.make(ENV_NAME, render_mode=render_mode)
    
    # 모델 로드
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim, hidden_size=256).to(device)
    
    # 모델 로드 시도 (여러 형식 지원)
    try:
        # 1. 직접 state_dict 로드 시도
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"모델 로드 성공 (state_dict): {model_path}")
    except Exception as e1:
        print(f"state_dict 로드 실패: {e1}")
        try:
            # 2. 체크포인트 형식 로드 시도
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"모델 로드 성공 (checkpoint): {model_path}")
            else:
                # 3. 호환성 모드로 로드 시도
                model.load_state_dict(checkpoint, strict=False)
                print(f"모델 로드 성공 (호환성 모드): {model_path}")
        except Exception as e2:
            print(f"모델 로드 실패: {e2}")
            return
    
    # 평가 모드로 설정
    model.eval()
    
    # 평가 결과 저장용
    total_rewards = []
    episode_lengths = []
    
    for ep in range(1, num_episodes+1):
        # 환경 초기화
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            state = reset_ret[0]
        else:
            state = reset_ret
        
        # 랜덤 초기 상태 설정 (선택적)
        if random_init:
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'init_qpos'):
                # MuJoCo 환경에서 랜덤 초기 상태 설정
                qpos = unwrapped.init_qpos + np.random.uniform(-0.1, 0.1, unwrapped.init_qpos.shape)
                qvel = unwrapped.init_qvel + np.random.uniform(-0.1, 0.1, unwrapped.init_qvel.shape)
                unwrapped.set_state(qpos, qvel)
                state = unwrapped._get_obs()
        
        done = False
        total_reward = 0.0
        step_count = 0
        
        print(f"에피소드 {ep} 시작...")
        start_time = time.time()
        
        while not done:
            # 상태를 텐서로 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # 모델에서 액션 추출 (결정적 모드)
            with torch.no_grad():
                (mu, _), _ = model(state_tensor)
                action = mu.cpu().numpy().flatten()
            
            # 액션을 환경의 범위로 클리핑
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # 환경에서 한 스텝 진행
            try:
                step_ret = env.step(action)
                
                # 반환값 처리 (gym 버전에 따라 다름)
                if len(step_ret) == 5:  # 새로운 Gym API (v0.26+)
                    next_state, reward, terminated, truncated, _ = step_ret
                    done = terminated or truncated
                else:  # 이전 Gym API
                    next_state, reward, done, _ = step_ret
                
                # 보상 누적 및 상태 업데이트
                total_reward += reward
                state = next_state
                step_count += 1
                
                # 너무 오래 실행되는 것 방지
                if step_count >= 1000:
                    print("최대 스텝 수 도달")
                    break
                    
            except Exception as e:
                print(f"환경 스텝 실행 중 오류: {e}")
                break
        
        # 에피소드 결과 저장
        episode_time = time.time() - start_time
        total_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        print(f"에피소드 {ep} 완료 → 보상: {total_reward:.1f}, 스텝: {step_count}, 시간: {episode_time:.2f}초")
    
    # 평가 결과 요약
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n===== 평가 결과 요약 =====")
    print(f"에피소드 수: {num_episodes}")
    print(f"평균 보상: {avg_reward:.1f}")
    print(f"평균 스텝 수: {avg_length:.1f}")
    print(f"최대 보상: {max(total_rewards):.1f}")
    print(f"최소 보상: {min(total_rewards):.1f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_episodes+1), total_rewards, alpha=0.7)
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'평균: {avg_reward:.1f}')
    plt.xlabel('에피소드')
    plt.ylabel('총 보상')
    plt.title('InvertedDoublePendulum 평가 결과')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"evaluation_results_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"평가 결과 그래프 저장: {plot_path}")
    
    # 환경 종료
    env.close()
    print("환경 종료")
    
    return avg_reward, total_rewards

# =====================
# 메인 실행
# =====================
if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='InvertedDoublePendulum 모델 평가')
    parser.add_argument('--model', type=str, default="ppo_InvertedDoublePendulum_final.pth",
                        help='평가할 모델 파일 이름')
    parser.add_argument('--episodes', type=int, default=5, 
                        help='평가할 에피소드 수')
    parser.add_argument('--no-render', action='store_true',
                        help='렌더링 비활성화')
    parser.add_argument('--record', action='store_true',
                        help='비디오 녹화 활성화')
    parser.add_argument('--no-random', action='store_true',
                        help='랜덤 초기화 비활성화')
    
    args = parser.parse_args()
    
    # 모델 경로 설정
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
    print(f"모델 경로: {model_path}")
    
    # 모델 평가 실행
    evaluate_model(
        model_path=model_path,
        num_episodes=args.episodes,
        render=not args.no_render,
        record=args.record,
        random_init=not args.no_random
    )

import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# 경고 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =====================
# 하이퍼파라미터
# =====================
ENV_NAME = "InvertedDoublePendulum-v4"  # 두 개의 링크를 함께 수직으로 세우기
HIDDEN_SIZE = 512        # 은닉층 크기: 512로 증가하여 표현력 향상
LEARNING_RATE = 3e-4     # 학습률: PPO에 권장되는 값으로 조정
GAMMA = 0.99             # 할인율: 장기 보상 고려
LAMBDA = 0.95            # GAE λ: 편향-분산 트레이드오프
EPS_CLIP = 0.2           # 클리핑 범위: PPO 논문 권장값
K_EPOCH = 10             # 반복 학습 횟수: 데이터 재사용
ROLLOUT_LENGTH = 8000    # 롤아웃 길이: 1024로 줄여 더 자주 업데이트
BATCH_SIZE = 128         # 배치 크기: 128로 증가하여 안정적인 그라디언트
MAX_EPISODES = 10000     # 전체 에피소드 수
SAVE_INTERVAL = 50       # 저장 간격
PLOT_INTERVAL = 10       # 그래프 그리기 간격
NUM_ENVS = 8             # 병렬 환경 수
VALUE_COEF = 0.7         # 가치 손실 계수: 0.7로 증가하여 Critic 빠른 수렴
ENTROPY_COEF = 0.02      # 엔트로피 계수: 0.02로 증가하여 탐색 강화
MAX_GRAD_NORM = 0.5      # 그래디언트 클리핑 최대 노름
EARLY_STOP_REWARD = 9000 # 조기 종료 보상 기준 (InvertedDoublePendulum 최대 성능 근처)
WARMUP_STEPS = 100       # 웜업 스텝 수

# 디렉토리 설정
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"학습 디바이스: {device}")

# =====================
# 정규화 클래스 (안전한 구현)
# =====================
class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon
        
    def update(self, x):
        try:
            # 배열 형태 확인 및 변환
            if isinstance(x, list):
                x = np.array(x, dtype=np.float32)
            
            # 배열 차원 확인
            if len(x.shape) == 1:
                x = np.expand_dims(x, 0)  # 단일 상태를 2D로 변환
                
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / (self.count + batch_count)
            
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            
            self.mean = new_mean
            self.var = M2 / (self.count + batch_count)
            self.count += batch_count
        except Exception as e:
            print(f"정규화 업데이트 중 오류: {e}")
        
    def normalize(self, x):
        try:
            if isinstance(x, list):
                x = np.array(x, dtype=np.float32)
                
            # 배열 차원 확인
            if len(x.shape) == 1:
                x = np.expand_dims(x, 0)  # 단일 상태를 2D로 변환
                
            return (x - self.mean) / np.sqrt(self.var + 1e-8)
        except Exception as e:
            print(f"정규화 중 오류: {e}")
            return x  # 오류 발생 시 원본 반환

# =====================
# 병렬 환경 클래스 (최적화)
# =====================
from gym.wrappers import TimeLimit
class ParallelEnvs:
    def __init__(self, env_name, num_envs=NUM_ENVS):
        base = gym.make(env_name)
        max_steps = base.spec.max_episode_steps or 1000
        self.envs = [
            TimeLimit(gym.make(env_name), max_episode_steps=max_steps)
            for _ in range(num_envs)
        ]
        self.is_vector_env = True
        
        # 환경 정보 가져오기
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # 환경 초기화
        self.states = self.reset()
        
    def reset(self):
        states = []
        for env in self.envs:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]  # gym 최신 버전 호환
            else:
                state = reset_result
            states.append(state)
        return np.array(states, dtype=np.float32)
        
    def step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            try:
                step_result = env.step(action)
                
                if len(step_result) == 5:  # 새로운 Gym API (v0.26+)
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # 이전 Gym API
                    next_state, reward, done, info = step_result
                    
                if done:
                    # 환경 재설정
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        reset_state = reset_result[0]
                    else:
                        reset_state = reset_result
                    next_state = reset_state
                    
                next_states.append(next_state)
                rewards.append(float(reward))  # 스칼라로 변환
                dones.append(bool(done))       # 불리언으로 변환
                infos.append(info)
            except Exception as e:
                print(f"환경 {i} 스텝 실행 중 오류: {e}")
                # 오류 발생 시 리셋하고 계속 진행
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    reset_state = reset_result[0]
                else:
                    reset_state = reset_result
                next_states.append(reset_state)
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
            
        self.states = np.array(next_states, dtype=np.float32)
        return np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos
        
    def close(self):
        for env in self.envs:
            env.close()

# =====================
# Actor-Critic 네트워크 (개선됨)
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super().__init__()
        # 공유 네트워크 레이어 (더 깊은 네트워크)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 추가 레이어
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
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        x = self.shared(x)
        
        # 액터: 평균과 표준편차
        mu = self.actor_mean(x)
        std = torch.exp(self.log_std.clamp(-20, 2))  # 안정성을 위한 클램핑
        
        # 크리틱: 상태 가치
        val = self.critic(x)
        
        return (mu, std), val
    
    def get_action(self, x, deterministic=False):
        (mu, std), val = self.forward(x)
        
        if deterministic:
            return mu, val
        
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, val

# =====================
# PPO 에이전트 (최적화)
# =====================
class PPOAgent:
    def __init__(self):
        # 병렬 환경 초기화
        self.envs = ParallelEnvs(ENV_NAME, NUM_ENVS)
        
        # 상태 및 액션 차원 가져오기
        self.state_dim = self.envs.observation_space.shape[0]
        self.action_dim = self.envs.action_space.shape[0]
        
        # 액션 범위 설정
        self.action_low = self.envs.action_space.low
        self.action_high = self.envs.action_space.high
        
        print(f"환경: {ENV_NAME}")
        print(f"상태 차원: {self.state_dim}")
        print(f"액션 차원: {self.action_dim}")
        print(f"액션 범위: {self.action_low} ~ {self.action_high}")
        print(f"병렬 환경 수: {NUM_ENVS}")
        
        # 모델 및 옵티마이저 초기화
        self.model = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # 웜업이 있는 학습률 스케줄러
        def lr_lambda(step):
            if step < WARMUP_STEPS:
                return float(step) / float(max(1, WARMUP_STEPS))
            else:
                return 1.0 - min(1.0, (float(step - WARMUP_STEPS) / float(max(1, MAX_EPISODES - WARMUP_STEPS))))
                
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # 상태 정규화 추가
        self.obs_normalizer = RunningMeanStd(shape=(self.state_dim,))
        
        # 보상 정규화
        self.reward_normalizer = RunningMeanStd()
        
        # 보상 기록용
        self.reward_history = deque(maxlen=100)
        self.all_rewards = []
        
        # 에피소드 정보
        self.episode_rewards = [0] * NUM_ENVS
        self.episode_lengths = [0] * NUM_ENVS
        self.completed_episodes = 0
        
        # 학습 통계
        self.timesteps = 0
        self.updates = 0
        
    def select_actions(self, states):
        # 상태 정규화 적용 (안전하게)
        try:
            normalized_states = self.obs_normalizer.normalize(states)
        except Exception as e:
            print(f"상태 정규화 중 오류: {e}")
            normalized_states = states
        
        # 상태를 텐서로 변환
        states_tensor = torch.FloatTensor(normalized_states).to(device)
        
        # 모델에서 액션 분포와 가치 추정
        with torch.no_grad():
            actions, log_probs, values = self.model.get_action(states_tensor)
            
        # 액션을 CPU로 이동하고 넘파이 배열로 변환
        actions_np = actions.cpu().numpy()
        
        # 액션을 환경의 범위로 클리핑
        actions_np = np.clip(actions_np, self.action_low, self.action_high)
        
        return actions_np, log_probs.cpu().numpy(), values.cpu().numpy()
    
    def collect_rollouts(self):
        states = []
        actions = []
        logprobs = []
        values = []
        rewards = []
        dones = []
        
        # 현재 상태에서 시작
        current_states = self.envs.states
        
        # 상태 정규화 업데이트
        self.obs_normalizer.update(current_states)
        
        for _ in range(ROLLOUT_LENGTH // NUM_ENVS):
            self.timesteps += NUM_ENVS
            
            action_array, logprob_array, value_array = self.select_actions(current_states)
            next_states, reward_array, done_array, _ = self.envs.step(action_array)
            
            # 보상 정규화 업데이트 및 적용
            self.reward_normalizer.update(reward_array)
            normalized_rewards = self.reward_normalizer.normalize(reward_array)
            
            # ★ 추가: (1, NUM_ENVS) 형태를 (NUM_ENVS,)로 바꿔줍니다.
            if normalized_rewards.ndim > 1:
                normalized_rewards = normalized_rewards.squeeze(0)
            
            # 데이터 저장
            states.append(current_states)
            actions.append(action_array)
            logprobs.append(logprob_array)
            values.append(value_array)
            rewards.append(normalized_rewards)  # 이제 1D 배열
            dones.append(done_array)
            
            # 에피소드 보상/길이 업데이트...
            # (이하 생략)
            
            current_states = next_states
            self.obs_normalizer.update(current_states)
        
        # 마지막 상태의 가치 추정
        _, _, next_values = self.select_actions(current_states)
        
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'logprobs': np.array(logprobs, dtype=np.float32),
            'values': np.array(values, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),  # shape: (timesteps, NUM_ENVS)
            'dones': np.array(dones, dtype=bool),            # shape: (timesteps, NUM_ENVS)
            'next_values': next_values
        }

    def compute_advantages(self, rollout_data):
        values = rollout_data['values']      # shape: (timesteps, NUM_ENVS, 1)
        rewards = rollout_data['rewards']    # shape: (timesteps, NUM_ENVS)
        dones = rollout_data['dones']        # shape: (timesteps, NUM_ENVS)
        next_values = rollout_data['next_values']
        
        timesteps = len(rewards)
        
        # ★ 추가: values도 (timesteps, NUM_ENVS)로 변환
        if values.ndim == 3 and values.shape[2] == 1:
            values = values.squeeze(2)
        
        # next_values 역시 (NUM_ENVS,)로
        if hasattr(next_values, 'shape') and next_values.ndim == 2 and next_values.shape[1] == 1:
            next_values = next_values.squeeze(1)
        
        advantages = np.zeros((timesteps, NUM_ENVS), dtype=np.float32)
        returns = np.zeros((timesteps, NUM_ENVS), dtype=np.float32)
        
        for env_idx in range(NUM_ENVS):
            last_gae = 0.0
            last_value = float(next_values[env_idx])
            
            for t in reversed(range(timesteps)):
                mask = 1.0 - float(dones[t, env_idx])
                
                next_value = last_value if t == timesteps - 1 else float(values[t+1, env_idx])
                current_value = float(values[t, env_idx])
                current_reward = float(rewards[t, env_idx])
                
                delta = current_reward + GAMMA * next_value * mask - current_value
                last_gae = delta + GAMMA * LAMBDA * last_gae * mask
                
                advantages[t, env_idx] = last_gae
                returns[t, env_idx] = last_gae + current_value
                last_value = current_value
        
        return advantages, returns

    def update_policy(self, rollout_data, advantages, returns):
        # 데이터 준비
        states = rollout_data['states'].reshape(-1, self.state_dim)
        actions = rollout_data['actions'].reshape(-1, self.action_dim)
        old_logprobs = rollout_data['logprobs'].reshape(-1)
        
        # 어드밴티지와 리턴 준비
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        
        # 어드밴티지 정규화
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # 상태 정규화 적용 (안전하게)
        try:
            normalized_states = self.obs_normalizer.normalize(states)
        except Exception as e:
            print(f"상태 정규화 중 오류: {e}")
            normalized_states = states
        
        # 텐서 변환
        states_tensor = torch.FloatTensor(normalized_states).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        old_logprobs_tensor = torch.FloatTensor(old_logprobs).to(device)
        advantages_tensor = torch.FloatTensor(advantages_flat).to(device)
        returns_tensor = torch.FloatTensor(returns_flat).to(device)
        
        # 데이터셋 및 데이터로더 생성
        dataset = TensorDataset(
            states_tensor, actions_tensor, old_logprobs_tensor, 
            advantages_tensor, returns_tensor
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # PPO 업데이트 (K_EPOCH 횟수만큼 반복)
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(K_EPOCH):
            for mb_states, mb_actions, mb_old_logprobs, mb_advantages, mb_returns in dataloader:
                # 모델에서 새로운 액션 분포와 가치 추정
                (mu, std), values = self.model(mb_states)
                dist = Normal(mu, std)
                
                # 새로운 액션 로그 확률 계산
                new_logprobs = dist.log_prob(mb_actions).sum(dim=1)
                
                # 엔트로피 계산 (탐색 촉진)
                entropy = dist.entropy().sum(dim=1).mean()
                
                # 정책 비율 계산 (새 정책 / 이전 정책)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                
                # PPO 클리핑 목적 함수
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * mb_advantages
                
                # 액터 손실 (음수 취함 - 최대화 목적)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 크리틱 손실 (MSE)
                critic_loss = nn.MSELoss()(values.squeeze(-1), mb_returns)
                
                # 전체 손실 (액터 + 크리틱 - 엔트로피 보너스)
                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy
                
                # 그래디언트 계산 및 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑 (폭발 방지)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                
                # 파라미터 업데이트
                self.optimizer.step()
                
                # 손실 누적
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # 스케줄러 업데이트
        self.scheduler.step()
        self.updates += 1
        
        # 평균 손실 계산
        num_batches = K_EPOCH * len(dataloader)
        avg_actor_loss = total_actor_loss / num_batches if num_batches > 0 else 0
        avg_critic_loss = total_critic_loss / num_batches if num_batches > 0 else 0
        avg_entropy = total_entropy / num_batches if num_batches > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_entropy
    
    def train(self):
        # 학습 시작 시간 기록
        start_time = datetime.now()
        start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
        print(f"학습 시작: {start_time_str}")
        
        # 최고 성능 기록용
        best_avg_reward = -float('inf')
        
        try:
            # 메인 학습 루프
            while self.completed_episodes < MAX_EPISODES:
                # 롤아웃 수집
                rollout_data = self.collect_rollouts()
                
                # 어드밴티지 및 리턴 계산
                advantages, returns = self.compute_advantages(rollout_data)
                
                # 정책 업데이트
                actor_loss, critic_loss, entropy = self.update_policy(rollout_data, advantages, returns)
                
                # 평균 보상 계산
                if len(self.reward_history) > 0:
                    avg_reward = sum(self.reward_history) / len(self.reward_history)
                else:
                    avg_reward = 0
                
                # 현재 학습률
                current_lr = self.scheduler.get_last_lr()[0]
                
                # 학습 진행 상황 출력
                print(f"에피소드: {self.completed_episodes}/{MAX_EPISODES} | "
                      f"타임스텝: {self.timesteps} | "
                      f"업데이트: {self.updates} | "
                      f"평균 보상: {avg_reward:.2f} | "
                      f"Actor 손실: {actor_loss:.4f} | "
                      f"Critic 손실: {critic_loss:.4f} | "
                      f"엔트로피: {entropy:.4f} | "
                      f"학습률: {current_lr:.6f}")
                
                # 최고 성능 모델 저장
                if avg_reward > best_avg_reward and len(self.reward_history) >= 10:
                    best_avg_reward = avg_reward
                    self.save_model(f"best_model_{start_time_str}.pth")
                    print(f"최고 성능 모델 저장 (평균 보상: {avg_reward:.2f})")
                
                # 주기적 체크포인트 저장
                if self.completed_episodes % SAVE_INTERVAL == 0 and self.completed_episodes > 0:
                    self.save_model(f"checkpoint_ep{self.completed_episodes}_{start_time_str}.pth")
                
                # 학습 곡선 그리기
                if self.completed_episodes % PLOT_INTERVAL == 0 and self.completed_episodes > 0:
                    self.plot_learning_curve(self.completed_episodes, start_time_str)
                
                # 조기 종료 확인
                if avg_reward >= EARLY_STOP_REWARD and len(self.reward_history) >= 10:
                    print(f"목표 보상 {EARLY_STOP_REWARD} 달성! 학습 조기 종료.")
                    break
                
        except KeyboardInterrupt:
            print("학습 중단됨 - 현재 모델 저장 중...")
        
        finally:
            # 최종 모델 저장
            final_model_path = os.path.join(CHECKPOINT_DIR, f"final_model_{start_time_str}.pth")
            self.save_model(f"final_model_{start_time_str}.pth")
            
            # 표준 이름으로도 저장 (평가 코드용)
            standard_path = os.path.join(SAVE_DIR, "ppo_InvertedDoublePendulum_final.pth")
            torch.save(self.model.state_dict(), standard_path)
            print(f"표준 이름으로 모델 저장: {standard_path}")
            
            # 환경 종료
            self.envs.close()
            
            # 학습 시간 계산
            end_time = datetime.now()
            training_time = end_time - start_time
            print(f"학습 완료! 총 소요 시간: {training_time}")
            
            # 최종 학습 곡선 그리기
            self.plot_learning_curve(self.completed_episodes, start_time_str, final=True)
    
    def save_model(self, filename):
        """모델 저장"""
        path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'reward_history': list(self.reward_history),
            'all_rewards': self.all_rewards,
            'completed_episodes': self.completed_episodes,
            'timesteps': self.timesteps,
            'obs_normalizer_mean': self.obs_normalizer.mean,
            'obs_normalizer_var': self.obs_normalizer.var,
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_var': self.reward_normalizer.var,
        }, path)
        print(f"모델 저장됨: {path}")
    
    def plot_learning_curve(self, episode, timestamp, final=False):
        """학습 곡선 그리기"""
        plt.figure(figsize=(12, 8))
        
        # 에피소드 보상 그래프
        plt.subplot(2, 1, 1)
        plt.plot(self.all_rewards, label='Episode Reward', alpha=0.6)
        
        # 이동 평균 계산 및 플롯
        if len(self.all_rewards) >= 10:
            moving_avg = [np.mean(self.all_rewards[max(0, i-9):i+1]) 
                         for i in range(len(self.all_rewards))]
            plt.plot(moving_avg, label='10 Episode Moving Avg', color='red', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'InvertedDoublePendulum PPO Learning Curve (Episode {episode})')
        plt.legend()
        plt.grid(True)
        
        # 학습 통계 그래프
        if len(self.all_rewards) > 1:
            plt.subplot(2, 1, 2)
            
            # 누적 분포 그래프
            rewards_sorted = np.sort(self.all_rewards)
            p = 1. * np.arange(len(rewards_sorted)) / (len(rewards_sorted) - 1)
            plt.plot(rewards_sorted, p, label='Cumulative Distribution')
            
            # 통계 정보 표시
            plt.axvline(x=np.mean(self.all_rewards), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(self.all_rewards):.1f}')
            plt.axvline(x=np.median(self.all_rewards), color='g', linestyle='--', 
                       label=f'Median: {np.median(self.all_rewards):.1f}')
            
            plt.xlabel('Reward')
            plt.ylabel('Cumulative Probability')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        prefix = "final_" if final else ""
        plot_path = os.path.join(SAVE_DIR, f"{prefix}learning_curve_ep{episode}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()

# =====================
# 메인 실행
# =====================
if __name__ == "__main__":
    # 에이전트 생성 및 학습
    agent = PPOAgent()
    agent.train()

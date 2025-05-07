#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, Bool

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
import time
import os # 모델 저장을 위해 추가

# =====================
# 하이퍼파라미터 (사용자 설정 유지)
# =====================
ENV_NAME       = "InvertedPendulum-v4"  # (참고용)
LEARNING_RATE  = 1e-2
GAMMA          = 0.5
LAMBDA         = 0.95
EPS_CLIP       = 0.3
K_EPOCH        = 4
ROLLOUT_LENGTH = 1024
BATCH_SIZE     = 128
MAX_EPISODES   = 1000

# 모델 저장 경로 설정
SAVE_DIR = "ppo_models_ros2_userbase" # 저장할 디렉토리 이름
FINAL_MODEL_NAME = "ppo_invertedpendulum_ros2_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 저장 디렉토리 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# =====================
# Actor-Critic 네트워크 (사용자 정의 유지)
# =====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh() # 사용자 코드 유지 (출력 평균 -1~1)
        )
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
        value = self.critic(x)
        return (mu, std), value

# =====================
# GAE 계산 (사용자 정의 유지)
# =====================
def compute_gae(rewards, dones, values, next_value):
    advantages = []
    gae = 0
    # dones는 bool 리스트, values는 float 리스트로 가정
    values_with_next = values + [next_value] # 계산 편의를 위해 다음 값 추가
    for i in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[i]) # True -> 0.0, False -> 1.0
        delta = rewards[i] + GAMMA * values_with_next[i+1] * mask - values[i]
        gae = delta + GAMMA * LAMBDA * gae * mask
        advantages.insert(0, gae)
        # next_value = values[i] # 이 부분은 GAE 계산에 필요 없음
    # returns 계산 추가 필요
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

# =====================
# 액션 선택 (사용자 정의 + 범위 수정)
# =====================
# 실제 액션 범위 (InvertedPendulum-v4)
ACTION_LOW_ENV = np.array([-3.0], dtype=np.float32)
ACTION_HIGH_ENV = np.array([3.0], dtype=np.float32)

def select_action(model, state):
    # 상태를 numpy 배열로 변환 (필요 시)
    if not isinstance(state, np.ndarray):
        state = np.array(state, dtype=np.float32)

    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        (mu, std), value = model(state_tensor)
    dist = Normal(mu, std)
    raw_action = dist.sample()
    log_prob = dist.log_prob(raw_action).sum(dim=-1) # 다차원 액션 대비 sum

    # CPU로 이동 및 numpy 변환
    action = raw_action.cpu().numpy().flatten() # shape (1,)

    # 환경의 실제 액션 범위로 클리핑 (중요 수정!)
    action = np.clip(action, ACTION_LOW_ENV, ACTION_HIGH_ENV)

    return action, log_prob.item(), value.item()

# =====================
# ROS2 노드 (사용자 정의 기반 + 수정)
# =====================
class PPOAgentNode(Node):
    def __init__(self):
        super().__init__('ppo_agent')
        self.get_logger().info(f"Initializing PPOAgentNode (User Base) on device: {device}")
        # 구독자
        self.obs_sub = self.create_subscription(Float32MultiArray, '/inverted_pendulum/observation', self.obs_callback, 10)
        self.rew_sub = self.create_subscription(Float32, '/inverted_pendulum/reward', self.reward_callback, 10)
        self.done_sub = self.create_subscription(Bool, '/inverted_pendulum/done', self.done_callback, 10)
        # 퍼블리셔
        self.action_pub = self.create_publisher(Float32MultiArray, '/inverted_pendulum/action', 10)

        # 내부 상태 변수 및 플래그 (동기화 개선용)
        self.current_obs = None
        self.current_reward = None
        self.current_done = None # Bool 또는 None
        self.obs_received = False
        self.reward_received = False
        self.done_received = False

        # 모델·버퍼 준비
        self.state_dim = None
        self.action_dim = 1 # InvertedPendulum-v4 액션 차원 = 1 (중요 수정!)
        self.model = None
        self.optimizer = None
        # 버퍼 (리스트 사용)
        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': []}
        self.reward_history = deque(maxlen=100)
        self.total_steps = 0 # 총 타임스텝 카운터 추가

        self.get_logger().info("Waiting for first observation to initialize model...")

    # --- 콜백 함수들 ---
    def obs_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if self.model is None: # 첫 관측 시 모델 초기화
            self.state_dim = data.shape[0]
            self.get_logger().info(f"First observation received. State dim: {self.state_dim}, Action dim: {self.action_dim}")
            # 모델 생성 (수정된 action_dim 사용)
            self.model = ActorCritic(self.state_dim, self.action_dim).to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
            self.get_logger().info("Model and optimizer initialized.")
        self.current_obs = data
        self.obs_received = True

    def reward_callback(self, msg):
        self.current_reward = msg.data
        self.reward_received = True

    def done_callback(self, msg):
        self.current_done = msg.data # True or False
        self.done_received = True

    # --- 스텝 데이터 대기 헬퍼 (개선된 방식) ---
    def wait_for_step_data(self):
        start_time = time.time()
        timeout_sec = 5.0 # 5초 타임아웃
        while not (self.obs_received and self.reward_received and self.done_received) and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01) # 짧게 spin
            if time.time() - start_time > timeout_sec:
                 self.get_logger().warn("Timeout waiting for step data from environment node.")
                 return False # 실패 반환
        return True

    # --- 학습 루프 (개선된 동기화 적용) ---
    def run(self):
        # 모델 초기화를 위해 첫 관측 대기
        while rclpy.ok() and self.model is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.model is None:
                self.get_logger().info("Still waiting for first observation...")
            time.sleep(0.1)

        if not rclpy.ok(): return # 종료 확인

        self.get_logger().info("Starting training loop...")
        episode = 0

        while episode < MAX_EPISODES and rclpy.ok():
            # 에피소드 시작 시 현재 상태 (초기 상태) 확인
            if self.current_obs is None:
                self.get_logger().warn("Episode start: current observation is None, waiting...")
                self.obs_received = False # 다음 spin_once에서 받을 수 있도록 리셋
                while not self.obs_received and rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.1)
                if not rclpy.ok() or self.current_obs is None:
                    self.get_logger().error("Failed to get initial observation for the episode.")
                    break # 에피소드 진행 불가
            
            state = self.current_obs.copy() # 현재 상태 저장
            ep_reward = 0
            step_in_episode = 0 # 에피소드 내 스텝 카운터

            # 에피소드 루프
            while rclpy.ok():
                # 1) 액션 선택 및 퍼블리시
                action, logp, value = select_action(self.model, state)
                action_msg = Float32MultiArray(data=action.tolist())
                self.action_pub.publish(action_msg)

                # 2) 다음 스텝 데이터 수신 대기를 위한 플래그 리셋
                self.obs_received = False
                self.reward_received = False
                self.done_received = False

                # 3) 환경 노드로부터 다음 상태, 보상, 완료 여부 대기
                if not self.wait_for_step_data():
                    self.get_logger().error("Failed to get step data. Breaking episode.")
                    # 이 경우 에피소드를 강제 종료하거나 다른 처리 필요
                    break # 현재 에피소드 중단

                # 4) 수신 데이터 유효성 확인 및 버퍼 저장
                if self.current_obs is None or self.current_reward is None or self.current_done is None:
                    self.get_logger().warn("Incomplete step data received after wait. Breaking episode.")
                    break # 현재 에피소드 중단

                # 버퍼에 저장 (state는 액션 전 상태, 나머지는 액션 결과)
                self.buffer['states'].append(state)
                self.buffer['actions'].append(action) # shape (1,) 또는 (action_dim,)
                self.buffer['logprobs'].append(logp)
                self.buffer['rewards'].append(self.current_reward)
                self.buffer['dones'].append(self.current_done)
                self.buffer['values'].append(value) # state에 대한 가치

                state = self.current_obs.copy() # 다음 스텝을 위해 상태 업데이트
                ep_reward += self.current_reward
                self.total_steps += 1
                step_in_episode += 1

                # 5) Rollout 길이 도달 시 정책 업데이트
                # (total_steps 사용 또는 len(buffer['states']) 사용 가능)
                if len(self.buffer['states']) >= ROLLOUT_LENGTH:
                    # 다음 상태 가치 계산 (현재 state 사용)
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                        _, next_val_tensor = self.model(state_tensor)
                        # 마지막 스텝이 done이면 next_value는 0
                        next_value = next_val_tensor.item() if not self.current_done else 0.0

                    # GAE 및 Returns 계산
                    advantages, returns = compute_gae(
                        self.buffer['rewards'], self.buffer['dones'],
                        self.buffer['values'], next_value
                    )

                    # 텐서 변환
                    states_tensor = torch.FloatTensor(np.array(self.buffer['states'])).to(device)
                    # actions 버퍼 내용 확인 및 텐서 변환
                    try:
                        actions_np = np.array(self.buffer['actions'])
                        # InvertedPendulum은 액션이 1차원이므로 (N,) 또는 (N, 1) 형태
                        if actions_np.ndim == 1:
                             actions_np = actions_np[:, np.newaxis] # (N, 1) 형태로 변환
                        actions_tensor = torch.FloatTensor(actions_np).to(device)
                    except Exception as e:
                        self.get_logger().error(f"Error converting actions to tensor: {e}")
                        self.get_logger().error(f"Actions buffer content: {self.buffer['actions']}")
                        # 버퍼 클리어하고 다음 롤아웃 진행
                        for k in self.buffer: self.buffer[k].clear()
                        continue # 다음 스텝으로

                    oldlogp_tensor = torch.FloatTensor(self.buffer['logprobs']).to(device)
                    returns_tensor = torch.FloatTensor(returns).to(device)
                    advs_tensor = torch.FloatTensor(advantages).to(device)

                    # Advantage 정규화
                    advs_tensor = (advs_tensor - advs_tensor.mean()) / (advs_tensor.std() + 1e-8)

                    # PPO 업데이트 (사용자 코드 기반)
                    dataset_size = states_tensor.size(0)
                    for _ in range(K_EPOCH):
                        # 인덱스 셔플링 (선택 사항)
                        perm = torch.randperm(dataset_size)
                        for start in range(0, dataset_size, BATCH_SIZE):
                            end = start + BATCH_SIZE
                            mb_idx = perm[start:end] # 셔플된 인덱스 사용

                            (mu, std), vals = self.model(states_tensor[mb_idx])
                            dist = Normal(mu, std)
                            newlogp = dist.log_prob(actions_tensor[mb_idx]).sum(dim=-1)
                            entropy = dist.entropy().mean() # 평균 엔트로피

                            ratio = torch.exp(newlogp - oldlogp_tensor[mb_idx])
                            surr1 = ratio * advs_tensor[mb_idx]
                            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advs_tensor[mb_idx]
                            actor_loss = -torch.min(surr1, surr2).mean()
                            critic_loss = nn.MSELoss()(vals.flatten(), returns_tensor[mb_idx])
                            # Entropy 계수 사용자 코드 확인 (0.01 사용?)
                            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                            self.optimizer.zero_grad()
                            loss.backward()
                            # 그래디언트 클리핑 (선택 사항, 안정성 도움)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                            self.optimizer.step()

                    # 버퍼 비우기
                    for k in self.buffer: self.buffer[k].clear()

                # 6) 에피소드 종료 조건 확인
                if self.current_done:
                    episode += 1
                    self.reward_history.append(ep_reward)
                    avg_reward = np.mean(self.reward_history)
                    self.get_logger().info(
                        f"Episode {episode} finished | Steps: {step_in_episode} | Reward: {ep_reward:.1f} | Avg(100): {avg_reward:.1f}"
                    )
                    # 에피소드 내부 루프 종료
                    break

        # 학습 완료 후 최종 모델 저장
        if self.model is not None:
            final_model_path = os.path.join(SAVE_DIR, FINAL_MODEL_NAME)
            try:
                torch.save(self.model.state_dict(), final_model_path)
                self.get_logger().info(f"Final model saved: {final_model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to save final model: {e}")
        else:
             self.get_logger().warn("Model was not initialized, nothing to save.")

        self.get_logger().info("Training finished or interrupted.")


def main(args=None):
    rclpy.init(args=args)
    agent_node = PPOAgentNode()
    try:
        agent_node.run()
    except KeyboardInterrupt:
        agent_node.get_logger().info("Keyboard interrupt received, shutting down.")
        # 인터럽트 시 모델 저장 시도 (이미 run 루프에서 처리됨)
    except Exception as e:
        agent_node.get_logger().error(f"Unhandled exception in agent run loop: {e}", exc_info=True)
    finally:
        # 노드 정리 및 ROS 종료
        if agent_node and rclpy.ok(): # 노드가 성공적으로 생성되었는지 확인
             agent_node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()

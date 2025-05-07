import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

# =====================
# 하이퍼파라미터
# =====================

ENV_NAME       = "Pendulum-v1"                
LEARNING_RATE  = 1e-3                         # TD3에 최적화된 학습률
GAMMA          = 0.99                         # 할인율
BATCH_SIZE     = 256                          # 배치 크기
BUFFER_SIZE    = 1000000                      # 리플레이 버퍼 크기
TAU            = 0.005                        # 타겟 네트워크 소프트 업데이트 비율
POLICY_NOISE   = 0.2                          # 타겟 정책 스무딩 노이즈
NOISE_CLIP     = 0.5                          # 노이즈 클리핑 범위
POLICY_DELAY   = 2                            # 정책 업데이트 지연 스텝
EXPLORATION_NOISE = 0.1                       # 탐색 노이즈
HIDDEN_SIZE    = 256                          # 은닉층 크기
MAX_EPISODES   = 1000                         # 최대 에피소드 수
MAX_STEPS      = 500                          # 에피소드당 최대 스텝 수
WARMUP_STEPS   = 10000                        # 학습 전 랜덤 액션으로 데이터 수집 스텝

SAVE_DIR         = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_NAME = "td3_pendulum-v1_final.pth"
LOG_DIR          = os.path.join(SAVE_DIR, "runs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard 설정
writer = SummaryWriter(log_dir=LOG_DIR)

# =====================
# 리플레이 버퍼
# =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward.reshape(-1, 1), next_state, done.reshape(-1, 1)
    
    def __len__(self):
        return len(self.buffer)

# =====================
# 액터 네트워크
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
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state):
        return self.max_action * self.net(state)

# =====================
# 크리틱 네트워크
# =====================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        
        # Q1 네트워크
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Q2 네트워크
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x)

# =====================
# TD3 에이전트
# =====================
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action, HIDDEN_SIZE).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, HIDDEN_SIZE).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        
        self.critic = Critic(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.critic_target = Critic(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
        self.max_action = max_action
        self.total_it = 0
        
        # 타겟 네트워크 초기화
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def select_action(self, state, noise=0.0):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor(state).cpu().numpy().flatten()
        
        # 탐색 노이즈 추가
        if noise > 0:
            noise = np.random.normal(0, noise * self.max_action, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action
    
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        
        # 리플레이 버퍼에서 배치 샘플링
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        
        with torch.no_grad():
            # 타겟 정책 스무딩을 위한 노이즈 선택
            noise = torch.FloatTensor(action.shape).data.normal_(0, POLICY_NOISE).to(device)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            
            # 타겟 액션 계산
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # 타겟 Q 값 계산
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * GAMMA * target_q
        
        # 현재 Q 값 계산
        current_q1, current_q2 = self.critic(state, action)
        
        # 크리틱 손실 계산
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # 크리틱 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 지연된 정책 업데이트
        actor_loss_value = 0.0  # 기본값 설정
        
        if self.total_it % POLICY_DELAY == 0:
            # 액터 손실 계산
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            actor_loss_value = actor_loss.item()  # 텐서에서 스칼라 값 추출
            
            # 액터 업데이트
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 타겟 네트워크 소프트 업데이트
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return critic_loss.item(), actor_loss_value

    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

# =====================
# 평가 함수
# =====================
def evaluate_policy(agent, env, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, noise=0)  # 평가 시 노이즈 없음
            step_result = env.step(action)
            if len(step_result) == 5:  # 새로운 gym 버전
                next_state, reward, term, trunc, _ = step_result
                done = term or trunc
            else:  # 이전 gym 버전
                next_state, reward, done, _ = step_result
            
            avg_reward += reward
            state = next_state
            
    avg_reward /= eval_episodes
    return avg_reward

# =====================
# 메인 학습 루프
# =====================
def main():
    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 모델 그래프를 TensorBoard에 기록
    agent = TD3(state_dim, action_dim, max_action)
    
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    # 학습 통계
    reward_history = deque(maxlen=100)
    best_reward = -1000
    global_step = 0
    
    # 초기 랜덤 데이터 수집 (워밍업)
    state, _ = env.reset()
    for i in range(WARMUP_STEPS):
        action = env.action_space.sample()
        step_result = env.step(action)
        
        if len(step_result) == 5:  # 새로운 gym 버전
            next_state, reward, term, trunc, _ = step_result
            done = term or trunc
        else:  # 이전 gym 버전
            next_state, reward, done, _ = step_result
        
        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset()[0]
        
        if (i + 1) % 1000 == 0:
            print(f"워밍업: {i+1}/{WARMUP_STEPS} 스텝 완료")
    
    # 메인 학습 루프
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        episode_steps = 0
        
        for step in range(MAX_STEPS):
            global_step += 1
            
            # 액션 선택 및 실행
            action = agent.select_action(state, EXPLORATION_NOISE)
            step_result = env.step(action)
            
            if len(step_result) == 5:  # 새로운 gym 버전
                next_state, reward, term, trunc, _ = step_result
                done = term or trunc
            else:  # 이전 gym 버전
                next_state, reward, done, _ = step_result
            
            # 리플레이 버퍼에 저장
            replay_buffer.add(state, action, reward, next_state, float(done))
            
            # 에이전트 학습
            if len(replay_buffer) > BATCH_SIZE:
                critic_loss, actor_loss = agent.train(replay_buffer, BATCH_SIZE)
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # TensorBoard에 스텝별 정보 기록
            writer.add_scalar("Training/StepReward", reward, global_step)
            
            if done or step == MAX_STEPS - 1:
                break
        
        # 에피소드 통계 계산
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history)
        avg_critic_loss = episode_critic_loss / (episode_steps or 1)
        avg_actor_loss = episode_actor_loss / (episode_steps or 1)
        
        # TensorBoard에 에피소드 정보 기록
        writer.add_scalar("Episode/Reward", episode_reward, episode)
        writer.add_scalar("Episode/AvgReward100", avg_reward, episode)
        writer.add_scalar("Episode/Steps", episode_steps, episode)
        writer.add_scalar("Loss/Critic", avg_critic_loss, episode)
        writer.add_scalar("Loss/Actor", avg_actor_loss, episode)
        
        # 모델 파라미터 히스토그램 기록 (10 에피소드마다)
        if episode % 10 == 0:
            for name, param in agent.actor.named_parameters():
                writer.add_histogram(f"Actor/{name}", param.data, episode)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(f"Critic/{name}", param.data, episode)
        
        print(f"Episode {episode}: Reward={episode_reward:.1f}, AvgReward100={avg_reward:.1f}, Steps={episode_steps}")
        
        # 주기적인 정책 평가 (10 에피소드마다)
        if episode % 10 == 0:
            eval_reward = evaluate_policy(agent, eval_env)
            writer.add_scalar("Evaluation/Reward", eval_reward, episode)
            print(f"Evaluation: {eval_reward:.1f}")
            
            # 최고 성능 모델 저장
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = os.path.join(SAVE_DIR, "td3_pendulum_best.pth")
                agent.save(best_path)
                print(f"새로운 최고 점수! 모델 저장: {best_path}")
        
        # 주기적인 모델 저장 (100 에피소드마다)
        if episode % 100 == 0:
            save_path = os.path.join(SAVE_DIR, f"td3_pendulum_ep{episode}.pth")
            agent.save(save_path)
            print(f"모델 저장: {save_path}")
    
    # 최종 모델 저장
    final_path = os.path.join(SAVE_DIR, FINAL_MODEL_NAME)
    agent.save(final_path)
    print(f"최종 모델 저장 완료: {final_path}")
    
    # TensorBoard 종료
    writer.close()
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

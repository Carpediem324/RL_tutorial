#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QRSAC-Lagrangian 알고리즘 구현 + TensorBoard 동시 실행
• Quantile Regression SAC + Lagrange multiplier 업데이트
• Critic target에 reward 포함
• 매 10,000 스텝마다 체크포인트 저장 (DATE, TRIAL 변수 활용)
• TensorBoard 실시간 갱신(flush_secs=1)
"""
import os
import sys
import random
import torch
import numpy as np
from collections import deque, namedtuple
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

from custom.jethexa_noreward import JethexaEnv

# ─────────────────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_ACTOR       = 3e-4
LR_CRITIC      = 3e-4
LR_LAMBDA      = 5e-3
GAMMA          = 0.98
TAU            = 0.005
N_QUANTILES    = 32
BATCH_SIZE     = 256
REPLAY_SIZE    = int(1e6)
POLICY_FREQ    = 2
MULTI_UPDATE_D = 1000
MAX_ITERS      = 1_000_000
INIT_LAMBDAS   = [0.1, 0.1]
# ─────────────────────────────────────────────────────────────────────────────

# 사용자 설정 (TRAIN 모드일 때만 사용)
DATE  = "250505"
TRIAL = "D"

# ── 공용 클래스 / 함수 정의 ──────────────────────────────────────────────────

# Transition 및 ReplayBuffer
Transition = namedtuple("Transition",
    ("state","action","reward","constraints","next_state","done"))

class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

# Actor / Critic 네트워크 정의
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
        )
        self.mean    = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
    def forward(self, x):
        h       = self.net(x)
        mu      = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        std     = log_std.exp()
        return mu, std

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_quantiles):
        super().__init__()
        in_dim = obs_dim + act_dim
        self.quantiles = nn.Sequential(
            nn.Linear(in_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256, n_quantiles)
        )
    def forward(self, s, a):
        x = torch.cat([s,a], dim=-1)
        return self.quantiles(x)

class QRSACAgent:
    def __init__(self, env):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.act_lim = env.action_space.high[0]

        # 네트워크 + 타겟 네트워크
        self.actor   = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic1 = Critic(obs_dim, act_dim, N_QUANTILES).to(DEVICE)
        self.critic2 = Critic(obs_dim, act_dim, N_QUANTILES).to(DEVICE)
        self.target1 = Critic(obs_dim, act_dim, N_QUANTILES).to(DEVICE)
        self.target2 = Critic(obs_dim, act_dim, N_QUANTILES).to(DEVICE)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        # 옵티마이저
        self.opt_actor  = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=LR_CRITIC
        )
        self.lambdas    = torch.tensor(INIT_LAMBDAS, device=DEVICE, requires_grad=True)
        self.opt_lambda = optim.Adam([self.lambdas], lr=LR_LAMBDA)

        self.buffer    = ReplayBuffer(REPLAY_SIZE)
        self.total_it  = 0

    def select_action(self, state):
        s = torch.FloatTensor(state).to(DEVICE)
        mu, std = self.actor(s)
        z = torch.randn_like(mu)
        a = torch.tanh(mu + std*z) * self.act_lim
        return a.detach().cpu().numpy()

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None, None, self.lambdas.detach().cpu().numpy()

        self.total_it += 1
        trans = self.buffer.sample(BATCH_SIZE)

        # numpy → tensor
        s_arr, a_arr, r_arr, g_arr, s2_arr, d_arr = map(
            lambda x: np.array(x, dtype=np.float32),
            (trans.state, trans.action, trans.reward,
             trans.constraints, trans.next_state, trans.done)
        )
        r_arr = r_arr.reshape(-1,1)
        d_arr = d_arr.reshape(-1,1)

        s  = torch.from_numpy(s_arr).to(DEVICE)
        a  = torch.from_numpy(a_arr).to(DEVICE)
        r  = torch.from_numpy(r_arr).to(DEVICE)
        g  = torch.from_numpy(g_arr).to(DEVICE)
        s2 = torch.from_numpy(s2_arr).to(DEVICE)
        d  = torch.from_numpy(d_arr).to(DEVICE)

        # 1) Critic 타겟에 reward 포함
        with torch.no_grad():
            mu2, std2 = self.actor(s2)
            z2 = torch.randn_like(mu2)
            a2 = torch.tanh(mu2 + std2*z2) * self.act_lim
            q1_t = self.target1(s2, a2)
            q2_t = self.target2(s2, a2)
            q_t  = torch.min(q1_t, q2_t)
            target_q = r + GAMMA * (1 - d) * q_t

        # 2) Critic 업데이트 (quantile Huber loss)
        loss_critic = 0.0
        for critic in (self.critic1, self.critic2):
            q = critic(s, a)
            td = target_q.unsqueeze(-1) - q.unsqueeze(1)
            huber = torch.where(td.abs() <= 1, 0.5*td.pow(2), td.abs() - 0.5)
            pi = torch.arange(N_QUANTILES, device=DEVICE).float() / N_QUANTILES \
                 + 1/(2*N_QUANTILES)
            loss_q = (torch.abs(pi - (td < 0).float()) * huber).mean()
            self.opt_critic.zero_grad()
            loss_q.backward()
            self.opt_critic.step()
            loss_critic += loss_q.item()
        loss_critic /= 2

        # 3) Actor & λ 업데이트
        loss_actor = 0.0
        if self.total_it % POLICY_FREQ == 0:
            mu, std = self.actor(s)
            z = torch.randn_like(mu)
            a_pred = torch.tanh(mu + std*z) * self.act_lim
            q1_pi = self.critic1(s, a_pred)
            q2_pi = self.critic2(s, a_pred)
            q_pi   = torch.min(q1_pi, q2_pi).mean(dim=1, keepdim=True)
            lam_term = (self.lambdas * g.mean(dim=0)).sum().unsqueeze(0)
            loss_a = (-q_pi + lam_term).mean()
            self.opt_actor.zero_grad()
            loss_a.backward()
            self.opt_actor.step()
            loss_actor = loss_a.item()

        # 4) Lagrange multiplier 업데이트
        if self.total_it % MULTI_UPDATE_D == 0:
            mean_g = g.mean(dim=0)
            lam_loss = -(self.lambdas * mean_g).sum()
            self.opt_lambda.zero_grad()
            lam_loss.backward()
            self.opt_lambda.step()
            with torch.no_grad():
                self.lambdas.clamp_(min=0.0)

        # 5) 타겟 네트워크 soft update
        for tgt, src in ((self.target1, self.critic1),(self.target2, self.critic2)):
            for p_t, p in zip(tgt.parameters(), src.parameters()):
                p_t.data.mul_(1-TAU)
                p_t.data.add_(TAU * p.data)

        return loss_critic, loss_actor, self.lambdas.detach().cpu().numpy()

    def save(self, path):
        torch.save({
            'actor':   self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'lambdas': self.lambdas.detach().cpu()
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(data['actor'])
        self.critic1.load_state_dict(data['critic1'])
        self.critic2.load_state_dict(data['critic2'])
        lam = data['lambdas'].to(DEVICE)
        self.lambdas = torch.tensor(lam, device=DEVICE, requires_grad=True)
        self.opt_lambda = optim.Adam([self.lambdas], lr=LR_LAMBDA)

# ── TRAINING 스크립트 ────────────────────────────────────────────────────────
def train():
    # 로그 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir   = os.path.join(script_dir, f"save_{DATE}_{TRIAL}")
    tb_logdir  = os.path.join(script_dir, "runs", f"{DATE}_{TRIAL}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_logdir, exist_ok=True)

    # TensorBoard 서버 실행
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tb_logdir, "--port", "6006"])
    url = tb.launch()
    print(f"TensorBoard started at {url}")

    # 환경·에이전트·SummaryWriter
    env    = JethexaEnv()
    agent  = QRSACAgent(env)
    writer = SummaryWriter(tb_logdir, flush_secs=1)

    state, _    = env.reset()
    total_steps = 0

    while total_steps < MAX_ITERS:
        action = agent.select_action(state)
        next_s, _, done, _, info = env.step(action)
        reward = info['forward_vel']
        agent.buffer.push(state, action, reward,
                          info['constraints'], next_s, float(done))
        state = next_s
        total_steps += 1

        loss_c, loss_a, lambdas = agent.train_step()

        # TensorBoard 로깅
        if loss_c is not None:
            writer.add_scalar("Loss/Critic",       loss_c,      total_steps)
            writer.add_scalar("Loss/Actor",        loss_a,      total_steps)
            writer.add_scalar("Lambda/0",          lambdas[0],  total_steps)
            writer.add_scalar("Lambda/1",          lambdas[1],  total_steps)
            writer.add_scalar("Constraint/lateral", info['constraints'][0], total_steps)
            writer.add_scalar("Constraint/yaw",     info['constraints'][1], total_steps)
            writer.add_scalar("Reward/forward_vel", reward,      total_steps)

        if done:
            state, _ = env.reset()

        if total_steps % 10000 == 0:
            print(f"Step {total_steps}, λ = {lambdas}")
            ckpt = os.path.join(save_dir, f"{TRIAL}_ckpt_{total_steps}_{DATE}.pth")
            agent.save(ckpt)
            print(f"[Checkpoint] saved → {ckpt}")

    # 최종 모델 저장
    final = os.path.join(save_dir, f"{TRIAL}_final_{DATE}.pth")
    agent.save(final)
    print(f"[Final] saved → {final}")

    writer.close()
    env.close()

if __name__ == "__main__":
    train()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jethexa PPO: GPU 사용률 극대화 (compile 제거 버전)
────────────────────────────────────────────────────────────────────────────
• SubprocVecEnv 6개(spawn)
• 거대 MLP 2048‑2048‑1024
• rollout 8192, batch 16384, grad_accum 4
• TF32 활성화 (torch.set_float32_matmul_precision)
"""

import os, sys, warnings
from typing import Callable

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# ─── 경로 및 환경 register ───────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import __init__          # JethexaEnv register (없으면 무시)

# ─── 라이브러리 ───────────────────────────────────────────────────────────
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
import torch

# ── CPU 스레드 제한 ──────────────────────────────────────────────────────
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ── TF32 활성화 ──────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')

# ── 선형 스케줄 함수 ─────────────────────────────────────────────────────
def lin_sched(a: float, b: float) -> Callable[[float], float]:
    return lambda p: p * (a - b) + b

# ── 메인 함수 ────────────────────────────────────────────────────────────
def main() -> None:
    date, trial = "250503", "A_gpu_big_nocompile"

    # 체크포인트
    ckpt_dir = os.path.join(script_dir, f"save_model_{date}", trial)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix=f"{trial}_model_{date}",
        verbose=1,
    )

    # 벡터 환경
    env_id, n_envs = "Jethexa_shh", 6
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="spawn"),
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    print("Observation space:", env.observation_space)

    # 대형 네트워크
    policy_kwargs = dict(
        net_arch=dict(pi=[2048, 2048, 1024], vf=[2048, 2048, 1024]),
        activation_fn=torch.nn.SiLU,
    )

    # PPO 모델 (compile 없음!)
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,
        tensorboard_log=os.path.join(script_dir, f"{trial}_tb_log_{date}"),
        learning_rate=lin_sched(3e-4, 1e-6),
        clip_range=lin_sched(0.3, 0.05),
        n_steps=8192,
        batch_size=16384,
        n_epochs=40,
        ent_coef=1e-4,
        policy_kwargs=policy_kwargs,
    )

    # ── 학습 ──────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=50_000_000,
        callback=checkpoint_cb,
        tb_log_name=f"{trial}_tb_{date}",
    )

    # ── 저장 (compile X → 로드시 호환 OK) ─────────────────────────────────
    env.save(os.path.join(script_dir, f"{trial}_vecnormalize.pkl"))
    model.save(os.path.join(script_dir, f"{trial}_final"))
    print("Model saved without torch.compile()")

# ── 진입점 (멀티프로세스 안전) ────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

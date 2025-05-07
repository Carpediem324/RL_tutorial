# 파일명: Jet_train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch

# custom env 등록
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from custom.jethexa_shh import JethexaEnv

# 선형 스케줄러 헬퍼
def lin_schedule(init, final):
    return lambda p: p * (init - final) + final

DATE, TRIAL, N_ENVS = "250505", "LAST", 4

# 체크포인트 콜백 (50k 스텝마다)
ckpt_dir = os.path.join(script_dir, f"save_{DATE}_{TRIAL}")
os.makedirs(ckpt_dir, exist_ok=True)
checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=ckpt_dir,
    name_prefix=f"{TRIAL}_model_{DATE}",
    verbose=1
)

# 벡터 환경 + reward 정규화
env = make_vec_env(JethexaEnv, n_envs=N_ENVS)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# 정책 네트워크 설정
policy_kwargs = dict(
    net_arch=[dict(pi=[256,256,128], vf=[256,256,128])],
    activation_fn=torch.nn.Tanh,
)

model = PPO(
    "MlpPolicy",
    env=env,
    verbose=1,
    device="cpu",
    tensorboard_log=os.path.join(script_dir, f"{TRIAL}_tb_{DATE}"),
    learning_rate=lin_schedule(3e-4, 3e-6),
    clip_range=lin_schedule(0.3, 0.1),
    n_steps=256,
    batch_size=256 * N_ENVS,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=1e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
)

# 학습 시작 (TimeLimit은 Gym이 truncated 처리)
model.learn(
    total_timesteps=50_000_000,
    callback=checkpoint_cb,
    tb_log_name=f"{TRIAL}_{DATE}"
)

# VecNormalize 저장
vn_path = os.path.join(script_dir, f"{TRIAL}_vecnorm.pkl")
env.save(vn_path)

# 최종 모델 저장
final_path = os.path.join(script_dir, f"{TRIAL}_final")
model.save(final_path)

env.close()

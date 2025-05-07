import sys
import os

# 현재 디렉토리를 PYTHONPATH에 추가 (custom, __init__.py가 있는 곳)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import __init__  # 환경 등록 코드 실행 (파일명이 __init__.py라면)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from typing import Callable

# 선형 학습률 스케줄러 정의
def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

# 사용자 설정(date, trial)
date  = "250430"
trial = "F"

# 체크포인트 콜백: 매 100,000 스텝마다 저장
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=os.path.join(script_dir, f"save_model_{date}", trial),
    verbose=1,
    name_prefix=f"{trial}_model_{date}"
)

# 벡터 환경 생성 및 정규화
env_id = "Hexy-v4"
env = make_vec_env(env_id, n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
print(f"Observation space shape: {env.observation_space.shape}")

# PPO 모델 설정
model = PPO(
    "MlpPolicy",
    env=env,
    verbose=1,
    device="cuda",  # GPU 사용 시 "cuda", CPU 사용 시 "cpu"
    tensorboard_log=os.path.join(script_dir, f"{trial}_tb_log_{date}"),
    learning_rate=lin_schedule(3e-4, 3e-6),
    clip_range=lin_schedule(0.3, 0.1),
    n_epochs=10,
    ent_coef=1e-4,
    batch_size=256 * 4,
    n_steps=256
)

# 학습 실행(체크포인트 하나만 사용)
model.learn(
    total_timesteps=50_000_000,
    callback=checkpoint_callback,
    tb_log_name=f"{trial}_tb_{date}{trial}"
)

# 환경 정규화 값 저장
env.save(os.path.join(script_dir, f"{trial}_vecnormalize.pkl"))

# 모델 저장 및 로드
model_path = os.path.join(script_dir, f"{trial}_first")
model.save(model_path)
del model
model = PPO.load(model_path, device="cpu")

# 테스트용 환경(단일, 렌더링)
test_env = gym.make(env_id, render_mode="human")
# 동일한 정규화 적용
test_env = VecNormalize.load(os.path.join(script_dir, f"{trial}_vecnormalize.pkl"), test_env)
obs, _ = test_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    test_env.render()

# 정리
env.close()
test_env.close()

import sys
import os

# 현재 디렉토리를 PYTHONPATH에 추가 (custom, __init__.py가 있는 곳)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import __init__  # 환경 등록 코드 실행 (파일명이 __init__.py라면)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable

def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

date = "250429"
trial = "hexy_v2"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path='./save_model_' + date + '/' + trial,
    verbose=2,
    name_prefix='cheetah_model_' + date + trial
)
event_callback = EveryNTimesteps(
    n_steps=int(1e5),
    callback=checkpoint_on_event
)

# 환경 이름           설명
# HalfCheetah-v5      # 반치타
# Hopper-v5           # 호퍼(한발 점퍼)
# Walker2d-v5         # 2족 보행 로봇
# Ant-v5              # 4족 로봇
# Humanoid-v5         # 인간형 로봇
# HumanoidStandup-v5  # 인간형 일어서기
# InvertedPendulum-v5 # 도립 진자
# InvertedDoublePendulum-v5 # 2중 도립 진자
# Reacher-v5          # 2-DOF 암
# Swimmer-v5          # 뱀형 로봇
# Pusher-v5           # 밀기 로봇
# Throwe
# r-v5          # 던지기 로봇
# Striker-v5          # 스트라이커 로봇
env_id = "Hexy-v4"  # 최신 mujoco 환경명

# Gymnasium + mujoco는 render_mode 지정 필요 없음(벡터 환경에선 render 불가)
env = make_vec_env(env_id, n_envs=4)

model = PPO(
    "MlpPolicy",
    env=env,
    verbose=2,
    tensorboard_log='./cheetah_tb_log_' + date,
    learning_rate=lin_schedule(3e-4, 3e-6),
    clip_range=lin_schedule(0.3, 0.1),
    n_epochs=10,
    ent_coef=1e-4,
    batch_size=256*4,
    n_steps=256
)

model.learn(
    total_timesteps=50_000_000,
    callback=event_callback,
    tb_log_name='cheetah_tb_' + date + trial
)

model.save("cheetah_first")
del model
model = PPO.load("cheetah_first")

# 벡터 환경에서는 render 지원 안됨, 단일 환경으로 시각화
test_env = gym.make(env_id, render_mode="human")
obs, _ = test_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    test_env.render()

env.close()
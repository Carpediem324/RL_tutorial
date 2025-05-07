#!/usr/bin/env python3
import sys
import os
import time

# 스크립트 위치를 기준으로 custom 환경 등록
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import __init__  # Hexy-v4 환경을 register_env 해 줌

import gymnasium as gym
from custom.hexy_v4 import HexyEnv
from stable_baselines3 import PPO

# ─────────────────────────────────────────────────────────────────────────────
# 사용자 설정
DATE   = "250430"
TRIAL  = "D"
STEPS  = "8400000"   # 불러올 스텝 수를 문자열로 지정
# ─────────────────────────────────────────────────────────────────────────────

# 1) 커스텀 환경 생성 (렌더링만, ROS2 비사용)
env = HexyEnv(render_mode="human")

# 2) 학습된 모델 로드 (.zip 확장자 포함)
save_dir   = os.path.join(script_dir, f"save_model_{DATE}", TRIAL)
# 파일명 패턴: {trial}_model_{date}{trial}_{steps}_steps.zip
model_file = os.path.join(
    save_dir,
    f"{TRIAL}_model_{DATE}_{STEPS}_steps"
)
print(f"Loading model from: {model_file}")
model = PPO.load(model_file, device="cuda")  # 필요 시 device="cpu"

# 3) 환경 초기화
obs, info = env.reset()

try:
    while True:
        # 4) 액션 예측 및 5) 실행
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # 6) 에피소드 종료 시 리셋
        if terminated or truncated:
            obs, info = env.reset()

        # 7) 속도 제어
        time.sleep(0.02)

except KeyboardInterrupt:
    pass

# 8) 환경 종료
env.close()

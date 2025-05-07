#!/usr/bin/env python3
import os
import sys
import time
from stable_baselines3 import PPO
from custom.jethexa_shh import JethexaEnv

# 현재 스크립트 폴더를 PYTHONPATH에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# ─────────────────────────────────────────────────────────────────────────────
# 불러올 모델 설정 (train_jethexa.py 에 맞춘 이름)
DATE   = "250505"
TRIAL  = "LAST"      # train_jethexa.py 에서 사용한 TRIAL 값
STEPS  = "600000"        # 체크포인트 스텝 (예: 800000)
# ─────────────────────────────────────────────────────────────────────────────

# 체크포인트 디렉토리와 파일명 구성
save_dir   = os.path.join(script_dir, f"save_{DATE}_{TRIAL}")
model_file = os.path.join(
    save_dir,
    f"{TRIAL}_model_{DATE}_{STEPS}_steps.zip"  # CheckpointCallback 기본 네이밍
)
print(f"Loading model from: {model_file}")

# 모델 로드
model = PPO.load(model_file, device="cpu")

# 환경 생성 (렌더링 모드)
env = JethexaEnv(render_mode="human")
obs, _ = env.reset()

try:
    while True:
        # 예측 → 실행 → 렌더
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        # 에피소드 종료 시 리셋
        if done or truncated:
            obs, _ = env.reset()

        # 50Hz 재생 속도
        time.sleep(0.02)

except KeyboardInterrupt:
    # Ctrl+C 로 안전 종료
    pass

env.close()

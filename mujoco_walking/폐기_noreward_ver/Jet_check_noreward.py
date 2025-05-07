# Jet_check_noreward.py
#!/usr/bin/env python3
import sys, os, time
import torch

# 스크립트 위치를 기준으로 custom 환경·에이전트 등록
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from custom.jethexa_noreward import JethexaEnv
from Jet_train_noreward import QRSACAgent

# 사용자 설정
DATE   = "250505"
TRIAL  = "D"
STEPS  = "20000"  # 체크포인트 스텝

# 모델 경로
save_dir   = os.path.join(script_dir, f"save_{DATE}_{TRIAL}")
model_file = os.path.join(save_dir, f"{TRIAL}_ckpt_{STEPS}_{DATE}.pth")
print(f"Loading checkpoint from: {model_file}")

# env & agent 초기화
env   = JethexaEnv(render_mode="human")
agent = QRSACAgent(env)
agent.load(model_file)

# 실행
obs, _ = env.reset()
try:
    while True:
        action = agent.select_action(obs)
        obs, _, done, truncated, _ = env.step(action)
        env.render()
        if done or truncated:
            obs, _ = env.reset()
        time.sleep(0.02)  # 50Hz 재생
except KeyboardInterrupt:
    pass

env.close()

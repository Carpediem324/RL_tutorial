#!/usr/bin/env python3
import sys, os, time
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import __init__

from custom.jethexa_shh import JethexaEnv
from stable_baselines3 import PPO

DATE, TRIAL, STEPS = "250503", "A_gpu_big", "300000"
env = JethexaEnv(render_mode="human")

model_path = os.path.join(
    script_dir, f"save_model_{DATE}", TRIAL,
    f"{TRIAL}_model_{DATE}_{STEPS}_steps"
)
print("Loading:", model_path)
model = PPO.load(model_path, device="cuda")   # ← compile 안 된 새 zip 로드!

# 필요 시 추론 속도 올리고 싶으면 여기서 compile
# model.policy = torch.compile(model.policy)

obs, _ = env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        env.render()
        if term or trunc:
            obs, _ = env.reset()
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
env.close()

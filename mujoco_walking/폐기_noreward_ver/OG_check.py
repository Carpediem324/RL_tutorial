import gymnasium as gym
from custom.hexy_v4 import HexyEnv
from stable_baselines3 import PPO
import time

date  = "250429"
trial = "hexy_v2"
steps = "1000000"

# 1) 커스텀 환경을 human 렌더 모드로 생성
env = HexyEnv(render_mode="human")

# 2) 학습된 모델 로드 (.zip 확장자 포함)
save_path  = f'./save_model_{date}/{trial}/'
model_file = save_path + f"cheetah_model_{date}{trial}_{steps}_steps.zip"
model      = PPO.load(model_file, device="cuda")  # or device="cpu"

# 3) 환경 초기화
obs, info = env.reset()

try:
    while True:
        # 4) 정책으로 액션 예측
        action, _states = model.predict(obs, deterministic=True)

        # 5) 스텝 & 렌더
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # 6) 종료되면 환경 리셋
        if terminated or truncated:
            obs, info = env.reset()

        # 7) 너무 빠르게 돌지 않게 약간 슬립 (옵션)
        time.sleep(0.02)

except KeyboardInterrupt:
    # Ctrl+C 누르면 안전하게 종료
    pass

# 8) 창 닫기
env.close()

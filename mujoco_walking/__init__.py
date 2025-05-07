from gymnasium.envs.registration import register

# Mujoco
# ----------------------------------------

register(
    id='Hexy-v4',
    entry_point='custom.hexy_v4:HexyEnv',  # custom 폴더 내 hexy_v4.py의 HexyEnv
    max_episode_steps=3000,
)

register(
    id='Jethexa_shh',
    entry_point='custom.jethexa_shh:JethexaEnv',  # custom 폴더 내 hexy_v4.py의 HexyEnv
    max_episode_steps=3000,
)

register(
    id='Jethexa_noreward',
    entry_point='custom.jethexa_shh:JethexaEnv',  # custom 폴더 내 hexy_v4.py의 HexyEnv
    max_episode_steps=3000,
)
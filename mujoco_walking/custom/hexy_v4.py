#hexy_v4.py
import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces, utils

DEFAULT_CAMERA_CONFIG = {'distance': 1.5}

class HexyEnv(MujocoEnv, utils.EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        render_mode: str = None,
        **kwargs
    ):
        # EzPickle 초기화
        utils.EzPickle.__init__(self, xml_file, frame_skip, default_camera_config, render_mode, **kwargs)

        # XML 파일 절대경로 결정
        if xml_file is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            xml_file = os.path.join(module_dir, 'assets', 'Hexy_ver_2.2', 'hexy-v2.2.xml')

        # 부모 클래스 초기화 (model, self.data 생성)
        super().__init__(
            model_path=xml_file,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs
        )

        # 버퍼 크기 설정
        self.obs_dim = 18            # joint positions (6 legs × 3 joints)
        self.act_dim = self.model.nu  # actuator 수 (18)

        # 관측 및 행동 공간 정의
        buf_size = self.obs_dim * 3 + self.act_dim * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(buf_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.act_dim,), dtype=np.float32
        )

        # 히스토리 버퍼 초기화
        self._obs_buffer1 = np.zeros(self.obs_dim, dtype=np.float32)
        self._obs_buffer2 = np.zeros(self.obs_dim, dtype=np.float32)
        self._obs_buffer3 = np.zeros(self.obs_dim, dtype=np.float32)
        self._act_buffer1 = np.zeros(self.act_dim, dtype=np.float32)
        self._act_buffer2 = np.zeros(self.act_dim, dtype=np.float32)
        self._act_buffer3 = np.zeros(self.act_dim, dtype=np.float32)

    @property
    def is_healthy(self) -> bool:
        sv = self.state_vector()
        return (abs(sv[1]) < 0.5) and (abs(sv[3:6]) < 0.7).all()

    @property
    def done(self) -> bool:
        return not self.is_healthy

    def _get_obs(self) -> np.ndarray:
        # 과거 3프레임 관측 + 과거 3프레임 액션 스택
        return np.concatenate([
            self._obs_buffer1, self._obs_buffer2, self._obs_buffer3,
            self._act_buffer1, self._act_buffer2, self._act_buffer3
        ], axis=0)

    def reset_model(self) -> np.ndarray:
        # 초기 상태 설정
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)

        # joint position 18차원 초기화
        obs = self.state_vector()[6:6+self.obs_dim].astype(np.float32)
        self._obs_buffer1[:] = obs
        self._obs_buffer2[:] = obs
        self._obs_buffer3[:] = obs

        # action buffer 초기화
        self._act_buffer1[:] = 0
        self._act_buffer2[:] = 0
        self._act_buffer3[:] = 0
        
        return self._get_obs()

    def step(self, action: np.ndarray):
        # 1) 전진 거리 기반 reward를 위해 초기 x 좌표 저장
        x_init = float(self.state_vector()[0])
        # 2) 시뮬레이션 수행
        self.do_simulation(action, self.frame_skip)

        # 3) action history 업데이트
        self._act_buffer3 = self._act_buffer2.copy()
        self._act_buffer2 = self._act_buffer1.copy()
        self._act_buffer1 = action.copy()

        # 4) 관측 history 업데이트 (joint positions)
        obs = self.state_vector()[6:6+self.obs_dim].astype(np.float32)
        self._obs_buffer3 = self._obs_buffer2.copy()
        self._obs_buffer2 = self._obs_buffer1.copy()
        self._obs_buffer1 = obs.copy()

                        # 5) reward 계산 (원본 방정식 사용 및 control penalty 포함)
        x_del      = float(self.state_vector()[0] - x_init)
        y_err      = float(abs(self.state_vector()[1]))
        # action difference penalty
        ctrl       = float(np.sum((self._act_buffer1 - self._act_buffer2) ** 2))
        torque_rms = float(np.sqrt(np.mean(self.data.actuator_force ** 2)))
        reward     = x_del / (torque_rms + 1.0) / (y_err + 0.1)

        # 6) 종료 및 info 반환
        terminated = self.done
        truncated  = False
        info       = {
            'x_delta':      x_del,
            'y_error':      y_err,
            'control_norm': ctrl,
            'total_reward': reward
        }
        return self._get_obs(), reward, terminated, truncated, info

    def viewer_setup(self):
        for key, val in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(val, np.ndarray):
                getattr(self.viewer.cam, key)[:] = val
            else:
                setattr(self.viewer.cam, key, val)

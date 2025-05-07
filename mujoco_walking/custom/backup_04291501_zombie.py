import os
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces, utils
import numpy as np

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

        # XML 파일 절대경로 계산
        if xml_file is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            xml_file = os.path.join(module_dir, 'assets', 'Hexy_ver_2.3', 'hexy-v2.3.xml')

        # 부모 클래스 초기화 (모델과 self.data 생성)
        super().__init__(
            model_path=xml_file,
            frame_skip=frame_skip,
            observation_space=None,      # 이후 재정의
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs
        )

        # 버퍼 차원 설정
        self.obs_dim = 18
        self.act_dim = self.model.nu

        # 관측·행동 공간 정의
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim*3 + self.act_dim*3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.act_dim,), dtype=np.float32
        )

        # 스택 버퍼 초기화
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
        # 과거 3프레임 관측 + 과거 3프레임 액션 스택 반환
        return np.concatenate([
            self._obs_buffer1, self._obs_buffer2, self._obs_buffer3,
            self._act_buffer1, self._act_buffer2, self._act_buffer3
        ], axis=0)

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)

        # 관절 위치(12차원) 초기 관측
        obs12 = self.state_vector()[6:18]
        self._obs_buffer1[:] = obs12
        self._obs_buffer2[:] = obs12
        self._obs_buffer3[:] = obs12

        # 액션 버퍼 초기화
        self._act_buffer1[:] = 0
        self._act_buffer2[:] = 0
        self._act_buffer3[:] = 0

        # dtype을 float32로 맞춰 반환
        return self._get_obs().astype(np.float32)

    def step(self, action: np.ndarray):
        x_init = self.state_vector()[0]
        # 시뮬레이션 실행
        self.do_simulation(action, self.frame_skip)

        # 관측·액션 버퍼 갱신
        self._act_buffer3 = self._act_buffer2.copy()
        self._act_buffer2 = self._act_buffer1.copy()
        self._act_buffer1 = action.copy()
        obs12 = self.state_vector()[6:18]
        self._obs_buffer3 = self._obs_buffer2.copy()
        self._obs_buffer2 = self._obs_buffer1.copy()
        self._obs_buffer1 = obs12.copy()

        # 보상 계산
        x_del = self.state_vector()[0] - x_init
        y_err = abs(self.state_vector()[1])
        ctrl = np.sum((self._act_buffer1 - self._act_buffer2) ** 2)
        torque_rms = np.sqrt(np.mean(self.data.actuator_force ** 2))
        reward = x_del / (torque_rms + 1) / (y_err + 0.1)

        terminated = self.done
        truncated  = False
        info = {
            'x_delta': x_del,
            'y_error': y_err,
            'control_norm': ctrl,
            'total': reward
        }

        # dtype을 float32로 맞춰 반환
        return self._get_obs().astype(np.float32), reward, terminated, truncated, info

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

import os
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
import mujoco as _mujoco  # MuJoCo Python API for name-to-id lookup

DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}

class HexyEnv(MujocoEnv, utils.EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 5,
        render_mode: str = None,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, render_mode)

        # XML 파일 경로 결정
        if xml_file is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            xml_file = os.path.join(
                module_dir, 'assets', 'Hexy_ver_2.3', 'hexy-v2.3.xml'
            )

        super().__init__(
            model_path=xml_file,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )

        self.act_dim = self.model.nu
        self.obs_dim = 12

        # 관측·행동 공간 정의
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim * 3 + self.act_dim * 3,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.act_dim,),
            dtype=np.float32
        )

        # 스택 버퍼 초기화 (전부 0으로 시작)
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

    def reset_model(self) -> np.ndarray:
        # 상태만 초기화, 버퍼는 모두 0으로 유지(원본 로직과 동일)
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)

        # (버퍼는 생성 시 0으로 설정됐으므로 여기서는 건드리지 않음)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self._obs_buffer1, self._obs_buffer2, self._obs_buffer3,
            self._act_buffer1, self._act_buffer2, self._act_buffer3
        ], axis=0)

    def step(self, action: np.ndarray):
        # 1) 전진 거리 측정용 초기 x
        x_init = self.state_vector()[0]
        self.do_simulation(action, self.frame_skip)

        # 2) 히스토리 버퍼 업데이트
        self._act_buffer3 = self._act_buffer2.copy()
        self._act_buffer2 = self._act_buffer1.copy()
        self._act_buffer1 = action.copy()

        obs12 = self.state_vector()[6:18].astype(np.float32)
        self._obs_buffer3 = self._obs_buffer2.copy()
        self._obs_buffer2 = self._obs_buffer1.copy()
        self._obs_buffer1 = obs12

        # 3) 보상 계산 (원본 로직 그대로)
        x_del      = float(self.state_vector()[0] - x_init)
        y_err      = float(abs(self.state_vector()[1]))
        ctrl       = float(np.sum((self._act_buffer1 - self._act_buffer2)**2))
        torque_rms = float(np.sqrt(np.mean(self.data.actuator_force**2)))
        reward     = x_del / (torque_rms + 1.0) / (y_err + 0.1)

        # 4) 종료 플래그 및 info
        terminated = self.done
        truncated  = False
        info = {
            'x_delta':      x_del,
            'y_error':      y_err,
            'control_norm': ctrl,
            'total':        reward
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def viewer_setup(self):
        for key, val in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(val, np.ndarray):
                getattr(self.viewer.cam, key)[:] = val
            else:
                setattr(self.viewer.cam, key, val)

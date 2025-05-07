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

        # — 변경1: 관절 수와 프레임당 관측 차원 정의 —
        self.act_dim   = self.model.nu                     # actuator 개수(=18)
        self.obs_fdim  = self.act_dim * 2                  # [qpos, qvel] 합쳐서 프레임당 36
        self.hist_len  = 3                                 # 과거 3프레임
        total_obs_dim  = self.obs_fdim * self.hist_len \
                         + self.act_dim * self.hist_len    # 위치+속도 히스토리 + 액션 히스토리

        # 관측 및 행동 공간 정의
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.act_dim,), dtype=np.float32
        )

        # — 변경2: 히스토리 버퍼 (위치, 속도, 액션) 초기화 —
        self._pos_buf = [np.zeros(self.act_dim, dtype=np.float32) for _ in range(self.hist_len)]
        self._vel_buf = [np.zeros(self.act_dim, dtype=np.float32) for _ in range(self.hist_len)]
        self._act_buf = [np.zeros(self.act_dim, dtype=np.float32) for _ in range(self.hist_len)]

    @property
    def is_healthy(self) -> bool:
        sv = self.state_vector()
        return (abs(sv[1]) < 0.5) and (abs(sv[3:6]) < 0.7).all()

    @property
    def done(self) -> bool:
        return not self.is_healthy

    def _get_obs(self) -> np.ndarray:
        # 과거 히스토리 순서: 가장 오래된 → 최근
        pos_hist = np.concatenate(self._pos_buf, axis=0)
        vel_hist = np.concatenate(self._vel_buf, axis=0)
        act_hist = np.concatenate(self._act_buf, axis=0)
        return np.concatenate([pos_hist, vel_hist, act_hist], axis=0)

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)

        # 현재 프레임의 위치·속도
        cur_pos = self.data.qpos[6:].astype(np.float32)
        cur_vel = self.data.qvel[6:].astype(np.float32)

        # 버퍼 초기화
        for i in range(self.hist_len):
            self._pos_buf[i][:] = cur_pos
            self._vel_buf[i][:] = cur_vel
            self._act_buf[i][:] = 0.0

        return self._get_obs()

    def step(self, action: np.ndarray):
        x_init = float(self.state_vector()[0])
        self.do_simulation(action, self.frame_skip)

        # 히스토리 버퍼 갱신 (pop 앞, append 뒤)
        self._act_buf.pop(0);    self._act_buf.append(action.copy())
        cur_pos = self.data.qpos[6:].astype(np.float32)
        cur_vel = self.data.qvel[6:].astype(np.float32)
        self._pos_buf.pop(0);    self._pos_buf.append(cur_pos)
        self._vel_buf.pop(0);    self._vel_buf.append(cur_vel)

        # reward 계산 (앞으로 나간 거리 – 토크 패널티 – y편차 패널티)
        x_del      = float(self.state_vector()[0] - x_init)
        y_err      = float(abs(self.state_vector()[1]))
        ctrl_cost  = float(np.sum((action - self._act_buf[-2]) ** 2))
        # torque_rms 대신 control_penalty로만 처리
        reward     = 1.0 * x_del - 0.1 * ctrl_cost - 0.5 * y_err

        terminated = self.done
        truncated  = False
        info = {
            'x_delta':      x_del,
            'y_error':      y_err,
            'control_cost': ctrl_cost,
            'total_reward': reward
        }
        return self._get_obs(), reward, terminated, truncated, info

    def viewer_setup(self):
        for key, val in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(val, np.ndarray):
                getattr(self.viewer.cam, key)[:] = val
            else:
                setattr(self.viewer.cam, key, val)

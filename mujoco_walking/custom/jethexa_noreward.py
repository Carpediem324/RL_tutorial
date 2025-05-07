# jethexa_noreward.py
import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces, utils

DEFAULT_CAMERA_CONFIG = {'distance': 1.5}

class JethexaEnv(MujocoEnv, utils.EzPickle):
    """
    • 기본 reward = forward_vel
    • lateral, yaw 제약조건 반환
    • position-actuator 목표값 매핑
    • action smoothing 추가
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 xml_file=None,
                 frame_skip=5,
                 default_camera_config=DEFAULT_CAMERA_CONFIG,
                 render_mode=None,
                 **kwargs):
        utils.EzPickle.__init__(self, xml_file, frame_skip,
                                default_camera_config, render_mode, **kwargs)
        if xml_file is None:
            module_dir = os.path.dirname(__file__)
            xml_file = os.path.join(module_dir, 'assets/jethexa_shh/real.xml')
        super().__init__(model_path=xml_file,
                         frame_skip=frame_skip,
                         observation_space=None,
                         render_mode=render_mode,
                         default_camera_config=default_camera_config,
                         **kwargs)

        # state/action dims
        self.obs_dim   = 18
        self.act_dim   = self.model.nu
        self.pose_dim  = 6
        buf = self.obs_dim*3 + self.act_dim*3 + self.pose_dim

        # spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, (buf,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)

        # actuator ctrlrange → position 매핑
        cr = self.model.actuator_ctrlrange
        self.ctrl_center     = ((cr[:,0] + cr[:,1]) * 0.5).copy()
        self.ctrl_half_range = ((cr[:,1] - cr[:,0]) * 0.5).copy()

        # 히스토리 버퍼
        self._obs_buf = [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(3)]
        self._act_buf = [np.zeros(self.act_dim, dtype=np.float32) for _ in range(3)]

        # trackers
        self.last_x      = 0.0
        self.cur_step    = 0
        self.max_steps   = 1000
        self.stand_count = 0
        self.stand_thresh= 50

        # 제약조건 margin
        self.lat_margin = 0.2   # m
        self.yaw_margin = 0.1   # rad
        self.init_yaw   = 0.0

        # action smoothing
        self._last_action = np.zeros(self.act_dim, dtype=np.float32)
        self._alpha       = 0.7  # smoothing factor

    def _get_pose(self):
        qpos = self.data.qpos
        pos  = qpos[:3]
        qw,qx,qy,qz = qpos[3:7]
        sinr = 2*(qw*qx + qy*qz); cosr = 1-2*(qx*qx + qy*qy)
        roll = np.arctan2(sinr, cosr)
        sinp = np.clip(2*(qw*qy - qz*qx), -1,1)
        pitch= np.arcsin(sinp)
        siny = 2*(qw*qz + qx*qy); cosy = 1-2*(qy*qy + qz*qz)
        yaw  = np.arctan2(siny, cosy)
        return np.array([pos[0],pos[1],pos[2],roll,pitch,yaw], dtype=np.float32)

    @property
    def done(self):
        return (self.cur_step >= self.max_steps
                or self.stand_count >= self.stand_thresh)

    def _get_obs(self):
        pose = self._get_pose()
        return np.concatenate(self._obs_buf + self._act_buf + [pose], axis=0)

    def reset_model(self):
        qpos, qvel = self.init_qpos.copy(), self.init_qvel.copy()
        # small noise
        qpos[2] += np.random.uniform(-0.002,0.002)
        qpos[7:] += np.random.uniform(-0.02,0.02,len(qpos[7:]))
        self.set_state(qpos, qvel)

        base = self.state_vector()[6:6+self.obs_dim].astype(np.float32)
        for b in self._obs_buf: b[:] = base
        for b in self._act_buf: b[:] = 0.0

        pose = self._get_pose()
        self.init_yaw   = pose[5]
        self.last_x     = float(self.state_vector()[0])
        self.cur_step   = 0
        self.stand_count= 0
        self._last_action[:] = 0.0

        return self._get_obs()

    def step(self, action):
        self.cur_step += 1
        x0 = float(self.state_vector()[0])

        # smoothing
        action = self._alpha * action + (1-self._alpha) * self._last_action
        self._last_action = action.copy()

        # position 매핑
        target = self.ctrl_center + action * self.ctrl_half_range
        self.do_simulation(target, self.frame_skip)

        # 히스토리 갱신
        self._act_buf.pop(); self._act_buf.insert(0, action.copy())
        obs = self.state_vector()[6:6+self.obs_dim].astype(np.float32)
        self._obs_buf.pop(); self._obs_buf.insert(0, obs)

        # pose & constraints
        pose = self._get_pose()
        x1, y1, yaw = pose[0], pose[1], pose[5]
        if abs(x1 - self.last_x) < 5e-4:
            self.stand_count += 1
        else:
            self.stand_count = 0
        self.last_x = x1

        # forward velocity
        forward_vel = (x1 - x0) / (self.frame_skip * self.dt)
        # constraints
        g_lat = max(0.0, abs(y1) - self.lat_margin)
        g_yaw = max(0.0, abs(yaw - self.init_yaw) - self.yaw_margin)

        info = {
            'forward_vel': forward_vel,
            'constraints': np.array([g_lat, g_yaw], dtype=np.float32)
        }
        # env reward도 forward_vel로 반환
        return self._get_obs(), forward_vel, self.done, False, info

    def viewer_setup(self):
        for k,v in DEFAULT_CAMERA_CONFIG.items():
            setattr(self.viewer.cam, k, v)

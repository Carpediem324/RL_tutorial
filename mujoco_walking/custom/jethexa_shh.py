# 파일명: custom/jethexa_shh.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces, utils

DEFAULT_CAMERA_CONFIG = {'distance': 1.5}

class JethexaEnv(MujocoEnv, utils.EzPickle):
    """
    6족 로봇 Jethexa 환경
     - 관측: 과거 3프레임 관절 위치 + 과거 3프레임 액션 + 현재 자세(x,y,z,roll,pitch,yaw)
     - 행동: 각 관절 토크 (−1~1 정규화)
     - 보상: 
         + forward_scale * forward_vel  
         − energy_scale  * ∑action²  
         − lateral_scale * |y|  
         − yaw_scale     * |yaw − init_yaw|  
     - 넘어짐(is_healthy=False) 또는 “가만히 있음” (stand_count ≥ 20) 시 terminated  
       시간제한(TimeLimit)은 Gym이 자동으로 truncated 처리  
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
            xml_file = os.path.join(module_dir, 'assets/jethexa_shh', 'real.xml')

        super().__init__(
            model_path=xml_file,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs
        )

        # 차원 정의
        self.obs_dim   = 18
        self.act_dim   = self.model.nu
        self.pose_dim  = 6
        buf = self.obs_dim*3 + self.act_dim*3 + self.pose_dim

        # 공간 정의
        self.observation_space = spaces.Box(-np.inf, np.inf, (buf,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)

        # history 버퍼
        self._obs_buf = [np.zeros(self.obs_dim, np.float32) for _ in range(3)]
        self._act_buf = [np.zeros(self.act_dim, np.float32) for _ in range(3)]

        # 초기 heading
        self.init_yaw     = 0.0

        # “가만히 있음” 감지
        self.last_x       = 0.0
        self.stand_count  = 0
        self.stand_thresh = 20   # 20 스텝 이상 이동 없으면 done

        # 보상 스케일
        self.forward_scale = 5.0    # 전진 속도 보상
        self.energy_scale  = 0.001  # 토크 사용 페널티
        self.lateral_scale = 0.1    # 측면 편차 페널티
        self.yaw_scale     = 0.1    # yaw 편차 페널티

    def _get_pose(self):
        q = self.data.qpos
        pos = q[:3]
        qw,qx,qy,qz = q[3:7]
        sinr = 2*(qw*qx+qy*qz); cosr=1-2*(qx*qx+qy*qy)
        roll  = np.arctan2(sinr, cosr)
        sinp  = np.clip(2*(qw*qy-qz*qx), -1,1)
        pitch = np.arcsin(sinp)
        siny  = 2*(qw*qz+qx*qy); cosy=1-2*(qy*qy+qz*qz)
        yaw   = np.arctan2(siny, cosy)
        return np.array([pos[0],pos[1],pos[2],roll,pitch,yaw], np.float32)

    @property
    def is_healthy(self) -> bool:
        z = float(self.data.qpos[2])
        if z < 0.02: 
            return False
        qw,qx,qy,qz = self.data.qpos[3:7]
        roll  = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
        pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1,1))
        return abs(roll) < 0.8 and abs(pitch) < 0.8

    @property
    def done(self) -> bool:
        return (not self.is_healthy) or (self.stand_count >= self.stand_thresh)

    def _get_obs(self):
        pose = self._get_pose()
        return np.concatenate(self._obs_buf + self._act_buf + [pose], axis=0)

    def reset_model(self):
        qpos,qvel = self.init_qpos.copy(), self.init_qvel.copy()
        qpos[2] += np.random.uniform(-0.002,0.002)
        self.set_state(qpos, qvel)

        obs0 = self.state_vector()[6:6+self.obs_dim].astype(np.float32)
        for buf in self._obs_buf: buf[:] = obs0
        for buf in self._act_buf: buf[:] = 0.0

        self.init_yaw    = self._get_pose()[5]
        self.last_x      = float(self.state_vector()[0])
        self.stand_count = 0
        return self._get_obs()

    def step(self, action):
        # 1) x0 저장
        x0 = float(self.state_vector()[0])
        # 2) 시뮬레이션
        self.do_simulation(action, self.frame_skip)
        # 3) 히스토리 업데이트
        obs1 = self.state_vector()[6:6+self.obs_dim].astype(np.float32)
        self._act_buf.pop(); self._act_buf.insert(0, action.copy())
        self._obs_buf.pop(); self._obs_buf.insert(0, obs1)
        # 4) pose & forward_vel
        pose = self._get_pose()
        x1, y1, yaw = pose[0], pose[1], pose[5]
        dt = self.frame_skip * self.dt
        forward_vel = (x1 - x0) / dt
        # 5) 페널티
        energy_cost = np.sum(action**2)
        lateral_err  = abs(y1)
        yaw_err      = abs(yaw - self.init_yaw)
        # 6) 가만히 있음 감지
        if abs(x1 - self.last_x) < 1e-4:
            self.stand_count += 1
        else:
            self.stand_count = 0
        self.last_x = x1
        # 7) 보상
        reward = (
            self.forward_scale * forward_vel
            - self.energy_scale  * energy_cost
            - self.lateral_scale * lateral_err
            - self.yaw_scale     * yaw_err
        )
        return self._get_obs(), reward, self.done, False, {'forward_vel': forward_vel}

    def viewer_setup(self):
        for k,v in DEFAULT_CAMERA_CONFIG.items():
            setattr(self.viewer.cam, k, v.copy() if hasattr(v,'copy') else v)

#!/usr/bin/env python3
import sys
import os
import time

# ─── custom 환경 등록(__init__.py 경로) ─────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import __init__  # Hexy-v4 환경을 register_env 해 줌
# ─────────────────────────────────────────────────────────────

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import gymnasium as gym
from custom.hexy_v4 import HexyEnv
from stable_baselines3 import PPO

class JointCommandPublisher(Node):
    def __init__(self, joint_names):
        super().__init__('hexy_joint_command_pub')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        self.joint_names = joint_names

    def publish(self, positions):
        # positions가 numpy 배열일 경우 flatten, 리스트일 경우 그대로 사용
        vals = positions.flatten() if hasattr(positions, 'flatten') else positions
        # float로 변환
        flat_positions = [float(v) for v in vals]
        # 메시지 작성 및 퍼블리시
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = flat_positions
        self.pub.publish(msg)


def main():
    # 1) ROS2 초기화
    rclpy.init()

    # 2) 퍼블리셔 노드 생성
    joint_names = [
        'coxa_joint_LF','femur_joint_LF','tibia_joint_LF',
        'coxa_joint_LM','femur_joint_LM','tibia_joint_LM',
        'coxa_joint_LR','femur_joint_LR','tibia_joint_LR',
        'coxa_joint_RF','femur_joint_RF','tibia_joint_RF',
        'coxa_joint_RM','femur_joint_RM','tibia_joint_RM',
        'coxa_joint_RR','femur_joint_RR','tibia_joint_RR'
    ]
    node = JointCommandPublisher(joint_names)

    # ─────────────────────────────────────────────────────────────
    # 사용자 설정
    DATE   = "250430"
    TRIAL  = "D"
    STEPS  = "8400000"
    # ─────────────────────────────────────────────────────────────

    # 3) 환경 생성 - gym.make을 사용해 벡터 학습 시 사용한 동일 환경 생성(env_id는 register_env로 등록된 이름)
    env = gym.make("Hexy-v4", render_mode="human")

    # 4) 모델 로드
    save_dir   = os.path.join(script_dir, f"save_model_{DATE}", TRIAL)
    model_file = os.path.join(save_dir, f"{TRIAL}_model_{DATE}_{STEPS}_steps")

    print(f"Loading model from: {model_file}")
    model = PPO.load(model_file, device="cpu")

    # 5) 환경 초기화
    obs, _ = env.reset()

    try:
        while rclpy.ok():
            # 6) 액션 예측
            action, _ = model.predict(obs, deterministic=True)
            # 7) 퍼블리시
            node.publish(action)
            # 8) 환경 스텝 및 렌더
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()
            # 9) 종료 리셋
            if terminated or truncated:
                obs, _ = env.reset()
            # 10) 딜레이
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
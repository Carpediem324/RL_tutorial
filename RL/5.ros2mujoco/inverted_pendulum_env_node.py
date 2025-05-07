#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, Bool
import gym # gymnasium 대신 gym 사용 (사용자 PPO 코드에 맞춰)
import numpy as np
import time
import os

class InvertedPendulumEnvNode(Node):
    def __init__(self):
        super().__init__('inverted_pendulum_env')
        self.get_logger().info("Initializing InvertedPendulumEnvNode (using gym)...")

        # 1) Gym MuJoCo 환경 생성 (gym 사용)
        try:
            # gym 버전 확인 및 경고 처리
            if gym.__version__.startswith('0.26'):
                 self.env = gym.make("InvertedPendulum-v4", render_mode=None) # 최신 gym API 유사
            else:
                 self.env = gym.make("InvertedPendulum-v4") # 구버전 gym API
            self.get_logger().info("Gym environment 'InvertedPendulum-v4' created successfully.")

            self.obs_dim = self.env.observation_space.shape[0]
            self.act_dim = self.env.action_space.shape[0] # 1
            self.act_low = self.env.action_space.low     # [-3.]
            self.act_high = self.env.action_space.high   # [3.]
            self.get_logger().info(f"Observation Dim: {self.obs_dim}, Action Dim: {self.act_dim}")
            self.get_logger().info(f"Action Low: {self.act_low}, Action High: {self.act_high}")

        except Exception as e:
            self.get_logger().error(f"Failed to create Gym environment: {e}")
            rclpy.shutdown()
            return

        # 2) ROS2 퍼블리셔
        self.obs_pub    = self.create_publisher(Float32MultiArray, '/inverted_pendulum/observation', 10)
        self.rew_pub    = self.create_publisher(Float32,            '/inverted_pendulum/reward',      10)
        self.done_pub   = self.create_publisher(Bool,               '/inverted_pendulum/done',        10)
        self.get_logger().info("ROS2 publishers created.")

        # 3) 액션 구독자
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/inverted_pendulum/action',
            self.action_callback,
            10
        )
        self.get_logger().info("ROS2 subscriber created.")

        # 4) 초기 환경 리셋 및 관측 퍼블리시 (짧은 지연 시간 추가)
        try:
            # gym 버전별 reset 반환값 처리
            reset_ret = self.env.reset()
            if isinstance(reset_ret, tuple): # 최신 gym 스타일 (obs, info)
                self.current_obs = reset_ret[0]
            else: # 구버전 gym 스타일 (obs)
                self.current_obs = reset_ret

            self.get_logger().info("Environment reset. Pausing briefly before first publish...")
            time.sleep(0.5) # 구독 노드 준비 시간 확보
            self.publish_observation(self.current_obs)
            self.get_logger().info("Initial observation published.")
        except Exception as e:
            self.get_logger().error(f"Error during initial environment reset: {e}")
            rclpy.shutdown()
            return

    def publish_observation(self, obs):
        msg = Float32MultiArray()
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        msg.data = obs.astype(np.float32).tolist()
        self.obs_pub.publish(msg)

    def action_callback(self, msg: Float32MultiArray):
        action = np.array(msg.data, dtype=np.float32)
        # 에이전트가 Tanh 출력을 하므로, 환경의 실제 범위로 스케일링 필요 가능성 있음
        # 또는 에이전트의 clip을 믿거나 여기서 clip 수행
        action = np.clip(action, self.act_low, self.act_high)

        try:
            # gym 버전별 step 반환값 처리
            step_ret = self.env.step(action)
            if len(step_ret) == 4: # 구버전 gym (obs, reward, done, info)
                 next_obs, reward, done_flag, info = step_ret
            elif len(step_ret) == 5: # 최신 gym (obs, reward, terminated, truncated, info)
                 next_obs, reward, terminated, truncated, info = step_ret
                 done_flag = terminated or truncated
            else:
                 self.get_logger().error(f"Unexpected return format from env.step: {step_ret}")
                 return

            # 결과 퍼블리시
            self.publish_observation(next_obs)
            rew_msg = Float32(data=float(reward))
            self.rew_pub.publish(rew_msg)
            done_msg = Bool(data=bool(done_flag))
            self.done_pub.publish(done_msg)

            # 에피소드 종료 시 리셋
            if done_flag:
                reset_ret = self.env.reset()
                if isinstance(reset_ret, tuple):
                    self.current_obs = reset_ret[0]
                else:
                    self.current_obs = reset_ret
                self.publish_observation(self.current_obs)
                # self.get_logger().info("Episode finished. Environment reset.")

        except Exception as e:
            self.get_logger().error(f"Error during environment step or reset: {e}")
            done_msg = Bool(data=True)
            self.done_pub.publish(done_msg)

    def destroy_node(self):
        if hasattr(self, 'env'):
             self.env.close()
             self.get_logger().info("Environment closed.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = InvertedPendulumEnvNode()
    if hasattr(node, 'env') and node.env is not None:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Keyboard interrupt received, shutting down.")
        except Exception as e:
            node.get_logger().error(f"Unhandled exception during spin: {e}")
        finally:
            node.destroy_node()
            # rclpy.shutdown() # destroy_node에서 처리될 수 있음
    else:
        print("Failed to initialize the environment node.")
    rclpy.shutdown() # 노드 종료 후 shutdown 호출

if __name__ == '__main__':
    main()

#!/bin/bash

STEP_TIME=0.5  # 각 자세 유지 시간

# 모든 관절 초기자세
initial_pose() {
  ros2 topic pub --once /joint_command sensor_msgs/msg/JointState "{
    name: [
      'coxa_joint_LF','femur_joint_LF','tibia_joint_LF',
      'coxa_joint_LM','femur_joint_LM','tibia_joint_LM',
      'coxa_joint_LR','femur_joint_LR','tibia_joint_LR',
      'coxa_joint_RF','femur_joint_RF','tibia_joint_RF',
      'coxa_joint_RM','femur_joint_RM','tibia_joint_RM',
      'coxa_joint_RR','femur_joint_RR','tibia_joint_RR'],
    position: [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]}"
}

# Tripod 1: RF, LR, RR 다리를 앞으로 이동하며 든다
tripod_1() {
  ros2 topic pub --once /joint_command sensor_msgs/msg/JointState "{
    name: [
      'femur_joint_RF','tibia_joint_RF','coxa_joint_RF',
      'femur_joint_LR','tibia_joint_LR','coxa_joint_LR',
      'femur_joint_RR','tibia_joint_RR','coxa_joint_RR'],
    position: [
      0.4,-0.7,0.2,
      0.4,-0.7,0.2,
      0.4,-0.7,0.2]}"
}

# Tripod 2: LF, RM, LM 다리를 앞으로 이동하며 든다
tripod_2() {
  ros2 topic pub --once /joint_command sensor_msgs/msg/JointState "{
    name: [
      'femur_joint_LF','tibia_joint_LF','coxa_joint_LF',
      'femur_joint_RM','tibia_joint_RM','coxa_joint_RM',
      'femur_joint_LM','tibia_joint_LM','coxa_joint_LM'],
    position: [
      0.4,-0.7,0.2,
      0.4,-0.7,0.2,
      0.4,-0.7,0.2]}"
}

# 걷기 루프
echo "Starting correct tripod gait... (Ctrl+C to stop)"
initial_pose
sleep $STEP_TIME

while true; do
  tripod_1
  sleep $STEP_TIME

  initial_pose
  sleep $STEP_TIME

  tripod_2
  sleep $STEP_TIME

  initial_pose
  sleep $STEP_TIME
done

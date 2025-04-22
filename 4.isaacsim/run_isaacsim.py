#!/usr/bin/env python3
# 파일명: run_hexapod_sim.py

from omni.isaac.kit import SimulationApp

# 1) SimulationApp 초기화
config = {
    "headless": False,                  # 그래픽 창을 띄웁니다
    "renderer": "RayTracedLighting",    # 사용할 렌더러 지정
    "scene.name": "HexapodSim"          # 씬 이름 설정
}
simulation_app = SimulationApp(config)

# 2) USD 파일 경로 설정 및 로드
hexapod_usd_path = "/home/ubuntu/ros2_ws/src/jethexa_description/urdf/robot/jethexa/hexapod_0422_1012.usd"
import omni.usd
omni.usd.get_context().open_stage(hexapod_usd_path)

# 3) World 생성 및 기본 지면 추가
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane

world = World(stage_units_in_meters=1.0)
ground = GroundPlane(prim_path="/ground_plane")
world.scene.add(ground)
world.reset()

# 4) 시뮬레이션 루프
try:
    while simulation_app.is_running():
        # 물리 연산 및 렌더링을 같이 수행
        world.step(render=True)
        # 윈도우 이벤트 처리(키보드, 마우스 등)
        simulation_app.update()
finally:
    # 시뮬레이터 종료
    simulation_app.close()

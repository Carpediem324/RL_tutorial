#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import tempfile
import mujoco
import glfw
from shutil import which

class CartPoleVisualizer:
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        # 1) URDF → MJCF 변환
        self.mjcf_xml = self._convert_urdf_to_mjcf(urdf_path)
        # 2) MJCF(XML) → MjModel/MjData 생성
        print(f"▶ MJCF 로드 시도: {self.mjcf_xml}")
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_xml)
        self.data  = mujoco.MjData(self.model)
        print(f"▶ MjModel 준비 완료: nbody={self.model.nbody}, njnt={self.model.njnt}")

    def _convert_urdf_to_mjcf(self, urdf_path: str) -> str:
        # simulate 바이너리 위치 확인
        simulate_bin = which('simulate')
        if simulate_bin is None:
            raise RuntimeError(
                "MuJoCo CLI `simulate` 를 찾을 수 없습니다. "
                "MuJoCo 설치 디렉터리/bin을 PATH에 추가해주세요."
            )
        print(f"1️⃣ URDF → MJCF 변환 명령: {simulate_bin} convert {urdf_path} <temp.xml>")
        # 임시 디렉터리 생성
        tmpdir = tempfile.mkdtemp(prefix="mjcf_")
        out_xml = os.path.join(tmpdir, "model.xml")
        # 변환 실행
        subprocess.run(
            [simulate_bin, "convert", urdf_path, out_xml],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"▶ 변환 완료, MJCF 파일 위치: {out_xml}")
        return out_xml

    def init_viewer(self, width: int = 800, height: int = 600, title: str = "CartPole Visualizer"):
        print("2️⃣ GLFW 초기화 시도")
        if not glfw.init():
            raise RuntimeError("GLFW 초기화에 실패했습니다.")
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW 윈도우 생성에 실패했습니다.")
        print("▶ GLFW 윈도우 생성 성공")
        glfw.make_context_current(self.window)

        # MuJoCo 렌더러 설정
        self.scene    = mujoco.MjvScene(self.model, maxgeom=1000)
        self.cam      = mujoco.MjvCamera()
        self.pert     = mujoco.MjvPerturb()
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.ctx      = mujoco.MjrContext(self.model, 150)

        # 카메라 기본 위치
        self.cam.lookat    = self.model.stat.center
        self.cam.distance  = self.model.stat.extent * 1.5
        self.cam.azimuth   = 90
        self.cam.elevation = -30

    def run(self):
        print("3️⃣ 렌더링 루프 시작")
        while not glfw.window_should_close(self.window):
            mujoco.mj_step(self.model, self.data)
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = fb_w, fb_h

            mujoco.mjv_updateScene(
                self.model, self.data,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.pert, self.cam, self.scene
            )
            mujoco.mjr_render(self.viewport, self.scene, self.ctx)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        print("4️⃣ 루프 종료, GLFW 정리")
        glfw.terminate()


if __name__ == "__main__":
    try:
        # 스크립트 기준 URDF 경로 지정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_file  = os.path.abspath(os.path.join(script_dir, "../urdf/cart_pole.urdf"))
        print(f"🔍 사용할 URDF 파일: {urdf_file}")

        viz = CartPoleVisualizer(urdf_file)
        viz.init_viewer()
        viz.run()

    except subprocess.CalledProcessError as e:
        print("❌ URDF → MJCF 변환 중 오류:\n", e.stderr.decode())
        sys.exit(1)
    except Exception as e:
        print("❌ 에러 발생:\n", e)
        sys.exit(1)

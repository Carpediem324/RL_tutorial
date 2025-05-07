#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_urdf_no_deps.py

- URDF 파일이 올바른 XML인지 파싱을 통해 검사
- link/joint 개수 확인
- <mesh> 태그로 선언된 모든 파일이 실제로 존재하는지 확인
"""

import os
import sys
import xml.etree.ElementTree as ET

class URDFChecker:
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.base_dir = os.path.dirname(os.path.abspath(urdf_path))

    def parse_xml(self):
        try:
            self.tree = ET.parse(self.urdf_path)
            self.root = self.tree.getroot()
            print(f"✅ XML 파싱 성공: root 태그 = '{self.root.tag}'")
        except ET.ParseError as e:
            print(f"❌ XML 파싱 실패:\n  {e}")
            sys.exit(1)

    def summarize(self):
        links  = self.root.findall('link')
        joints = self.root.findall('joint')
        meshes = self.root.findall('.//mesh')
        print(f"링크 개수  : {len(links)}")
        print(f"조인트 개수: {len(joints)}")
        print(f"메쉬 선언 : {len(meshes)}개")

    def check_mesh_files(self):
        print("\n[메쉬 파일 존재 여부 검사]")
        meshes = self.root.findall('.//mesh')
        missing = 0
        for mesh in meshes:
            fn = mesh.attrib.get('filename')
            path = os.path.normpath(os.path.join(self.base_dir, fn))
            if os.path.exists(path):
                print(f"  ✔ {fn}")
            else:
                print(f"  ✘ {fn} (파일 없음: {path})")
                missing += 1
        if missing == 0:
            print("모든 메쉬 파일이 존재합니다.")
        else:
            print(f"{missing}개 메쉬 파일을 찾을 수 없습니다.")

    def run(self):
        self.parse_xml()
        self.summarize()
        self.check_mesh_files()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("사용법: python urdf_viewer.py URDF경로")
        sys.exit(1)
    checker = URDFChecker(sys.argv[1])
    checker.run()

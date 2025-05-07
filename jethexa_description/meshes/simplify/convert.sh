#!/bin/bash
# convert.sh: jethexa_leg_center.urdf + all .stl 을 임시 디렉토리로 옮겨서 변환

# 1) 임시 디렉토리 만들기
TMP=$(mktemp -d)

# 2) URDF 와 모든 STL 복사
cp jethexa_leg_center.urdf *.stl "$TMP"/

# 3) 그 디렉토리로 이동
pushd "$TMP" >/dev/null

# 4) 변환 실행 (mesh 파일이 모두 여기 있으므로 성공)
urdf2mjcf jethexa_leg_center.urdf \
  --output jethexa.xml \
  --copy-meshes

# 5) 결과물을 원래 폴더로 복사
cp jethexa.xml "$OLDPWD"/

# 6) 뒤끝 정리
popd >/dev/null
rm -rf "$TMP"

echo "✅ jethexa.xml 이 현재 디렉토리에 생성되었습니다."

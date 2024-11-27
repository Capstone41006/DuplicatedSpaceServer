#!/bin/bash
# Conda 초기화 스크립트 로드
source ~/anaconda3/etc/profile.d/conda.sh

conda activate dust3r

mv ./result_object/input.ply ./scripts/

# preprocessing - blackpoints remove
python ./scripts/plyFilter.py
python ./scripts/ply2mesh_open3d.py

# train.py 실행
blender --background --python ./scripts/ply2All.py -- ./scripts/mesh.ply ./result_object/output.fbx ./result_object/output_texture.png

# 재사용을 위해 중간에 처리했던 ply 파일들 삭제
rm ./scripts/input.ply ./scripts/filtered.ply ./scripts/mesh.ply

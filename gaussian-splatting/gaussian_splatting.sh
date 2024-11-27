#!/bin/bash

# 입력 매개변수 확인
if [ $# -eq 0 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

# 사용자로부터 받은 이름으로 작업 수행
name=$1

# 프로젝트 디렉토리로 이동
cd ~/DuplicatedSplaceServer/gaussian-splatting/

# convert.py 실행
python convert.py -s "data/$name"

# train.py 실행
python train.py -s "data/$name" --eval

# 결과물 옮기기
SOURCE_FILE=$(find ./output -type f -path "*/point_cloud/iteration_3000/point_cloud.ply")
TARGET_DIR="../Server/result_space"

echo "result move"
mv "$SOURCE_FILE" "$TARGET_DIR/point_cloud.ply"

# 다음 사용을 위해 초기화
rm -r ./data/capstone/*
mkdir ./data/capstone/input/
rm -r ./output/*

#! /bin/bash

GPU_ID=0
DATA_ROOT_DIR="$HOME/DuplicatedSplaceServer/InstantSplat/data"
DATASETS=(
    # TT
    # sora
    # mars
    # sumin
    capstone
    )

SCENES=(
    # Family
    # Barn
    # Francis
    # Horse
    # Ignatius
    # santorini
    # agumon
    capstone
    )


# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=1000
pose_lr=1x

# 데이터셋과 씬을 순회
for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do

        # 파일을 세고자 하는 디렉토리 경로 설정
        # 'desired_subdir'을 실제 파일들이 있는 하위 디렉토리로 변경하세요
        FILE_DIR="${DATA_ROOT_DIR}/TEMP/images"

        # 디렉토리가 존재하는지 확인
        if [ ! -d "$FILE_DIR" ]; then
            echo "디렉토리 $FILE_DIR 이 존재하지 않습니다. 스킵합니다."
            continue
        fi

        # 디렉토리 내 파일 개수 세기
        # 특정 파일 유형만 세고 싶다면 -name 옵션을 추가하세요 (예: '*.jpg')
        N_VIEW=$(find "$FILE_DIR" -type f | wc -l)

        echo "데이터셋: $DATASET, 씬: $SCENE, 뷰 개수: $N_VIEW"

        # 경로 설정
        SOURCE_PATH="${DATA_ROOT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views"
        MODEL_PATH="./output/infer/${DATASET}/${SCENE}/${N_VIEW}_views_${gs_train_iter}Iter_${pose_lr}PoseLR/"
        DEST_PATH="${SOURCE_PATH}/images"

        RESULT_DIR="./output/infer/${DATASET}/${SCENE}/${N_VIEW}_views_${gs_train_iter}Iter_${pose_lr}PoseLR/input.ply"     # 나중에 결과물로 바꿀 것
        DEST_PATH2="../Server/result_object/"

        # 대상 디렉토리 생성 (존재하지 않을 경우)
        if [ ! -d "$DEST_PATH" ]; then
            mkdir -p "$DEST_PATH"
        fi

        # 파일 이동 및 원본 파일 삭제
        echo "Image moving: ${FILE_DIR} to ${DEST_PATH}..."
        mv "$FILE_DIR"/* "$DEST_PATH"/

        # ----- (1) Dust3r_coarse_geometric_initialization -----
        CMD_D1="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./coarse_init_infer.py \
        --img_base_path ${SOURCE_PATH} \
        --n_views ${N_VIEW}  \
        --focal_avg \
        "

        # ----- (2) Train: jointly optimize pose -----
        CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train_joint.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH}  \
        --n_views ${N_VIEW}  \
        --scene ${SCENE} \
        --iter ${gs_train_iter} \
        --optim_pose \
        "

        # ----- (3) Render interpolated pose & output video -----
        CMD_RI="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./render_by_interp.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH}  \
        --n_views ${N_VIEW}  \
        --scene ${SCENE} \
        --iter ${gs_train_iter} \
        --eval \
        --get_video \
        "

        # 명령어 실행
        echo "========= ${SCENE}: Dust3r_coarse_geometric_initialization ========="
        eval $CMD_D1
        echo "========= ${SCENE}: Train: jointly optimize pose ========="
        eval $CMD_T
        # echo "========= ${SCENE}: Render interpolated pose & output video ========="
        # eval $CMD_RI

        # 결과물 이동 및 이전 데이터 정리
        echo "Result moving: ${RESULT_DIR} to ${DEST_PATH2}..."
        cp "$RESULT_DIR" "$DEST_PATH2"/
        rm -r "$SOURCE_PATH" "$MODEL_PATH"
        
    done
done


#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ### Feature extraction - Sift 추출
    ### $ colmap feature_extractor --database_path data/$name/distorted/database.db --image_path data/$name/input
    ### $ --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)                # colmap을 사용하기 위해 터미널에 입력되는 코드
    exit_code = os.system(feat_extracton_cmd)                    # 터미널에 실행
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Feature matching - Sift 매칭
    ### $ colmap exhaustive_matcher --database_path data/$name/distorted/database.db --SiftExtraction.use_gpu 1
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)                  # colmap을 사용하기 위해 터미널에 입력되는 코드
    exit_code = os.system(feat_matching_cmd)                     # 터미널에서 실행
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment - Global 매칭
    ### $ colmap feature_extractor --database_path data/$name/distorted/database.db --image_path data/$name/input
    ### $ --output_path data/$name/distorted/sparse --Mapper.ba_global_function_tolerance=0.000001
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")          # colmap을 사용하기 위해 터미널에 입력되는 코드
    exit_code = os.system(mapper_cmd)                             # 터미널에서 실행
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion - 이미지 왜곡 해결
### $ colmap image_undistorter --image_path data/$name/input --input_path data/$name/distorted/sparse/0
### $ --output_path data/$name --output_type COLMAP
# We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")                                         # colmap을 사용하기 위해 터미널에 입력되는 코드
exit_code = os.system(img_undist_cmd)                              # 터미널에서 실행
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")                   # 특정 파일에 있는 목록을 모두 불러옴
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

############################### 생성 결과 #######################################
## distorted/sparse/0 안에 cameras.bin, images.bin, points3D.bin, project.ini ##
## distorted 안에 database.db                                                 ##
###############################################################################

#### ADDED ####
### Bin to TXT - 생성한 바이너리 파일을 TXT 파일로 생성(이후에 DUSt3R에서 이 포맷에 맞춰서 입력이 되게끔 할 예정)
### $ colmap model_converter --input_path data/test_0513/distorted/sparse/0 --output_path data/test_0513/distorted/sparse/0 --output_type TXT
# bin2txt_cmd = (colmap_command + " model_converter \
#     --input_path " + args.source_path + "/distorted/sparse/0 \
#     --output_path " + args.source_path + "/distorted/sparse/0 \
#     --output_type TXT")                                         # TXT로 변환하기 위한 코드
# exit_code = os.system(bin2txt_cmd)                              # 터미널에서 실행

# # 특정 파일에 있는 목록을 모두 불러옴
# bin_file_path = os.path.join(args.source_path, "distorted/sparse/0")
# files_in_directory = os.listdir(bin_file_path)

bin2txt_cmd = (colmap_command + " model_converter \
    --input_path " + args.source_path + "/sparse/0 \
    --output_path " + args.source_path + "/sparse/0 \
    --output_type TXT")                                         # TXT로 변환하기 위한 코드
exit_code = os.system(bin2txt_cmd)                              # 터미널에서 실행

# 특정 파일에 있는 목록을 모두 불러옴
bin_file_path = os.path.join(args.source_path, "sparse/0")
files_in_directory = os.listdir(bin_file_path)
# .bin 파일 필터링
bin_files = [file for file in files_in_directory if file.endswith(".bin")]
# .bin 파일 삭제
for file in bin_files:
    file_path = os.path.join(bin_file_path, file)
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")


# args.resize = false
if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")

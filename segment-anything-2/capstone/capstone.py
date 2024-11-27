import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from sam2.build_sam import build_sam2_video_predictor
import firebase_admin
from firebase_admin import credentials, db, storage

#### 파이어 베이스 설정 ####
cred = credentials.Certificate('***')
firebase_admin.initialize_app(cred, {
    'databaseURL': '***',
    'storageBucket': '***'
})

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    

##########################################
## Functions                            ##
##########################################
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def load_image_with_orientation(image_path):
    image = Image.open(image_path)
    original_size = image.size  # (width, height)
    
    # EXIF 데이터에서 방향 정보 가져오기
    try:
        exif = image._getexif()
        if exif is not None:
            orientation_key = [k for k, v in ExifTags.TAGS.items() if v == 'Orientation'][0]
            orientation = exif.get(orientation_key)
        else:
            orientation = None
    except (AttributeError, KeyError, TypeError):
        orientation = None

    rotation_angle = 0
    # EXIF 데이터를 기반으로 이미지 회전
    if orientation:
        if orientation == 3:
            image = image.rotate(180, expand=True)
            rotation_angle = 180
        elif orientation == 6:
            image = image.rotate(270, expand=True)
            rotation_angle = 270
        elif orientation == 8:
            image = image.rotate(90, expand=True)
            rotation_angle = 90

    rotated_size = image.size  # After rotation
    return image, rotation_angle, original_size, rotated_size

def rotate_normalized_points(points, rotation_angle):
    # points: numpy array of shape (N, 2), values in [0, 1]
    if rotation_angle == 0:
        x_new = points[:, 0]
        y_new = points[:, 1]
    elif rotation_angle == 90:
        # 90도 반시계 방향 회전
        x_new = points[:, 1]
        y_new = 1 - points[:, 0]
    elif rotation_angle == 180:
        # 180도 회전
        x_new = 1 - points[:, 0]
        y_new = 1 - points[:, 1]
    elif rotation_angle == 270:
        # 270도 반시계 방향 회전
        # x_new = 1 - points[:, 1]
        x_new = points[:, 0]
        y_new = points[:, 1]
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation_angle}")
    return np.stack([x_new, y_new], axis=1)


def rotate_mask(mask, angle):
    # 마스크 배열의 차원이 3차원이라면, 2차원으로 변환
    mask = np.squeeze(mask)  # 불필요한 차원 제거

    # 마스크를 PIL 이미지로 변환
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255))  # 마스크를 0~1에서 0~255 범위로 변환
    mask_image = mask_image.convert("L")  # 그레이스케일 이미지로 변환

    # 마스크를 이미지와 동일한 각도로 회전
    rotated_mask_image = mask_image.rotate(angle, expand=True)
    rotated_mask = np.array(rotated_mask_image) / 255  # 다시 0~1 범위로 변환
    return rotated_mask


def process_images_in_folder(input_folder, output_folder, point1, point2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 이미지 프레임 이름 목록 가져오기
    frame_names = [
        p for p in os.listdir(input_folder)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # 초기화
    inference_state = predictor.init_state(video_path=input_folder)
    predictor.reset_state(inference_state)

    # 첫 번째 프레임의 원본 이미지 크기 및 회전 각도 가져오기
    first_frame_path = os.path.join(input_folder, frame_names[0])
    frame_image, rotation_angle, original_size, rotated_size = load_image_with_orientation(first_frame_path)

    # 입력된 포인트 (정규화된 좌표)
    points_normalized = np.array([point1, point2], dtype=np.float32)
    labels = np.array([1, 1], np.int32)

    # Predictor에 전달할 포인트 (원본 이미지의 픽셀 좌표)
    points_pixel = points_normalized * np.array(original_size)

    # Predictor에 포인트 추가
    ann_frame_idx = 0
    ann_obj_id = 1
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_pixel,
        labels=labels,
    )

    # 비디오 내 모든 프레임에 대해 전파
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # 프레임에 대해 시각화 및 마스크 저장
    for out_frame_idx in range(len(frame_names)):
        if out_frame_idx in video_segments:
            frame_path = os.path.join(input_folder, frame_names[out_frame_idx])

            # 이미지 로드 및 회전 각도 가져오기
            frame_image, rotation_angle, original_size, rotated_size = load_image_with_orientation(frame_path)
            frame_image_np = np.array(frame_image)

            # 포인트를 회전 각도에 따라 변환 (정규화된 좌표)
            rotated_points_normalized = rotate_normalized_points(points_normalized, rotation_angle)

            # 회전된 이미지의 크기를 사용하여 픽셀 좌표로 변환
            rotated_points_pixel = rotated_points_normalized * np.array(rotated_size)

            # 마스크 회전
            masks = []
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                rotated_mask = rotate_mask(out_mask, rotation_angle)
                masks.append(rotated_mask)

            # 시각화
            # plt.figure(figsize=(6, 4))
            # plt.title(f"frame {out_frame_idx}")
            # plt.imshow(frame_image_np)

            # 포인트 시각화
            show_points(rotated_points_pixel, labels, plt.gca(), marker_size=200)

            # 마스크 시각화
            # for mask in masks:
            #     show_mask(mask, plt.gca(), obj_id=out_obj_id)
	    #
            # plt.show()

            # 마스크 저장
            save_masks(frame_image_np, masks, output_folder, frame_names[out_frame_idx])
            os.remove(frame_path)


def save_masks(image, out_masks, output_folder, original_filename):
    # 마스크 처리
    # black_background = np.zeros_like(image)
    # 투명 배경을 적용하기 위해 RGBA로 이미지 변환
    rgba_image = np.dstack((image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255))  # 알파 채널 추가

    # Iterate through each mask and save
    for i, out_mask in enumerate(out_masks):
        # 마스크의 차원이 3차원인 경우 2차원으로 축소
        if len(out_mask.shape) == 3 and out_mask.shape[0] == 1:
            out_mask = out_mask.squeeze(0)

        # 마스크를 사용하여 이미지와 배경을 결합
        mask_expanded = out_mask[:, :, np.newaxis]  # (height, width, 1)
        #masked_image = np.where(mask_expanded, image, black_background)
        # 알파 채널을 사용해 배경을 투명하게 처리
        masked_image = np.where(mask_expanded, rgba_image, [0, 0, 0, 0])

        # PIL 이미지로 변환
        #masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))
        masked_image_pil = Image.fromarray(masked_image.astype(np.uint8), 'RGBA')

        # 파일 이름 수정 및 저장
        base_filename, ext = os.path.splitext(original_filename)
        output_filename = f"{base_filename}_masked_{i}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        masked_image_pil.save(output_path)

def clear_gpu_memory():
    # 명시적으로 메모리에서 변수 삭제
    del sam2_model
    del predictor
    torch.cuda.empty_cache()  # 캐시된 메모리 해제
    
    
    
    
##########################################
## Main Code                            ##
##########################################
ref_x1 = db.reference('/ref_x1')
ref_y1 = db.reference('/ref_y1')
ref_x2 = db.reference('/ref_x2')
ref_y2 = db.reference('/ref_y2')
ref_t = db.reference('/ref_t')

x1 = ref_x1.get()
y1 = ref_y1.get()
x2 = ref_x2.get()
y2 = ref_y2.get()
start_time = ref_t.get()

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

input_folder = "./videos/TEST/input"
output_folder = "./videos/TEST/output"
point1 = [x1,y1]
point2 = [x2,y2]
print("point1: ", point1)
print("point2: ", point2)
# point1 = [0.7,0.25]
# point2 = [0.5,0.25]
process_images_in_folder(input_folder, output_folder, point1, point2)
print("## Background Remove Done ##\n")

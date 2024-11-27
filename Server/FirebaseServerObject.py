import numpy as np
import firebase_admin
from firebase_admin import credentials, db, storage
import subprocess, time, os
from PIL import Image
os.environ["MKL_THREADING_LAYER"] = "GNU"	# 현재 환경에서 이거 없으면 에러나고 있음

cred = credentials.Certificate('***')
firebase_admin.initialize_app(cred, {
    'databaseURL': '***',
    'storageBucket': '***'
})


def download_images1():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="Images/")  # Image 폴더 안의 모든 파일을 가져옴
    image_count = 0
    
    for index, blob in enumerate(blobs):
        file_extension = blob.name.split('.')[-1]  # 파일 이름에서 확장자 추출
        destination_filename = f"../InstantSplat/data/TEMP/images/image{index}.{file_extension}"
        blob.download_to_filename(destination_filename)
        print(f"Downloaded {blob.name} from Firebase Storage as {destination_filename}")
        image_count += 1
    
    return image_count
    

def download_video_object():
	bucket = storage.bucket()
	blobs = bucket.list_blobs(prefix="Video_Object/")  # Video 폴더 안의 모든 파일을 가져옴
	image_count = 0
	file_extension = 'mp4'
	
	for index, blob in enumerate(blobs):
		file_name = os.path.basename(blob.name)  # 파일 이름만 추출 (경로 제외)
		file_extension = file_name.split('.')[-1] if '.' in file_name else ''  # 확장자가 있을 때만 추출
		if not file_extension:  # 확장자가 없을 경우 오류 방지
			print(f"Warning: {blob.name} has no file extension, skipping...")
			continue
    	
		file_name = os.path.basename(blob.name)
		file_extension = file_name.split('.')[-1] if '.' in file_name else 'mp4'
		destination_filename = f"../segment-anything-2/capstone/videos/TEST/raw_video/input_video.{file_extension}"
        
        # 파일 다운로드
		try:
			print(f"Downloading {blob.name} to {destination_filename}.{file_extension}")
			blob.download_to_filename(destination_filename)
			print(f"Successfully downloaded {blob.name}.")
		except Exception as e:
			print(f"Failed to download {blob.name}: {e}")
        
	return file_extension
        
        
def process_video(file_extension):			# 정밀화 작업 필요!!!!!
	destination_filename = f"../segment-anything-2/capstone/videos/TEST/raw_video/input_video.{file_extension}"
    # 동영상 파일일 경우 FFmpeg 명령어 실행
	if file_extension == "mp4":
		ffmpeg_command = [
        "ffmpeg", "-i", destination_filename, "-r", "1", "-q:v", "2", 
        "-start_number", "0", "../segment-anything-2/capstone/videos/TEST/input/%05d.jpg"
        ]
		subprocess.run(ffmpeg_command)
		print(f"FFmpeg command executed: {' '.join(ffmpeg_command)}")
    
    # 경로에 있는 이미지 파일 수 세기
	image_dir = "../segment-anything-2/capstone/videos/TEST/input"
	image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]  # 이미지 확장자 필터링
	image_count = len(image_files)
    
	return image_count
    

# def download_images():
#     bucket = storage.bucket()
#     for i in range(1, 6):
#         image_blob = bucket.blob(f"image{i}.jpg")
#         image_blob.download_to_filename(f"../../gaussian-splatting/data/test2/input/image{i}.jpg")
#         print("Download images from a Firebase Storage")


def delete_video(start_time):
    image_dir = "../segment-anything-2/capstone/videos/TEST/input"
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg'))]  # 이미지 확장자 필터링
    
    remove_cnt = 0
    
    # 파일 삭제
    for image_file in image_files:
        # 파일 이름에서 숫자 추출 (예: 00001.jpg -> 00001)
        file_number = int(os.path.splitext(image_file)[0])

        # 숫자가 ref_t보다 작으면 삭제
        if file_number < start_time:
            file_path = os.path.join(image_dir, image_file)
            os.remove(file_path)
            remove_cnt += 1
            # print(f"Deleted {file_path}")
            
    return remove_cnt
            
            
def delete_transparent_images():
    image_dir = "../segment-anything-2/capstone/videos/TEST/output"
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]  # PNG 파일만 필터링
    
    remove_cnt = 0
    
    # 파일 삭제
    for image_file in image_files:
        file_path = os.path.join(image_dir, image_file)
        
        # 이미지 열기
        with Image.open(file_path) as img:
            img = img.convert("RGBA")  # RGBA 모드로 변환
            rgba_data = np.array(img)  # 이미지를 NumPy 배열로 변환
            
            # 이미지가 전체적으로 [0, 0, 0, 0]인지 확인
            if np.all(rgba_data == [0, 0, 0, 0]):
                os.remove(file_path)
                remove_cnt += 1
                # print(f"Deleted {file_path}")
    
    return remove_cnt


def move_sam2_result():
    # 현재 스크립트의 위치를 기준으로 경로 설정
    source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../segment-anything-2/capstone/videos/TEST/output/'))
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../InstantSplat/data/TEMP/images'))
    
    # 목적지 디렉토리가 없으면 생성
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # source_dir의 모든 파일을 iterating
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        
        # 파일인지 확인하고 이동
        if os.path.isfile(source_file):
            os.rename(source_file, dest_file)
            # print(f"Moved: {source_file} -> {dest_file}")
    

	  
def upload_point_cloud_space(image_count):
    bucket = storage.bucket() # Storage 클라이언트 가져오기
    
    # 파일 이름 설정
    storage_file_name = 'point_cloud_space.ply' # Firebase Storage에 저장될 파일 이름
    local_file_path = f'../InstantSplat/output/infer/capstone/test2/{image_count}_views_1000Iter_1xPoseLR/point_cloud/iteration_1000/point_cloud.ply'  # 업로드할 로컬 파일 경로 및 이름
	
    # 파일 업로드
    blob = bucket.blob(storage_file_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to Firebase Storage as {storage_file_name}")
    
    

def upload_point_cloud_object(image_count, obj_count):
    bucket = storage.bucket() # Storage 클라이언트 가져오기
    
    # 파일 이름 설정
    storage_file_name = f'point_cloud_obj{obj_count}.ply' # Firebase Storage에 저장될 파일 이름
    local_file_path = f'../InstantSplat/output/infer/capstone/test2/{image_count}_views_1000Iter_1xPoseLR/point_cloud/iteration_1000/point_cloud.ply'  # 업로드할 로컬 파일 경로 및 이름
	
    # 파일 업로드
    blob = bucket.blob(storage_file_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to Firebase Storage as {storage_file_name}")
    

def upload_object(obj_count):
    bucket = storage.bucket() # Storage 클라이언트 가져오기
    
    # 파일 이름 설정
    storage_file_name = f'Objects/obj{obj_count}.fbx' # Firebase Storage에 저장될 파일 이름
    local_file_path = f'./result_object/output.fbx'  # 업로드할 로컬 파일 경로 및 이름
    storage_file_name2 = f'Objects/texture{obj_count}.png' # Firebase Storage에 저장될 파일 이름
    local_file_path2 = f'./result_object/output_texture.png'  # 업로드할 로컬 파일 경로 및 이름
	
    # 파일 업로드
    blob = bucket.blob(storage_file_name)
    blob.upload_from_filename(local_file_path)
    blob2 = bucket.blob(storage_file_name2)
    blob2.upload_from_filename(local_file_path2)
    print(f"Uploaded {local_file_path} to Firebase Storage as {storage_file_name}")
    print(f"Uploaded {local_file_path2} to Firebase Storage as {storage_file_name2}")


def monitor_database():
	ref_inputs = db.reference('/inputs')
	ref_modeObject = db.reference('/modeObject')
	ref_isMadeObject = db.reference('/isMadeObject')
	ref_isTrain = db.reference('/isTrain')
    
	processing_ref_x1 = db.reference('/ref_x1')
	processing_ref_y1 = db.reference('/ref_y1')
	processing_ref_x2 = db.reference('/ref_x2')
	processing_ref_y2 = db.reference('/ref_y2')
	processing_ref_t = db.reference('/ref_t')
	
	raw_video_dir = '../segment-anything-2/capstone/videos/TEST/raw_video/input_video.mp4'
	fbx_dir = './result_object/output.fbx'
	png_dir = './result_object/output_texture.png'
    
	while True:
		is_train = ref_isTrain.get()
		is_obejct_train = ref_modeObject.get()

		if (is_train == False) and (is_obejct_train == True):			# 오브젝트 학습 플래그 ON, 학습 중 X일 때
			ref_isTrain.set(True)	    # isTrain 상태를 false로 변경
			print("\n#### SAM2 & InstantSplat Operating - Object Making ####\n")
			
			inputs = ref_inputs.get()
			input_count = len(inputs)				# 입력받은 입력 값들 집합의 갯수
			obj_count = 0
			print(f"I found {input_count} inputs")
        		
			# download video
			time.sleep(5)
			print("\n--- Donwload Video ----\n")
			file_extension = download_video_object()
        		
			for input_key, input_value in inputs.items():		# inputs 자식의 수 만큼 반복
				ref_x1 = input_value.get('ref_x1')
				ref_y1 = input_value.get('ref_y1')
				ref_x2 = input_value.get('ref_x2')
				ref_y2 = input_value.get('ref_y2')
				ref_t = input_value.get('ref_t')
				processing_ref_x1.set(ref_x1)					# SAM2에서 참조할 값
				processing_ref_y1.set(ref_y1)
				processing_ref_x2.set(ref_x2)
				processing_ref_y2.set(ref_y2)
				processing_ref_t.set(ref_t)
        			
				# preprocessing 1 - 입력받은 시간부터의 이미지를 처리
				print("\n--- Collecting Object Images ----\n")
				image_count = process_video(file_extension)
				image_count -= delete_video(ref_t)				# ref_t를 참조하여 이미지 삭제
				print(f"Founded Image Count = {image_count}")
        		
				# SAM2
				print("\n## ==== Process 1 - SAM2 ==== ##")
				subprocess.run(['./capstone/SAM2.sh'], cwd='../segment-anything-2')
				print("## ==== Process 1 Done ==== ##\n")
        		
				# preprocessing 2 - 객체가 인식 안된 것 삭제
				print("\n--- Extracting Only Object Images ----\n")
				image_count -= delete_transparent_images()
				print("\n--- Moving Images for Instantsplat ----\n")
				move_sam2_result()
        		
				# InstantSplat
				print("\n## ==== Process 2 - InstantSplat ==== ##")
				subprocess.run(['./scripts/run_train_infer_capstone.sh'], cwd='../InstantSplat')
				print("## ==== Process 2 Done ==== ##\n")
        		
				# PLY to FBX
				print("\n## ==== Process 3 - PLY to FBX ==== ##")
				subprocess.run(['./ply2fbx.sh'])
				print("## ==== Process 3 Done ==== ##\n")
        	
				# 결과 업로드
				print("\n## ==== Process 4 - Upload to Firebase ==== ##")
				obj_count += 1
				# upload_point_cloud_object(image_count, obj_count)
				upload_object(obj_count)
				print("## ==== Process 4 Done ==== ##\n")
        		
        		# postprocess
			ref_modeObject.set(False)	  # Mode 상태를 false로 변경
			ref_isMadeObject.set(True)    # Made 상태를  true로 변경
			ref_isTrain.set(False)	      # isTrain 상태를 false로 변경
			os.remove(raw_video_dir)
			os.remove(fbx_dir)
			os.remove(png_dir)
			
			print("\n#### Object Making Finished ####\n")
        	
		else:
			time.sleep(3)  # 3초마다 Realtime Database를 확인
			print("Waiting Firebase Server--")


if __name__ == "__main__":
	monitor_database()

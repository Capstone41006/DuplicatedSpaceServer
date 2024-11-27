from plyfile import PlyData, PlyElement
import numpy as np
from sklearn.cluster import DBSCAN

def filter_largest_cluster(input_ply_path, output_ply_path, eps=0.02, min_samples=10):
    # PLY 파일 읽기
    ply_data = PlyData.read(input_ply_path)

    # PLY 파일의 vertices (점) 데이터 가져오기
    vertex_data = ply_data['vertex'].data
    coords = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)

    # 각 vertex의 RGB 컬러 정보 추출
    r = vertex_data['red']
    g = vertex_data['green']
    b = vertex_data['blue']

    # 검은색인 포인트 (R=0, G=0, B=0)를 제외한 인덱스 선택
    non_black_indices = np.where((r != 0) | (g != 0) | (b != 0))[0]

    # 검은색이 아닌 포인트들만 필터링
    filtered_vertex_data = vertex_data[non_black_indices]
    filtered_coords = coords[non_black_indices]

    # DBSCAN을 사용하여 포인트들을 클러스터링
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_coords)
    labels = db.labels_

    # -1은 노이즈, 즉 클러스터에 속하지 않는 포인트
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]  # 가장 큰 클러스터의 라벨

    # 가장 큰 클러스터에 속하는 포인트들만 선택
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # 최종 필터링된 포인트 데이터
    final_filtered_vertex_data = filtered_vertex_data[largest_cluster_indices]

    # 새로운 PLY 파일 생성
    new_vertex_element = PlyElement.describe(final_filtered_vertex_data, 'vertex')

    # 수정된 데이터로 새로운 PLY 파일 저장
    PlyData([new_vertex_element], text=True).write(output_ply_path)

    print(f"Filtered PLY saved to {output_ply_path}. Removed {len(vertex_data) - len(final_filtered_vertex_data)} points.")

# 사용 예시
input_ply = './scripts/input.ply'
output_ply = './scripts/filtered.ply'

filter_largest_cluster(input_ply, output_ply, eps=0.02, min_samples=10)


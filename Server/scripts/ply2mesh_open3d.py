import open3d as o3d
from tqdm import tqdm
import time
import numpy as np

# 진행률 리스트 정의
stages = ["Pointcloud Load", "Calculate Normals", "Making Mesh by Using Poisson", "Mesh Smoothing", "Saving PLY"]

# tqdm으로 단계별 진행률 표시
with tqdm(total=len(stages), desc="processing", unit="phase") as pbar:
    # 1. PLY 파일에서 점 구름 읽기
    pcd = o3d.io.read_point_cloud("./scripts/filtered.ply")
    pbar.update(1)  # 진행률 업데이트

    # 2. 법선 계산이 필요함
    # print("법선 계산 중...")
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(120)
    pbar.update(1)  # 진행률 업데이트

    # 3. 삼각형 메쉬 생성 (Poisson 알고리즘 사용)
    # print("Poisson 알고리즘으로 삼각형 메쉬 생성 중...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    pbar.update(1)  # 진행률 업데이트
    
    # 4. 메쉬 스무딩 및 간소화
    # print("메쉬 스무딩 및 간소화 중...")
    # 밀도가 낮은 정점 제거 (하위 1% 제거)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    # 스무딩 적용
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    # 메쉬 간소화 (삼각형 개수를 절반으로 줄임)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=len(mesh.triangles) // 1)
    pbar.update(1)  # 진행률 업데이트

    # 6. 메쉬 파일로 저장 (PLY 형식)
    o3d.io.write_triangle_mesh("./scripts/mesh.ply", mesh)
    # print("PLY 파일로 저장 완료!")
    pbar.update(1)  # 진행률 업데이트


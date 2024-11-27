import bpy
import sys
import os
import mathutils

# 명령줄에서 인자로 받은 PLY 파일 경로와 출력 파일 경로들
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # "--" 이후의 인자들만 사용
input_ply = argv[0]  # PLY 파일 경로
output_fbx = argv[1]  # FBX 파일 경로
output_texture = argv[2]  # 생성될 텍스처 파일 경로 (PNG)

# 기존 데이터 삭제 (초기화)
bpy.ops.wm.read_factory_settings(use_empty=True)

# PLY 파일 불러오기
bpy.ops.import_mesh.ply(filepath=input_ply)

# 현재 활성 객체 가져오기
obj = bpy.context.selected_objects[0]
mesh = obj.data

# UV 맵 생성 (프로그래밍 방식으로 UV 좌표 할당)
if not mesh.uv_layers:
    mesh.uv_layers.new(name='UVMap')

# UV 레이어 접근
uv_layer = mesh.uv_layers.active.data

# 모든 정점의 최소 및 최대 X, Y 좌표를 계산하여 UV 좌표로 변환
min_x = min((v.co.x for v in mesh.vertices))
max_x = max((v.co.x for v in mesh.vertices))
min_y = min((v.co.y for v in mesh.vertices))
max_y = max((v.co.y for v in mesh.vertices))
range_x = max_x - min_x
range_y = max_y - min_y

# UV 좌표 할당
for poly in mesh.polygons:
    for loop_index in poly.loop_indices:
        loop = mesh.loops[loop_index]
        vertex = mesh.vertices[loop.vertex_index]
        uv = uv_layer[loop_index].uv
        uv[0] = (vertex.co.x - min_x) / range_x if range_x != 0 else 0.0
        uv[1] = (vertex.co.y - min_y) / range_y if range_y != 0 else 0.0

# 재질 및 노드 설정
material = bpy.data.materials.new(name="Material")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links

# 기본 노드 제거
nodes.clear()

# 필요한 노드 추가
output_node = nodes.new(type='ShaderNodeOutputMaterial')
bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
vertex_color_node.layer_name = mesh.vertex_colors.active.name

# 노드 연결
links.new(vertex_color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

# 객체에 재질 할당
mesh.materials.append(material)

# 베이크를 위한 이미지 생성
image = bpy.data.images.new("BakedTexture", width=1024, height=1024)
image.filepath = output_texture
image.file_format = 'PNG'

# 이미지 텍스처 노드 생성 및 활성화
image_texture_node = nodes.new(type='ShaderNodeTexImage')
image_texture_node.image = image
image_texture_node.select = True
nodes.active = image_texture_node

# 렌더 엔진 설정
bpy.context.scene.render.engine = 'CYCLES'

# 베이킹 설정
bpy.context.scene.cycles.device = 'GPU'  # CPU 사용 (필요 시 GPU로 변경 가능)
bpy.context.scene.cycles.samples = 1  # 샘플 수 설정 (속도 향상)
bpy.context.scene.render.bake.use_selected_to_active = False
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.use_pass_color = True
bpy.context.scene.cycles.bake_type = 'DIFFUSE'

# 객체 활성화 및 선택
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# 베이크 실행
bpy.ops.object.bake(type='DIFFUSE')

# 베이킹된 이미지 저장
image.save()

# 베이킹 후 재질 업데이트 (베이킹된 텍스처 사용)
links.new(image_texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])
nodes.remove(vertex_color_node)

# FBX 파일로 내보내기
bpy.ops.export_scene.fbx(filepath=output_fbx, embed_textures=False)

print(f"FBX export done: {output_fbx}")
print(f"Texture export done: {output_texture}")


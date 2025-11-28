import trimesh
import numpy as np
import pandas as pd
import os
import random

def compute_orientation(normal):
    n = normal / np.linalg.norm(normal)
    dip = np.degrees(np.arccos(np.abs(n[2])))
    theta = np.degrees(np.arctan2(n[1], n[0]))
    azimuth = (270 - theta) % 360
    return round(azimuth), round(dip)

def sample_surface_with_curvature(mesh, sample_count, base_direction=None):
    vertex_normals = mesh.vertex_normals
    faces = mesh.faces
    face_normals = mesh.face_normals

    face_weights = np.zeros(len(faces))
    for i, f in enumerate(faces):
        n0, n1, n2 = vertex_normals[f]
        variation = np.linalg.norm(n0 - n1) + np.linalg.norm(n1 - n2) + np.linalg.norm(n2 - n0)
        face_weights[i] = variation

    if np.sum(face_weights) == 0:
        face_weights[:] = 1.0

    face_probs = face_weights / np.sum(face_weights)
    tri_indices = np.random.choice(len(faces), size=sample_count, replace=True, p=face_probs)
    triangles = mesh.triangles[tri_indices]

    u = np.random.rand(sample_count, 1)
    v = np.random.rand(sample_count, 1)
    is_over = u + v > 1
    u[is_over] = 1 - u[is_over]
    v[is_over] = 1 - v[is_over]
    w = 1 - u - v
    sampled_points = u * triangles[:, 0] + v * triangles[:, 1] + w * triangles[:, 2]
    sampled_normals = face_normals[tri_indices]

    # 统一法向量方向（如果给了基准方向）
    if base_direction is not None:
        aligned_normals = []
        for n in sampled_normals:
            n_norm = n / np.linalg.norm(n)
            if np.dot(n_norm, base_direction) < 0:
                n_norm = -n_norm
            aligned_normals.append(n_norm)
        sampled_normals = np.array(aligned_normals)

    return sampled_points, sampled_normals

def sample_fault_surface_top_down(mesh, sample_count, layer_count=1):
    face_centroids = mesh.triangles_center
    face_normals = mesh.face_normals
    z_values = face_centroids[:, 2]

    z_min, z_max = z_values.min(), z_values.max()
    layer_bounds = np.linspace(z_max, z_min, layer_count + 1)

    sampled_points = []
    sampled_normals = []

    base_direction = None

    for i in range(layer_count):
        z_top, z_bottom = layer_bounds[i], layer_bounds[i + 1]
        in_layer = (z_values <= z_top) & (z_values > z_bottom)
        indices = np.where(in_layer)[0]

        if len(indices) == 0:
            continue

        count_per_layer = sample_count // layer_count
        chosen = np.random.choice(indices, size=min(count_per_layer, len(indices)), replace=False)
        sampled = mesh.triangles_center[chosen]
        normals = face_normals[chosen]

        if base_direction is None and len(normals) > 0:
            base_direction = normals[0] / np.linalg.norm(normals[0])

        # 统一方向
        aligned_normals = []
        for n in normals:
            n_norm = n / np.linalg.norm(n)
            if np.dot(n_norm, base_direction) < 0:
                n_norm = -n_norm
            aligned_normals.append(n_norm)

        sampled_points.append(sampled)
        sampled_normals.append(np.array(aligned_normals))

    return np.vstack(sampled_points), np.vstack(sampled_normals), base_direction

def process_stl_batch(
        stl_paths,
        formation_names,
        output_dir="./output/gempy_batch_combined",
        polarity=1,
        use_curvature_weighted=True,
        fault_mode=False,
        random_seed=123
):
    os.makedirs(output_dir, exist_ok=True)
    all_surface = []
    all_orient = []

    np.random.seed(random_seed)
    random.seed(random_seed)

    for idx, (stl_path, formation_name) in enumerate(zip(stl_paths, formation_names), start=1):
        print(f"\n🚀 处理第 {idx} 个：{formation_name}")
        mesh = trimesh.load(stl_path)
        mesh.fix_normals()

        surface_count = np.random.randint(500, 800)
        orientation_count = np.random.randint(50, min(200, surface_count))

        print(f"📐 surface 点数: {surface_count} | orientation 点数: {orientation_count}")

        if fault_mode:
            sampled_points, sampled_normals, base_dir = sample_fault_surface_top_down(mesh, surface_count)
            # 曲率采样时用断层采样的方向统一法向量
            # 如果你想对曲率采样也做法向量统一，可以传 base_dir 过去
        elif use_curvature_weighted:
            # 先用故障采样获得基准方向（可选）
            base_dir = None
            sampled_points, sampled_normals = sample_surface_with_curvature(mesh, surface_count, base_direction=base_dir)
        else:
            sampled_points, face_index = trimesh.sample.sample_surface_even(mesh, count=surface_count)
            sampled_normals = mesh.face_normals[face_index]
            base_dir = None

        if len(sampled_points) < orientation_count:
            print(f"⚠️ 警告：生成的 sampled_points 数量不足（仅 {len(sampled_points)} 个），调整 orientation_count。")
            orientation_count = len(sampled_points)

        selected_indices = sorted(random.sample(range(len(sampled_points)), orientation_count))

        for i in range(len(sampled_points)):
            x, y, z = sampled_points[i]
            x, y, z = round(x, 2), round(y, 2), round(z, 2)
            all_surface.append([x, y, z, formation_name])

            if i in selected_indices:
                azimuth, dip = compute_orientation(sampled_normals[i])
                all_orient.append([x, y, z, azimuth, dip, polarity, formation_name])

    # 一次性保存所有合并数据，避免覆盖
    df_surface = pd.DataFrame(all_surface, columns=["X", "Y", "Z", "formation"])
    df_orientation = pd.DataFrame(all_orient, columns=["X", "Y", "Z", "azimuth", "dip", "polarity", "formation"])

    surface_path = os.path.join(output_dir, "surface_points.csv")
    orient_path = os.path.join(output_dir, "orientations.csv")

    df_surface.to_csv(surface_path, index=False, float_format='%.2f', encoding='utf-8')
    df_orientation.to_csv(orient_path, index=False, float_format='%.2f', encoding='utf-8')

    print(f"\n✅ 全部完成！合并输出：\n📄 surface: {surface_path}\n📄 orientation: {orient_path}")


if __name__ == "__main__":
    stl_paths = [
        "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/Loop_example/ground1.stl",
        "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/Loop_example/ground2.stl",
        "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/Loop_example/ground3.stl",
        "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/Loop_example/fault.stl",
        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/diqian/fault2.stl",

        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/silver_star/SiltyClay.stl",
        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/silver_star/FinegrainedSand.stl",
        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/silver_star/Mudstone1.stl",
        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/silver_star/Coal.stl",
        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/silver_star/CoarsegrainedSand.stl",
        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/silver_star/Mudstone2.stl",
        # "./data_from_real/7_models_with_faults_y_easy/simple_test/paper/silver_star/Basement.stl",

    ]

    formation_names = ['ground1','ground2','ground3','fault']
    #
    process_stl_batch(
        stl_paths=stl_paths,
        formation_names=formation_names,
        output_dir="./output/gempy_batch_combined/simple_test/paper/Loop_example",
        polarity=1,
        use_curvature_weighted=True,
        fault_mode=True ,  # 断层模式
        random_seed=10
    )

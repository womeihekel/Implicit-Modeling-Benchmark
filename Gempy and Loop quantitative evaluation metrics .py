import numpy as np
import pyvista as pv
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import trimesh
from collections import defaultdict
import os
import warnings

# ========== 字体设置 (Arial) ==========
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

warnings.filterwarnings("ignore", message="The current behavior of `pv.PolyData.n_faces` has been deprecated.")

# 配色方案
COLORS = {
    'gempy': '#5F9ED1',  # GemPy 蓝色
    'loop': '#59A14F',  # Loop 绿色
    'neutral': '#443988',  # 深紫
    'accent': '#ff3f20'  # 强调色
}


class LayerWiseModelComparison:
    """分层地质建模对比分析类 - 增强断层分析"""

    def __init__(self, ground_truth_paths, gempy_paths, loop_paths, model_type='graben'):
        self.model_type = model_type
        self.ground_truth = {}
        self.gempy_models = {}
        self.loop_models = {}

        print("正在加载模型...")
        loaded_keys = []
        for name in ground_truth_paths.keys():
            print(f"  加载 {name}...")
            try:
                gt_path = ground_truth_paths.get(name)
                gempy_path = gempy_paths.get(name)
                loop_path = loop_paths.get(name)

                if gt_path and os.path.exists(gt_path):
                    self.ground_truth[name] = self._load_mesh(gt_path)
                if gempy_path and os.path.exists(gempy_path):
                    self.gempy_models[name] = self._load_mesh(gempy_path)
                if loop_path and os.path.exists(loop_path):
                    self.loop_models[name] = self._load_mesh(loop_path)

                if name in self.ground_truth and name in self.gempy_models and name in self.loop_models:
                    loaded_keys.append(name)
                else:
                    print(f"  [警告] {name} 的文件不完整，将跳过此层。")

            except Exception as e:
                print(f"  [错误] 加载 {name} 失败: {e}")

        all_keys_sorted = sorted(list(ground_truth_paths.keys()))
        valid_keys_sorted = [k for k in all_keys_sorted if k in loaded_keys]

        self.layers = [k for k in valid_keys_sorted if 'ground' in k.lower()]
        self.faults = [k for k in valid_keys_sorted if 'fault' in k.lower()]
        self.all_layers = self.layers + self.faults

        if not self.all_layers:
            raise FileNotFoundError("没有找到任何匹配的文件组。")

        self.results = defaultdict(dict)

        print(f"模型加载完成！")
        print(f"  地层: {self.layers}")
        print(f"  断层: {self.faults}")

    def _load_mesh(self, path):
        """加载网格文件"""
        if path.endswith('.stl'):
            mesh = trimesh.load_mesh(path)
            return pv.wrap(mesh)
        elif path.endswith(('.vtp', '.vtk')):
            return pv.read(path)
        else:
            raise ValueError(f"不支持的文件格式: {path}")

    def _sample_points_from_mesh(self, mesh, n_points):
        """从网格采样点"""
        if hasattr(mesh, 'points'):
            points = np.array(mesh.points)
            if len(points) == 0:
                raise ValueError("网格中没有点")
            if len(points) > n_points:
                indices = np.random.choice(len(points), n_points, replace=False)
                return points[indices]
            else:
                return points
        else:
            raise ValueError("无法从网格采样点")

    def compute_hausdorff_distance(self, mesh1, mesh2, sample_points=10000):
        """计算Hausdorff距离"""
        try:
            points1 = self._sample_points_from_mesh(mesh1, sample_points)
            points2 = self._sample_points_from_mesh(mesh2, sample_points)

            hausdorff_1to2 = directed_hausdorff(points1, points2)[0]
            hausdorff_2to1 = directed_hausdorff(points2, points1)[0]
            hausdorff_dist = max(hausdorff_1to2, hausdorff_2to1)

            return {'hausdorff': hausdorff_dist}
        except Exception as e:
            print(f"  [警告] Hausdorff 计算失败: {e}")
            return {'hausdorff': np.nan}

    def compute_chamfer_distance(self, mesh1, mesh2, sample_points=10000):
        """计算Chamfer距离"""
        try:
            points1 = self._sample_points_from_mesh(mesh1, sample_points)
            points2 = self._sample_points_from_mesh(mesh2, sample_points)

            tree1 = cKDTree(points1)
            tree2 = cKDTree(points2)

            dist_1to2, _ = tree2.query(points1)
            dist_2to1, _ = tree1.query(points2)

            chamfer_dist = (np.mean(dist_1to2) + np.mean(dist_2to1)) / 2

            return {'chamfer': chamfer_dist}
        except Exception as e:
            print(f"  [警告] Chamfer 计算失败: {e}")
            return {'chamfer': np.nan}

    def compute_surface_deviation(self, ground_truth, reconstructed, sample_points=10000):
        """计算表面偏差统计"""
        try:
            gt_points = self._sample_points_from_mesh(ground_truth, sample_points)
            recon_points = np.array(reconstructed.points)

            if recon_points.shape[0] == 0:
                raise ValueError("重构网格中没有点")

            recon_tree = cKDTree(recon_points)
            deviations, _ = recon_tree.query(gt_points)

            return {
                'mean': np.mean(deviations),
                'std': np.std(deviations),
                'median': np.median(deviations),
                'rmse': np.sqrt(np.mean(deviations ** 2)),
                'mae': np.mean(np.abs(deviations)),
                'percentile_90': np.percentile(deviations, 90),
                'percentile_95': np.percentile(deviations, 95),
                'percentile_99': np.percentile(deviations, 99),
                'max': np.max(deviations),
                'min': np.min(deviations),
                'distribution': deviations
            }
        except Exception as e:
            print(f"  [警告] 表面偏差计算失败: {e}")
            return {
                'mean': np.nan, 'std': np.nan, 'median': np.nan, 'rmse': np.nan,
                'mae': np.nan, 'percentile_90': np.nan, 'percentile_95': np.nan,
                'percentile_99': np.nan, 'max': np.nan, 'min': np.nan,
                'distribution': np.array([np.nan])
            }

    def compute_volume_difference(self, mesh1, mesh2):
        """计算体积差异（如果网格是封闭的）"""
        try:
            vol1 = mesh1.volume
            vol2 = mesh2.volume

            vol_diff = abs(vol1 - vol2)
            vol_ratio = vol_diff / vol1 * 100  # 百分比

            return {
                'volume_gt': vol1,
                'volume_recon': vol2,
                'volume_diff': vol_diff,
                'volume_ratio': vol_ratio
            }
        except:
            return None

    def compute_normal_consistency(self, mesh1, mesh2, sample_points=5000):
        """计算法向量一致性"""
        try:
            mesh1_with_normals = mesh1.compute_normals(
                point_normals=True,
                cell_normals=False,
                auto_orient_normals=True
            )
            mesh2_with_normals = mesh2.compute_normals(
                point_normals=True,
                cell_normals=False,
                auto_orient_normals=True
            )

            points1 = self._sample_points_from_mesh(mesh1_with_normals, sample_points)

            mesh1_points = np.array(mesh1_with_normals.points)
            mesh1_normals = mesh1_with_normals.point_data['Normals']

            tree1 = cKDTree(mesh1_points)
            _, indices1 = tree1.query(points1)
            normals1 = mesh1_normals[indices1]

            mesh2_points = np.array(mesh2_with_normals.points)
            mesh2_normals = mesh2_with_normals.point_data['Normals']

            tree2 = cKDTree(mesh2_points)
            _, indices2 = tree2.query(points1)
            normals2 = mesh2_normals[indices2]

            consistencies = np.abs(np.einsum('ij,ij->i', normals1, normals2))

            return {
                'mean_consistency': np.mean(consistencies),
                'std_consistency': np.std(consistencies),
                'median_consistency': np.median(consistencies)
            }
        except Exception as e:
            print(f"  [警告] 法向量计算失败: {e}")
            return {
                'mean_consistency': np.nan,
                'std_consistency': np.nan,
                'median_consistency': np.nan
            }

    def compute_mesh_quality_metrics(self, mesh):
        """计算网格质量指标"""
        try:
            if mesh.n_faces == 0:
                return {'n_vertices': mesh.n_points, 'n_faces': 0, 'is_watertight': False}

            faces = mesh.faces.reshape(-1, 4)[:, 1:]
            trimesh_obj = trimesh.Trimesh(
                vertices=np.array(mesh.points),
                faces=faces
            )

            face_angles = trimesh_obj.face_angles
            min_angles = np.min(face_angles, axis=1)
            edges = trimesh_obj.edges_unique_length

            return {
                'n_vertices': len(trimesh_obj.vertices),
                'n_faces': len(trimesh_obj.faces),
                'is_watertight': trimesh_obj.is_watertight,
                'min_angle_mean': np.mean(np.rad2deg(min_angles)),
                'edge_length_mean': np.mean(edges),
            }
        except Exception as e:
            print(f"  [警告] 网格质量计算失败: {e}")
            return {'n_vertices': mesh.n_points, 'n_faces': 0, 'is_watertight': False}

    # ========== 新增：断层特定指标 ==========

    def compute_fault_curvature(self, mesh, neighborhood_size=10):
        """
        计算断层曲率 - 评价局部形态弯曲程度

        参数:
            mesh: PyVista网格对象
            neighborhood_size: 邻域大小，用于曲率计算

        返回:
            dict: 包含平均曲率、高斯曲率等统计信息
        """
        try:
            # 确保mesh有足够的点
            if mesh.n_points < 3:
                print(f"  [警告] 网格点数过少，无法计算曲率")
                return {
                    'mean_curvature': np.nan,
                    'std_curvature': np.nan,
                    'max_curvature': np.nan,
                    'curvature_percentile_90': np.nan,
                    'curvature_percentile_95': np.nan
                }

            # 计算法向量
            mesh_with_curvature = mesh.compute_normals(
                point_normals=True,
                cell_normals=False,
                auto_orient_normals=True
            )

            # 使用PyVista的曲率计算
            try:
                curvature_mesh = mesh_with_curvature.curvature(curv_type='mean')

                # 检查是否成功计算曲率
                if 'Mean_Curvature' not in curvature_mesh.point_data:
                    raise ValueError("曲率计算未返回预期数据")

                mean_curvatures = curvature_mesh['Mean_Curvature']

            except Exception as e:
                # 如果PyVista的曲率计算失败，使用备用方法
                print(f"  [信息] PyVista曲率计算失败，使用备用方法...")
                return self._compute_curvature_alternative(mesh_with_curvature)

            # 过滤无效值
            valid_curvatures = mean_curvatures[~np.isnan(mean_curvatures)]
            valid_curvatures = valid_curvatures[np.isfinite(valid_curvatures)]

            if len(valid_curvatures) == 0:
                print(f"  [警告] 未能计算有效的曲率值，使用备用方法")
                return self._compute_curvature_alternative(mesh_with_curvature)

            return {
                'mean_curvature': np.mean(np.abs(valid_curvatures)),
                'std_curvature': np.std(valid_curvatures),
                'max_curvature': np.max(np.abs(valid_curvatures)),
                'curvature_percentile_90': np.percentile(np.abs(valid_curvatures), 90),
                'curvature_percentile_95': np.percentile(np.abs(valid_curvatures), 95),
                'curvature_distribution': valid_curvatures
            }
        except Exception as e:
            print(f"  [警告] 曲率计算失败: {e}")
            return {
                'mean_curvature': np.nan,
                'std_curvature': np.nan,
                'max_curvature': np.nan,
                'curvature_percentile_90': np.nan,
                'curvature_percentile_95': np.nan
            }

    def _compute_curvature_alternative(self, mesh):
        """
        备用曲率计算方法：基于局部法向量变化

        参数:
            mesh: 已计算法向量的PyVista网格对象

        返回:
            dict: 曲率统计信息
        """
        try:
            points = np.array(mesh.points)
            normals = mesh.point_data['Normals']

            if len(points) < 10:
                return {
                    'mean_curvature': np.nan,
                    'std_curvature': np.nan,
                    'max_curvature': np.nan,
                    'curvature_percentile_90': np.nan,
                    'curvature_percentile_95': np.nan
                }

            # 使用KDTree找到每个点的邻域
            tree = cKDTree(points)
            k_neighbors = min(10, len(points) - 1)  # 使用10个邻居或更少

            curvatures = []

            # 对采样点计算曲率（避免计算所有点）
            sample_size = min(1000, len(points))
            sample_indices = np.random.choice(len(points), sample_size, replace=False)

            for idx in sample_indices:
                # 找到k近邻
                distances, indices = tree.query(points[idx], k=k_neighbors + 1)

                # 排除自身（第一个点）
                neighbor_indices = indices[1:]
                neighbor_distances = distances[1:]

                if len(neighbor_indices) < 3:
                    continue

                # 计算法向量变化
                normal_current = normals[idx]
                normals_neighbors = normals[neighbor_indices]

                # 法向量差异的均值作为曲率的近似
                normal_diffs = np.linalg.norm(normals_neighbors - normal_current, axis=1)

                # 考虑距离权重
                weights = 1.0 / (neighbor_distances + 1e-6)
                weighted_diff = np.average(normal_diffs, weights=weights)

                curvatures.append(weighted_diff)

            curvatures = np.array(curvatures)
            curvatures = curvatures[np.isfinite(curvatures)]

            if len(curvatures) == 0:
                return {
                    'mean_curvature': np.nan,
                    'std_curvature': np.nan,
                    'max_curvature': np.nan,
                    'curvature_percentile_90': np.nan,
                    'curvature_percentile_95': np.nan
                }

            return {
                'mean_curvature': np.mean(curvatures),
                'std_curvature': np.std(curvatures),
                'max_curvature': np.max(curvatures),
                'curvature_percentile_90': np.percentile(curvatures, 90),
                'curvature_percentile_95': np.percentile(curvatures, 95),
                'curvature_distribution': curvatures
            }

        except Exception as e:
            print(f"  [警告] 备用曲率计算也失败: {e}")
            return {
                'mean_curvature': np.nan,
                'std_curvature': np.nan,
                'max_curvature': np.nan,
                'curvature_percentile_90': np.nan,
                'curvature_percentile_95': np.nan
            }

    def compute_fault_segments(self, mesh, connectivity_threshold=2.0):
        """
        计算断层分段数 - 评价拓扑结构

        通过分析网格的连通性来识别断层是否正确分段
        （例如在X型或λ型交叉处是否正确断开）

        参数:
            mesh: PyVista网格对象
            connectivity_threshold: 连通性阈值（边长倍数）

        返回:
            dict: 包含分段数、最大段尺寸等信息
        """
        try:
            faces = mesh.faces.reshape(-1, 4)[:, 1:]
            trimesh_obj = trimesh.Trimesh(
                vertices=np.array(mesh.points),
                faces=faces
            )

            components = trimesh_obj.split(only_watertight=False)
            n_segments = len(components)

            segment_sizes = [len(comp.vertices) for comp in components]
            segment_areas = [comp.area for comp in components]

            edges = trimesh_obj.edges_unique
            boundary_edges = trimesh_obj.edges[trimesh_obj.edges_unique_inverse]

            return {
                'n_segments': n_segments,
                'max_segment_size': max(segment_sizes) if segment_sizes else 0,
                'min_segment_size': min(segment_sizes) if segment_sizes else 0,
                'mean_segment_size': np.mean(segment_sizes) if segment_sizes else 0,
                'total_boundary_length': len(boundary_edges),
                'segment_size_std': np.std(segment_sizes) if len(segment_sizes) > 1 else 0,
                'largest_segment_ratio': max(segment_sizes) / sum(segment_sizes) if sum(segment_sizes) > 0 else 0
            }
        except Exception as e:
            print(f"  [警告] 分段数计算失败: {e}")
            return {
                'n_segments': np.nan,
                'max_segment_size': np.nan,
                'min_segment_size': np.nan,
                'mean_segment_size': np.nan,
                'total_boundary_length': np.nan,
                'segment_size_std': np.nan,
                'largest_segment_ratio': np.nan
            }

    def compute_fault_sinuosity(self, mesh, n_samples=1000):
        """
        计算断层弯曲度 (Sinuosity) - 评价全局路径

        弯曲度 = 实际路径长度 / 直线距离
        值越接近1表示越直，值越大表示越弯曲

        参数:
            mesh: PyVista网格对象
            n_samples: 采样点数，用于计算路径

        返回:
            dict: 包含整体弯曲度、局部弯曲度统计等
        """
        try:
            points = np.array(mesh.points)

            if len(points) < 3:
                print(f"  [警告] 点数过少，无法计算弯曲度")
                return {
                    'overall_sinuosity': np.nan,
                    'mean_local_sinuosity': np.nan,
                    'max_local_sinuosity': np.nan
                }

            # 基于主方向的整体弯曲度
            pca = PCA(n_components=1)
            pca.fit(points)

            projected = pca.transform(points)

            sorted_indices = np.argsort(projected[:, 0])
            sorted_points = points[sorted_indices]

            if len(sorted_points) > n_samples:
                step = len(sorted_points) // n_samples
                sampled_points = sorted_points[::step]
            else:
                sampled_points = sorted_points

            path_segments = np.diff(sampled_points, axis=0)
            actual_length = np.sum(np.linalg.norm(path_segments, axis=1))

            straight_distance = np.linalg.norm(sampled_points[-1] - sampled_points[0])

            if straight_distance < 1e-6:
                overall_sinuosity = 1.0
            else:
                overall_sinuosity = actual_length / straight_distance

            # 局部弯曲度分析
            window_size = min(50, len(sampled_points) // 10)
            local_sinuosities = []

            if window_size >= 3:
                for i in range(len(sampled_points) - window_size):
                    window_points = sampled_points[i:i + window_size]
                    window_segments = np.diff(window_points, axis=0)
                    window_length = np.sum(np.linalg.norm(window_segments, axis=1))
                    window_straight = np.linalg.norm(window_points[-1] - window_points[0])

                    if window_straight > 1e-6:
                        local_sinuosity = window_length / window_straight
                        local_sinuosities.append(local_sinuosity)

            return {
                'overall_sinuosity': overall_sinuosity,
                'mean_local_sinuosity': np.mean(local_sinuosities) if local_sinuosities else np.nan,
                'std_local_sinuosity': np.std(local_sinuosities) if local_sinuosities else np.nan,
                'max_local_sinuosity': np.max(local_sinuosities) if local_sinuosities else np.nan,
                'median_local_sinuosity': np.median(local_sinuosities) if local_sinuosities else np.nan,
                'sinuosity_percentile_90': np.percentile(local_sinuosities, 90) if local_sinuosities else np.nan
            }
        except Exception as e:
            print(f"  [警告] 弯曲度计算失败: {e}")
            return {
                'overall_sinuosity': np.nan,
                'mean_local_sinuosity': np.nan,
                'max_local_sinuosity': np.nan
            }

    def evaluate_layer(self, layer_name, sample_points=10000):
        """评估单个地层/断层"""
        print(f"\n{'=' * 60}")
        print(f"评估: {layer_name}")
        print(f"{'=' * 60}")

        if layer_name not in self.ground_truth or \
                layer_name not in self.gempy_models or \
                layer_name not in self.loop_models:
            print(f"  [跳过] {layer_name} 的模型文件不完整")
            nan_dev = {
                'mean': np.nan, 'std': np.nan, 'median': np.nan, 'rmse': np.nan,
                'mae': np.nan, 'percentile_90': np.nan, 'percentile_95': np.nan,
                'percentile_99': np.nan, 'max': np.nan, 'min': np.nan,
                'distribution': np.array([np.nan])
            }
            nan_qual = {'n_vertices': 0, 'n_faces': 0, 'is_watertight': False}
            nan_fault = {
                'curvature': {'mean_curvature': np.nan},
                'segments': {'n_segments': np.nan},
                'sinuosity': {'overall_sinuosity': np.nan}
            }
            self.results[layer_name] = {
                'hausdorff': {'gempy': {'hausdorff': np.nan}, 'loop': {'hausdorff': np.nan}},
                'chamfer': {'gempy': {'chamfer': np.nan}, 'loop': {'chamfer': np.nan}},
                'deviation': {'gempy': nan_dev, 'loop': nan_dev},
                'volume': {'gempy': None, 'loop': None},
                'normal': {'gempy': {'mean_consistency': np.nan}, 'loop': {'mean_consistency': np.nan}},
                'quality': {'gt': nan_qual, 'gempy': nan_qual, 'loop': nan_qual},
                'fault_metrics': {'gempy': nan_fault, 'loop': nan_fault} if 'fault' in layer_name.lower() else None
            }
            return self.results[layer_name]

        gt = self.ground_truth[layer_name]
        gempy = self.gempy_models[layer_name]
        loop = self.loop_models[layer_name]

        print("  计算Hausdorff距离...")
        hausdorff_gempy = self.compute_hausdorff_distance(gt, gempy, sample_points)
        hausdorff_loop = self.compute_hausdorff_distance(gt, loop, sample_points)

        print("  计算Chamfer距离...")
        chamfer_gempy = self.compute_chamfer_distance(gt, gempy, sample_points)
        chamfer_loop = self.compute_chamfer_distance(gt, loop, sample_points)

        print("  计算表面偏差...")
        deviation_gempy = self.compute_surface_deviation(gt, gempy, sample_points)
        deviation_loop = self.compute_surface_deviation(gt, loop, sample_points)

        print("  计算体积差异...")
        volume_gempy = self.compute_volume_difference(gt, gempy)
        volume_loop = self.compute_volume_difference(gt, loop)

        print("  计算法向量一致性...")
        normal_gempy = self.compute_normal_consistency(gt, gempy)
        normal_loop = self.compute_normal_consistency(gt, loop)

        print("  计算网格质量...")
        quality_gt = self.compute_mesh_quality_metrics(gt)
        quality_gempy = self.compute_mesh_quality_metrics(gempy)
        quality_loop = self.compute_mesh_quality_metrics(loop)

        # 如果是断层，计算额外的断层特定指标
        fault_metrics_gempy = None
        fault_metrics_loop = None

        if 'fault' in layer_name.lower():
            print("  [断层] 计算曲率...")
            curvature_gt = self.compute_fault_curvature(gt)
            curvature_gempy = self.compute_fault_curvature(gempy)
            curvature_loop = self.compute_fault_curvature(loop)

            print("  [断层] 计算分段数...")
            segments_gt = self.compute_fault_segments(gt)
            segments_gempy = self.compute_fault_segments(gempy)
            segments_loop = self.compute_fault_segments(loop)

            print("  [断层] 计算弯曲度...")
            sinuosity_gt = self.compute_fault_sinuosity(gt)
            sinuosity_gempy = self.compute_fault_sinuosity(gempy)
            sinuosity_loop = self.compute_fault_sinuosity(loop)

            fault_metrics_gempy = {
                'curvature': curvature_gempy,
                'segments': segments_gempy,
                'sinuosity': sinuosity_gempy,
                'curvature_gt': curvature_gt,
                'segments_gt': segments_gt,
                'sinuosity_gt': sinuosity_gt
            }

            fault_metrics_loop = {
                'curvature': curvature_loop,
                'segments': segments_loop,
                'sinuosity': sinuosity_loop,
                'curvature_gt': curvature_gt,
                'segments_gt': segments_gt,
                'sinuosity_gt': sinuosity_gt
            }

            print(f"\n  [断层指标]")
            print(
                f"    真实值 | 曲率: {curvature_gt['mean_curvature']:.4f}, 分段数: {segments_gt['n_segments']}, 弯曲度: {sinuosity_gt['overall_sinuosity']:.4f}")
            print(
                f"    GemPy  | 曲率: {curvature_gempy['mean_curvature']:.4f}, 分段数: {segments_gempy['n_segments']}, 弯曲度: {sinuosity_gempy['overall_sinuosity']:.4f}")
            print(
                f"    Loop   | 曲率: {curvature_loop['mean_curvature']:.4f}, 分段数: {segments_loop['n_segments']}, 弯曲度: {sinuosity_loop['overall_sinuosity']:.4f}")

        self.results[layer_name] = {
            'hausdorff': {'gempy': hausdorff_gempy, 'loop': hausdorff_loop},
            'chamfer': {'gempy': chamfer_gempy, 'loop': chamfer_loop},
            'deviation': {'gempy': deviation_gempy, 'loop': deviation_loop},
            'volume': {'gempy': volume_gempy, 'loop': volume_loop},
            'normal': {'gempy': normal_gempy, 'loop': normal_loop},
            'quality': {'gt': quality_gt, 'gempy': quality_gempy, 'loop': quality_loop},
            'fault_metrics': {'gempy': fault_metrics_gempy, 'loop': fault_metrics_loop} if fault_metrics_gempy else None
        }

        print(f"\n  Hausdorff | GemPy: {hausdorff_gempy['hausdorff']:.4f}, Loop: {hausdorff_loop['hausdorff']:.4f}")
        print(f"  Chamfer   | GemPy: {chamfer_gempy['chamfer']:.4f}, Loop: {chamfer_loop['chamfer']:.4f}")
        print(f"  RMSE      | GemPy: {deviation_gempy['rmse']:.4f}, Loop: {deviation_loop['rmse']:.4f}")

        return self.results[layer_name]

    def evaluate_all(self, sample_points=10000):
        """评估所有地层和断层"""
        for layer in self.all_layers:
            self.evaluate_layer(layer, sample_points)

    def plot_comparison_charts(self, save_path=None):
        """绘制综合对比图表（包含断层特定指标）"""
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})

        # 根据是否有断层决定子图布局
        has_faults = len(self.faults) > 0
        if has_faults:
            fig = plt.figure(figsize=(24, 24))
            gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        else:
            fig, axes = plt.subplots(3, 3, figsize=(20, 18))
            gs = None

        x = np.arange(len(self.all_layers))
        width = 0.35

        gempy_color = COLORS['gempy']
        loop_color = COLORS['loop']

        # 1. Hausdorff距离
        ax = fig.add_subplot(gs[0, 0]) if has_faults else axes[0, 0]
        hausdorff_gempy = [self.results[l]['hausdorff']['gempy']['hausdorff'] for l in self.all_layers]
        hausdorff_loop = [self.results[l]['hausdorff']['loop']['hausdorff'] for l in self.all_layers]
        ax.bar(x - width / 2, hausdorff_gempy, width, color=gempy_color, label='GemPy')
        ax.bar(x + width / 2, hausdorff_loop, width, color=loop_color, label='LoopStructural')
        ax.set_title('(a) Hausdorff Distance', fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel('Distance (m)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(self.all_layers, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 2. Chamfer距离
        ax = fig.add_subplot(gs[0, 1]) if has_faults else axes[0, 1]
        chamfer_gempy = [self.results[l]['chamfer']['gempy']['chamfer'] for l in self.all_layers]
        chamfer_loop = [self.results[l]['chamfer']['loop']['chamfer'] for l in self.all_layers]
        ax.bar(x - width / 2, chamfer_gempy, width, color=gempy_color, label='GemPy')
        ax.bar(x + width / 2, chamfer_loop, width, color=loop_color, label='LoopStructural')
        ax.set_title('(b) Chamfer Distance', fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel('Average Distance (m)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(self.all_layers, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 3. RMSE
        ax = fig.add_subplot(gs[0, 2]) if has_faults else axes[0, 2]
        rmse_gempy = [self.results[l]['deviation']['gempy']['rmse'] for l in self.all_layers]
        rmse_loop = [self.results[l]['deviation']['loop']['rmse'] for l in self.all_layers]
        ax.bar(x - width / 2, rmse_gempy, width, color=gempy_color, label='GemPy')
        ax.bar(x + width / 2, rmse_loop, width, color=loop_color, label='LoopStructural')
        ax.set_title('(c) Surface Deviation (RMSE)', fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel('RMSE (m)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(self.all_layers, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 4. 偏差分布直方图
        ax = fig.add_subplot(gs[1, 0]) if has_faults else axes[1, 0]
        if self.layers:
            layer_to_plot = self.layers[0]
            gempy_dist = np.array(self.results[layer_to_plot]['deviation']['gempy']['distribution'])
            loop_dist = np.array(self.results[layer_to_plot]['deviation']['loop']['distribution'])

            gempy_dist = gempy_dist[~np.isnan(gempy_dist)]
            loop_dist = loop_dist[~np.isnan(loop_dist)]

            if len(gempy_dist) > 0 and len(loop_dist) > 0:
                max_val = max(np.percentile(gempy_dist, 99) if len(gempy_dist) > 0 else 1,
                              np.percentile(loop_dist, 99) if len(loop_dist) > 0 else 1)
                if max_val == 0 or np.isnan(max_val): max_val = 1
                bins = np.linspace(0, max_val, 50)

                ax.hist(gempy_dist, bins=bins, alpha=0.7, color=gempy_color, density=True, label='GemPy (Dist.)')
                ax.hist(loop_dist, bins=bins, alpha=0.7, color=loop_color, density=True, label='Loop (Dist.)')

                median_gempy = np.nanmedian(gempy_dist)
                median_loop = np.nanmedian(loop_dist)

                ax.axvline(median_gempy, color=gempy_color, linestyle='--', linewidth=2,
                           label=f'GemPy Median: {median_gempy:.2f}')
                ax.axvline(median_loop, color=loop_color, linestyle=':', linewidth=2,
                           label=f'Loop Median: {median_loop:.2f}')

            ax.set_title(f'(d) Deviation Distribution ({layer_to_plot})', fontsize=12, fontweight='bold', loc='left')
            ax.set_xlabel('Deviation (m)', fontsize=10)
            ax.set_ylabel('Probability Density', fontsize=10)
            ax.legend(fontsize=9, frameon=False, loc='upper right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticks([])
            ax.tick_params(axis='x', which='major', labelsize=9)
        else:
            ax.set_title('(d) Deviation Distribution', fontsize=12, fontweight='bold', loc='left')

        # 5. 法向量一致性
        ax = fig.add_subplot(gs[1, 1]) if has_faults else axes[1, 1]
        normal_gempy = [self.results[l]['normal']['gempy']['mean_consistency'] for l in self.all_layers]
        normal_loop = [self.results[l]['normal']['loop']['mean_consistency'] for l in self.all_layers]
        ax.bar(x - width / 2, normal_gempy, width, color=gempy_color, label='GemPy')
        ax.bar(x + width / 2, normal_loop, width, color=loop_color, label='LoopStructural')
        ax.set_title('(e) Normal Consistency', fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel('Mean Cosine Similarity', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(self.all_layers, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 6. 体积差异
        ax = fig.add_subplot(gs[1, 2]) if has_faults else axes[1, 2]
        vol_ratios_gempy = []
        vol_ratios_loop = []
        vol_layers = []
        for l in self.all_layers:
            vol_data = self.results[l].get('volume')
            if vol_data and vol_data.get('gempy') and vol_data.get('loop'):
                vol_ratios_gempy.append(vol_data['gempy']['volume_ratio'])
                vol_ratios_loop.append(vol_data['loop']['volume_ratio'])
                vol_layers.append(l)

        if vol_layers:
            x_vol = np.arange(len(vol_layers))
            ax.bar(x_vol - width / 2, vol_ratios_gempy, width, color=gempy_color, label='GemPy')
            ax.bar(x_vol + width / 2, vol_ratios_loop, width, color=loop_color, label='LoopStructural')
            ax.set_title('(f) Volume Difference (%)', fontsize=12, fontweight='bold', loc='left')
            ax.set_ylabel('Relative Error (%)', fontsize=10)
            ax.set_xticks(x_vol)
            ax.set_xticklabels(vol_layers, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()

        # ===== 断层特定指标（如果有断层）=====
        if has_faults:
            x_fault = np.arange(len(self.faults))

            # 7. 断层曲率对比
            ax = fig.add_subplot(gs[2, 0])
            curvature_gempy = [self.results[f]['fault_metrics']['gempy']['curvature']['mean_curvature']
                               for f in self.faults]
            curvature_loop = [self.results[f]['fault_metrics']['loop']['curvature']['mean_curvature']
                              for f in self.faults]
            curvature_gt = [self.results[f]['fault_metrics']['gempy']['curvature_gt']['mean_curvature']
                            for f in self.faults]

            ax.bar(x_fault - width, curvature_gt, width, color=COLORS['neutral'], label='Ground Truth', alpha=0.8)
            ax.bar(x_fault, curvature_gempy, width, color=gempy_color, label='GemPy')
            ax.bar(x_fault + width, curvature_loop, width, color=loop_color, label='LoopStructural')
            ax.set_title('(g) Fault Curvature', fontsize=12, fontweight='bold', loc='left')
            ax.set_ylabel('Mean Curvature', fontsize=10)
            ax.set_xticks(x_fault)
            ax.set_xticklabels(self.faults, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 8. 断层分段数对比
            ax = fig.add_subplot(gs[2, 1])
            segments_gempy = [self.results[f]['fault_metrics']['gempy']['segments']['n_segments']
                              for f in self.faults]
            segments_loop = [self.results[f]['fault_metrics']['loop']['segments']['n_segments']
                             for f in self.faults]
            segments_gt = [self.results[f]['fault_metrics']['gempy']['segments_gt']['n_segments']
                           for f in self.faults]

            ax.bar(x_fault - width, segments_gt, width, color=COLORS['neutral'], label='Ground Truth', alpha=0.8)
            ax.bar(x_fault, segments_gempy, width, color=gempy_color, label='GemPy')
            ax.bar(x_fault + width, segments_loop, width, color=loop_color, label='LoopStructural')
            ax.set_title('(h) Fault Segment Count', fontsize=12, fontweight='bold', loc='left')
            ax.set_ylabel('Number of Segments', fontsize=10)
            ax.set_xticks(x_fault)
            ax.set_xticklabels(self.faults, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 9. 断层弯曲度对比
            ax = fig.add_subplot(gs[2, 2])
            sinuosity_gempy = [self.results[f]['fault_metrics']['gempy']['sinuosity']['overall_sinuosity']
                               for f in self.faults]
            sinuosity_loop = [self.results[f]['fault_metrics']['loop']['sinuosity']['overall_sinuosity']
                              for f in self.faults]
            sinuosity_gt = [self.results[f]['fault_metrics']['gempy']['sinuosity_gt']['overall_sinuosity']
                            for f in self.faults]

            ax.bar(x_fault - width, sinuosity_gt, width, color=COLORS['neutral'], label='Ground Truth', alpha=0.8)
            ax.bar(x_fault, sinuosity_gempy, width, color=gempy_color, label='GemPy')
            ax.bar(x_fault + width, sinuosity_loop, width, color=loop_color, label='LoopStructural')
            ax.set_title('(i) Fault Sinuosity', fontsize=12, fontweight='bold', loc='left')
            ax.set_ylabel('Sinuosity (path/straight)', fontsize=10)
            ax.set_xticks(x_fault)
            ax.set_xticklabels(self.faults, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 雷达图和统计表的位置调整
        radar_row = 3 if has_faults else 2

        # GemPy雷达图
        ax = plt.subplot(gs[radar_row, 0], projection='polar') if has_faults else plt.subplot(3, 3, 7,
                                                                                              projection='polar')
        categories_labels = ['Hausdorff', 'Chamfer', 'RMSE', 'Normals']
        self._plot_quality_radar(ax, 'gempy', self.all_layers, categories_labels)
        label = '(j)' if has_faults else '(g)'
        ax.set_title(f'{label} GemPy Avg. Quality', fontsize=12, fontweight='bold', pad=20)

        # Loop雷达图
        ax = plt.subplot(gs[radar_row, 1], projection='polar') if has_faults else plt.subplot(3, 3, 8,
                                                                                              projection='polar')
        self._plot_quality_radar(ax, 'loop', self.all_layers, categories_labels)
        label = '(k)' if has_faults else '(h)'
        ax.set_title(f'{label} Loop Avg. Quality', fontsize=12, fontweight='bold', pad=20)

        # 统计摘要表
        ax = fig.add_subplot(gs[radar_row, 2]) if has_faults else axes[2, 2]
        ax.axis('off')

        avg_haus_g = np.nanmean([self.results[l]['hausdorff']['gempy']['hausdorff'] for l in self.all_layers])
        avg_haus_l = np.nanmean([self.results[l]['hausdorff']['loop']['hausdorff'] for l in self.all_layers])
        avg_cham_g = np.nanmean([self.results[l]['chamfer']['gempy']['chamfer'] for l in self.all_layers])
        avg_cham_l = np.nanmean([self.results[l]['chamfer']['loop']['chamfer'] for l in self.all_layers])
        avg_rmse_g = np.nanmean([self.results[l]['deviation']['gempy']['rmse'] for l in self.all_layers])
        avg_rmse_l = np.nanmean([self.results[l]['deviation']['loop']['rmse'] for l in self.all_layers])
        avg_mae_g = np.nanmean([self.results[l]['deviation']['gempy']['mae'] for l in self.all_layers])
        avg_mae_l = np.nanmean([self.results[l]['deviation']['loop']['mae'] for l in self.all_layers])

        def percent_diff(v_g, v_l):
            if np.isnan(v_g) or np.isnan(v_l): return 'N/A'
            if v_g == 0: return 'N/A'
            diff = (v_g - v_l) / v_g * 100
            return f'{diff:+.1f}%'

        table_data = [
            ['Metric (Avg.)', 'GemPy', 'Loop', 'Improvement'],
            ['Hausdorff', f'{avg_haus_g:.2f}', f'{avg_haus_l:.2f}', percent_diff(avg_haus_g, avg_haus_l)],
            ['Chamfer', f'{avg_cham_g:.2f}', f'{avg_cham_l:.2f}', percent_diff(avg_cham_g, avg_cham_l)],
            ['RMSE', f'{avg_rmse_g:.2f}', f'{avg_rmse_l:.2f}', percent_diff(avg_rmse_g, avg_rmse_l)],
            ['MAE', f'{avg_mae_g:.2f}', f'{avg_mae_l:.2f}', percent_diff(avg_mae_g, avg_mae_l)],
        ]

        g_wins = sum([1 for l in self.all_layers
                      if not np.isnan(self.results[l]['deviation']['gempy']['rmse'])
                      and not np.isnan(self.results[l]['deviation']['loop']['rmse'])
                      and self.results[l]['deviation']['gempy']['rmse'] < self.results[l]['deviation']['loop']['rmse']])
        l_wins = sum([1 for l in self.all_layers
                      if not np.isnan(self.results[l]['deviation']['gempy']['rmse'])
                      and not np.isnan(self.results[l]['deviation']['loop']['rmse'])
                      and self.results[l]['deviation']['loop']['rmse'] < self.results[l]['deviation']['gempy']['rmse']])

        table_data.append(['Best RMSE', f'{g_wins}', f'{l_wins}', 'layers'])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.2, 0.2, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)

        for i in range(4):
            table[(0, i)].set_facecolor(COLORS['neutral'])
            table[(0, i)].set_text_props(weight='bold', color='white')

        for row in range(1, 6):
            if row < 5:
                table[(row, 1)].set_facecolor(gempy_color)
                table[(row, 1)].set_text_props(color='white')
                table[(row, 2)].set_facecolor(loop_color)
                table[(row, 2)].set_text_props(color='white')

                diff_val_str = table_data[row][3]
                if diff_val_str != 'N/A':
                    try:
                        diff_val = float(diff_val_str.strip('%+'))
                        color = '#006400' if diff_val > 0 else '#C70039'
                        table[(row, 3)].set_text_props(color=color, weight='bold')
                    except:
                        pass
            else:
                table[(row, 1)].set_text_props(weight='bold')
                table[(row, 2)].set_text_props(weight='bold')

        label = '(l)' if has_faults else '(i)'
        ax.set_title(f'{label} Summary Statistics', fontsize=12, fontweight='bold', loc='left', pad=20)

        # 使用constrained_layout代替tight_layout以避免与polar axes冲突
        # 或者手动调整子图间距
        try:
            if has_faults:
                # 对于有断层的情况，使用GridSpec已经设置了间距
                pass
            else:
                # 对于没有断层的情况，也跳过tight_layout
                pass
        except:
            pass

        fig.suptitle(f'Quantitative Model Comparison: {self.model_type.upper()}',
                     fontsize=16, fontweight='bold', y=0.995 if has_faults else 1.02)

        if save_path:
            # 保存时使用bbox_inches='tight'会自动调整布局
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n对比图表已保存至: {save_path}")

        plt.show()

    def _plot_quality_radar(self, ax, method, layers, categories_labels):
        """绘制质量雷达图"""
        categories = len(categories_labels)

        all_hausdorff = [self.results[l]['hausdorff'][m]['hausdorff']
                         for l in layers for m in ['gempy', 'loop']]
        all_chamfer = [self.results[l]['chamfer'][m]['chamfer']
                       for l in layers for m in ['gempy', 'loop']]
        all_rmse = [self.results[l]['deviation'][m]['rmse']
                    for l in layers for m in ['gempy', 'loop']]
        all_normal = [self.results[l]['normal'][m]['mean_consistency']
                      for l in layers for m in ['gempy', 'loop']]

        max_hausdorff = np.nanmax([v for v in all_hausdorff if not np.isnan(v)]) if any(
            not np.isnan(v) for v in all_hausdorff) else 1
        max_chamfer = np.nanmax([v for v in all_chamfer if not np.isnan(v)]) if any(
            not np.isnan(v) for v in all_chamfer) else 1
        max_rmse = np.nanmax([v for v in all_rmse if not np.isnan(v)]) if any(
            not np.isnan(v) for v in all_rmse) else 1

        hausdorff_vals = [self.results[l]['hausdorff'][method]['hausdorff'] for l in layers]
        chamfer_vals = [self.results[l]['chamfer'][method]['chamfer'] for l in layers]
        rmse_vals = [self.results[l]['deviation'][method]['rmse'] for l in layers]
        normal_vals = [self.results[l]['normal'][method]['mean_consistency'] for l in layers]

        avg_values = [
            1 - (np.nanmean(hausdorff_vals) / max_hausdorff) if max_hausdorff > 0 else 0,
            1 - (np.nanmean(chamfer_vals) / max_chamfer) if max_chamfer > 0 else 0,
            1 - (np.nanmean(rmse_vals) / max_rmse) if max_rmse > 0 else 0,
            np.nanmean(normal_vals)
        ]

        avg_values = np.nan_to_num(avg_values, nan=0.0)
        avg_values = np.clip(avg_values, 0, 1).tolist()

        angles = np.linspace(0, 2 * np.pi, categories, endpoint=False).tolist()
        avg_values_closed = avg_values + [avg_values[0]]
        angles_closed = angles + [angles[0]]

        color = COLORS['gempy'] if method == 'gempy' else COLORS['loop']

        ax.plot(angles_closed, avg_values_closed, 'o-', linewidth=2, color=color)
        ax.fill(angles_closed, avg_values_closed, alpha=0.25, color=color)

        ax.set_xticks(angles)
        ax.set_xticklabels(categories_labels, fontsize=9)

        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8, color='gray')
        ax.set_ylim(0, 1)

    def generate_detailed_report(self, output_path='detailed_comparison_report.csv'):
        """生成详细CSV报告（包含断层指标）"""
        report_data = []

        for layer in self.all_layers:
            layer_type = 'Layer' if layer in self.layers else 'Fault'

            for method in ['gempy', 'loop']:
                method_name = 'GemPy' if method == 'gempy' else 'LoopStructural'
                row = {
                    'Layer/Fault': layer,
                    'Type': layer_type,
                    'Method': method_name,
                    'Hausdorff': self.results[layer]['hausdorff'][method].get('hausdorff', np.nan),
                    'Chamfer': self.results[layer]['chamfer'][method].get('chamfer', np.nan),
                    'RMSE': self.results[layer]['deviation'][method].get('rmse', np.nan),
                    'MAE': self.results[layer]['deviation'][method].get('mae', np.nan),
                    'Mean_Deviation': self.results[layer]['deviation'][method].get('mean', np.nan),
                    'Normal_Consistency': self.results[layer]['normal'][method].get('mean_consistency', np.nan),
                }
                vol_data = self.results[layer]['volume']
                if vol_data and vol_data[method]:
                    row['Volume_Diff'] = vol_data[method].get('volume_diff', np.nan)
                    row['Volume_Ratio'] = vol_data[method].get('volume_ratio', np.nan)

                # 添加断层特定指标
                if layer_type == 'Fault' and self.results[layer].get('fault_metrics'):
                    fault_data = self.results[layer]['fault_metrics'][method]
                    if fault_data:
                        row['Mean_Curvature'] = fault_data['curvature'].get('mean_curvature', np.nan)
                        row['Max_Curvature'] = fault_data['curvature'].get('max_curvature', np.nan)
                        row['N_Segments'] = fault_data['segments'].get('n_segments', np.nan)
                        row['Largest_Segment_Ratio'] = fault_data['segments'].get('largest_segment_ratio', np.nan)
                        row['Overall_Sinuosity'] = fault_data['sinuosity'].get('overall_sinuosity', np.nan)
                        row['Mean_Local_Sinuosity'] = fault_data['sinuosity'].get('mean_local_sinuosity', np.nan)

                        # 与真实值的差异
                        row['Curvature_Diff'] = abs(fault_data['curvature'].get('mean_curvature', np.nan) -
                                                    fault_data['curvature_gt'].get('mean_curvature', np.nan))
                        row['Segments_Diff'] = abs(fault_data['segments'].get('n_segments', 0) -
                                                   fault_data['segments_gt'].get('n_segments', 0))
                        row['Sinuosity_Diff'] = abs(fault_data['sinuosity'].get('overall_sinuosity', np.nan) -
                                                    fault_data['sinuosity_gt'].get('overall_sinuosity', np.nan))

                report_data.append(row)

        df = pd.DataFrame(report_data)
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\n详细报告已保存至: {output_path}")
        return df

    def run_full_comparison(self, output_dir='./comparison_results', sample_points=10000):
        """运行完整对比分析"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"开始完整对比分析 - {self.model_type.upper()}")
        print(f"{'=' * 80}")

        print("\n【评估所有地层和断层】...")
        self.evaluate_all(sample_points)

        print("\n【生成对比图表】...")
        save_path_pdf = f'{output_dir}/{self.model_type}_comprehensive_comparison.pdf'
        self.plot_comparison_charts(save_path=save_path_pdf)

        print("\n【生成详细报告】...")
        self.generate_detailed_report(
            output_path=f'{output_dir}/{self.model_type}_detailed_report.csv'
        )

        print(f"\n{'=' * 80}")
        print(f"对比分析完成！结果保存至: {output_dir}")
        print(f"{'=' * 80}")

        return self.results


# ============== 使用示例 ==============
if __name__ == "__main__":

    print("=" * 80)
    print("示例：地堑模型对比分析（增强断层评估）")
    print("=" * 80)

    graben_gt_paths = {
        'ground1': './vtk/vtk_diqian/origin/ground1.stl',
        'ground2': './vtk/vtk_diqian/origin/ground2.stl',
        'ground3': './vtk/vtk_diqian/origin/ground3.stl',
        'fault1': './vtk/vtk_diqian/origin/fault1.stl',
        'fault2': './vtk/vtk_diqian/origin/fault2.stl',
    }

    graben_gempy_paths = {
        'ground1': './vtk/vtk_diqian/Gempy/ground1.vtp',
        'ground2': './vtk/vtk_diqian/Gempy/ground2.vtp',
        'ground3': './vtk/vtk_diqian/Gempy/ground3.vtp',
        'fault1': './vtk/vtk_diqian/Gempy/fault1.vtp',
        'fault2': './vtk/vtk_diqian/Gempy/fault2.vtp',
    }

    graben_loop_paths = {
        'ground1': './vtk/vtk_diqian/loop/ground1.vtk',
        'ground2': './vtk/vtk_diqian/loop/ground2.vtk',
        'ground3': './vtk/vtk_diqian/loop/ground3.vtk',
        'fault1': './vtk/vtk_diqian/loop/fault1.vtk',
        'fault2': './vtk/vtk_diqian/loop/fault2.vtk',
    }

    try:
        graben_comparison = LayerWiseModelComparison(
            ground_truth_paths=graben_gt_paths,
            gempy_paths=graben_gempy_paths,
            loop_paths=graben_loop_paths,
            model_type='graben'
        )

        results = graben_comparison.run_full_comparison(
            output_dir='./graben_comparison',
            sample_points=50000
        )

        print("\n" + "=" * 80)
        print("对比分析完成！主要输出文件：")
        print(f"  - graben_comprehensive_comparison.pdf  # 综合对比图表（包含断层指标）")
        print(f"  - graben_detailed_report.csv           # 详细指标CSV（包含断层特定数据）")
        print("\n新增断层评估指标：")
        print(f"  ✓ 曲率 (Curvature)：评价局部形态弯曲是否真实")
        print(f"  ✓ 分段数 (Segment Count)：评价拓扑结构（X型/λ型断开情况）")
        print(f"  ✓ 弯曲度 (Sinuosity)：评价全局路径是否过于理想化")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n[错误] 分析中止: {e}")
    except Exception as e:
        print(f"\n[严重错误] 发生意外: {e}")
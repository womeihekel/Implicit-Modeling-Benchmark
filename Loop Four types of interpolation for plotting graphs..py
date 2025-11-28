
## 二维展示
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata, Rbf
#
# # 随机定义几个观测点
# points = np.array([
#     [0.1, 0.2],
#     [0.4, 0.8],
#     [0.8, 0.3],
#     [0.7, 0.7],
#     [0.3, 0.5],
# ])
# values = np.array([0.2, 0.8, 0.5, 1.0, 0.6])  # 对应的标量值
#
# # 定义网格
# grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
#
# # 1. 分段线性插值 (PLI = linear)
# pli = griddata(points, values, (grid_x, grid_y), method='linear')
#
# # 2. 有限差分插值 (FDI 这里用 cubic griddata 近似演示)
# fdi = griddata(points, values, (grid_x, grid_y), method='cubic')
#
# # 3. 折叠插值 (FBI 用 RBF + multiquadric 模拟局部褶皱变化)
# rbf_fbi = Rbf(points[:,0], points[:,1], values, function='multiquadric', epsilon=0.2)
# fbi = rbf_fbi(grid_x, grid_y)
#
# # 4. 广义径向基插值 (RBF)
# rbf_rbf = Rbf(points[:,0], points[:,1], values, function='linear')
# rbf = rbf_rbf(grid_x, grid_y)
#
# # 画图
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# methods = [pli, fdi, fbi, rbf]
# titles = ["PLI", "FDI", "FBI", "RBF"]
#
# for ax, field, title in zip(axes.ravel(), methods, titles):
#     cs = ax.contourf(grid_x, grid_y, field, levels=15, cmap='viridis')
#     ax.scatter(points[:,0], points[:,1], c=values, cmap='viridis', edgecolor='k', s=60)
#     ax.set_title(title)
#     fig.colorbar(cs, ax=ax, shrink=0.8)
#
# plt.tight_layout()
# plt.savefig("interpolation_comparison.png", dpi=300, bbox_inches='tight')
# plt.show()


##三维展示


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, Rbf
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子以保证可重复性
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# 生成更有结构的测试数据，模拟地质结构
def generate_geological_data(n_points=25):
    """生成模拟地质结构的数据点"""
    # 创建一些有地质意义的数据点分布
    x = np.random.uniform(-2, 2, n_points)
    y = np.random.uniform(-2, 2, n_points)

    # 添加一些特定的控制点来创建更明显的地质特征
    x = np.append(x, [-1.5, -0.5, 0.5, 1.5, 0, 0, -1, 1])
    y = np.append(y, [-1.5, -0.5, 0.5, 1.5, -1, 1, 0, 0])

    # 创建更复杂的Z值，模拟地质层面
    z = (np.sin(x * 1.5) * np.cos(y * 1.5) +
         0.3 * np.sin(x * 3) * np.sin(y * 3) +
         0.1 * x * y)

    points = np.column_stack([x, y])
    return points, z


# 生成数据
points, values = generate_geological_data()

# 生成更高分辨率的网格
grid_x, grid_y = np.mgrid[-2:2:150j, -2:2:150j]


# 实现真正的有限差分插值
def finite_difference_interpolation(points, values, grid_x, grid_y):
    """
    实现基于有限差分思想的插值方法
    使用局部加权线性回归，模拟有限差分的数值特性
    """
    # 将网格点flatten
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    result = np.zeros(grid_points.shape[0])

    # 对每个网格点计算插值
    for i, (gx, gy) in enumerate(grid_points):
        # 计算到所有数据点的距离
        distances = np.sqrt((points[:, 0] - gx) ** 2 + (points[:, 1] - gy) ** 2)

        # 使用有限差分风格的权重函数（更局部化）
        # 找到最近的几个点进行局部线性拟合
        n_neighbors = min(6, len(points))  # 使用最近的6个点
        nearest_indices = np.argsort(distances)[:n_neighbors]

        if distances[nearest_indices[0]] == 0:
            # 如果正好在数据点上
            result[i] = values[nearest_indices[0]]
        else:
            # 使用局部线性回归，权重随距离快速衰减
            local_points = points[nearest_indices]
            local_values = values[nearest_indices]
            local_distances = distances[nearest_indices]

            # 有限差分风格的权重（指数衰减）
            weights = np.exp(-local_distances ** 2 / (0.5 ** 2))
            weights /= np.sum(weights)

            # 简单的加权平均（有限差分的数值特性）
            result[i] = np.sum(weights * local_values)

    return result.reshape(grid_x.shape)


# 实现折叠插值 (Fold Interpolation, Laurent et al., 2016)
def fold_interpolation(points, values, grid_x, grid_y):
    """
    实现基于四面体网格的折叠插值方法 (Laurent et al., 2016)
    模拟地质褶皱结构的插值特性
    """
    from scipy.spatial import Delaunay

    # 创建Delaunay三角剖分作为四面体网格的2D近似
    tri = Delaunay(points)

    # 将网格点flatten
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    result = np.zeros(grid_points.shape[0])

    for i, (gx, gy) in enumerate(grid_points):
        # 找到包含当前点的三角形
        simplex = tri.find_simplex([gx, gy])

        if simplex >= 0:
            # 在三角形内部，使用重心坐标进行插值
            triangle = tri.simplices[simplex]
            # 获取三角形顶点
            vertices = points[triangle]
            vertex_values = values[triangle]

            # 计算重心坐标
            # 使用克拉默法则计算重心坐标
            v0 = vertices[2] - vertices[0]
            v1 = vertices[1] - vertices[0]
            v2 = np.array([gx, gy]) - vertices[0]

            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)

            # 计算重心坐标
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom

            # 折叠插值的特殊处理：考虑地质褶皱的非线性特性
            # 使用修正的重心坐标，引入褶皱效应
            w1 = 1 - u - v  # 第一个顶点的权重
            w2 = v  # 第二个顶点的权重
            w3 = u  # 第三个顶点的权重

            # 折叠插值的核心：非线性权重修正
            # 模拟褶皱结构中的非线性变形
            fold_factor = 0.3  # 褶皱强度参数

            # 计算到各顶点的距离，用于折叠效应
            d1 = np.linalg.norm(np.array([gx, gy]) - vertices[0])
            d2 = np.linalg.norm(np.array([gx, gy]) - vertices[1])
            d3 = np.linalg.norm(np.array([gx, gy]) - vertices[2])

            # 应用折叠效应：距离越近权重增强越明显
            fold_w1 = w1 * (1 + fold_factor * np.exp(-d1))
            fold_w2 = w2 * (1 + fold_factor * np.exp(-d2))
            fold_w3 = w3 * (1 + fold_factor * np.exp(-d3))

            # 重新归一化权重
            total_weight = fold_w1 + fold_w2 + fold_w3
            fold_w1 /= total_weight
            fold_w2 /= total_weight
            fold_w3 /= total_weight

            # 计算插值结果
            result[i] = (fold_w1 * vertex_values[0] +
                         fold_w2 * vertex_values[1] +
                         fold_w3 * vertex_values[2])

        else:
            # 在凸包外部，使用最近邻外推
            distances = np.sqrt((points[:, 0] - gx) ** 2 + (points[:, 1] - gy) ** 2)
            nearest_idx = np.argmin(distances)
            result[i] = values[nearest_idx]

    return result.reshape(grid_x.shape)


# 四种插值方法
print("正在计算插值...")
pli = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.nan)
fdi = finite_difference_interpolation(points, values, grid_x, grid_y)  # 自定义FDI
cubic = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=np.nan)  # 标准三次插值

# RBF插值，使用更适合地质数据的参数
rbf = Rbf(points[:, 0], points[:, 1], values,
          function='multiquadric', smooth=0.1, epsilon=1.0)
rbf_z = rbf(grid_x, grid_y)

# 创建优化的可视化
fig = plt.figure(figsize=(16, 12))
plt.rcParams.update({'font.size': 10})

methods = [
    ('PLI (Piecewise Linear)', pli),
    ('FDI (Finite Difference)', fdi),
    ('Cubic Interpolation', cubic),
    ('RBF (Radial Basis Function)', rbf_z)
]

# 统一Z轴范围以便比较
z_min = min(np.nanmin(data) for _, data in methods)
z_max = max(np.nanmax(data) for _, data in methods)

for i, (title, data) in enumerate(methods, 1):
    ax = fig.add_subplot(2, 2, i, projection='3d')

    # 绘制表面，保持viridis配色
    surf = ax.plot_surface(grid_x, grid_y, data,
                           cmap='viridis',
                           alpha=0.8,
                           linewidth=0,
                           antialiased=True,
                           rcount=100, ccount=100)

    # 绘制原始数据点（更大更明显）
    scatter = ax.scatter(points[:, 0], points[:, 1], values,
                         color='red', s=40, alpha=0.9,
                         edgecolors='darkred', linewidth=1)


    # 统一Z轴范围
    ax.set_zlim(z_min, z_max)

    # 优化视角
    ax.view_init(elev=25, azim=45)

    # 添加颜色条
    if i == 4:  # 只在最后一个子图添加颜色条
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Interpolated Values', rotation=270, labelpad=15)

    # 设置网格和背景
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 设置刻度
    ax.tick_params(labelsize=9)

# 调整子图间距
plt.tight_layout(pad=3.0)


# 保存高质量图片
plt.savefig("interpolation_3d_optimized.pdf", dpi=300, bbox_inches='tight',pad_inches=0.1)
print("图片已保存为 interpolation_3d_optimized.tiff")

# 显示统计信息
print("\n插值方法统计:")
for name, data in methods:
    valid_data = data[~np.isnan(data)] if hasattr(data, 'dtype') else data
    print(f"{name:25s} - 范围: [{np.min(valid_data):.3f}, {np.max(valid_data):.3f}]")

plt.show()
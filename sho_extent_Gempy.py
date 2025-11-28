import pandas as pd
import numpy as np

def calculate_extent_from_surf(csv_path, buffer_ratio=0.1, round_to=10):
    df = pd.read_csv(csv_path)

    if not set(['X', 'Y', 'Z']).issubset(df.columns):
        raise ValueError("❌ CSV 必须包含列：X, Y, Z")

    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()
    z_min, z_max = df['Z'].min(), df['Z'].max()

    # 原始范围
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # 加 buffer（例如扩大 10%）
    x_buffer = x_range * buffer_ratio
    y_buffer = y_range * buffer_ratio
    z_buffer = z_range * buffer_ratio

    x_min -= x_buffer
    x_max += x_buffer
    y_min -= y_buffer
    y_max += y_buffer
    z_min -= z_buffer
    z_max += z_buffer

    # 可选：四舍五入到指定单位，比如 10m
    def round_extent(val, mode='floor'):
        return np.floor(val / round_to) * round_to if mode == 'floor' else np.ceil(val / round_to) * round_to

    extent = (
        [round_extent(x_min, 'floor'), round_extent(x_max, 'ceil')],
        [round_extent(y_min, 'floor'), round_extent(y_max, 'ceil')],
        [round_extent(z_min, 'floor'), round_extent(z_max, 'ceil')],
    )

    print(f"✅ 自动计算 GemPy extent 为：\n{extent}")
    return extent


extent = calculate_extent_from_surf("D:/study_for_python/霍林河14煤构造/output/gempy_batch_combined/simple_test/paper/jioacha_λ/surface_points_r.csv")

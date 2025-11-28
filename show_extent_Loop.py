import pandas as pd

df = pd.read_excel('./surface_points_q.xlsx')  # 或直接从 DataFrame 读

x_min, x_max = df['X'].min(), df['X'].max()
y_min, y_max = df['Y'].min(), df['Y'].max()
z_min, z_max = df['Z'].min(), df['Z'].max()

print("X:", x_min, x_max)
print("Y:", y_min, y_max)
print("Z:", z_min, z_max)

from LoopStructural import GeologicalModel
from LoopStructural.visualisation import Loop3DView


import numpy as np
import pandas as pd

# X Y Z的最大最小范围
origin = [-62,-361,-50]
extent = [575, 93, 500]
data = pd.read_excel("./surface_points_q.xlsx")
model = GeologicalModel(origin, extent)


data["val"].unique()
data["feature_name"].unique()

model.data=data

# 地层之间的高度
vals = [0,50,100,150]


strat_column = {"ground": {}}
for i in range(len(vals) - 1):
    strat_column["ground"]["unit_{}".format(i)] = {
        "min": vals[i],
        "max": vals[i + 1],
        "id": i,
    }

model.set_stratigraphic_column(strat_column)

strat_column['faults'] = {}

# -------------------------------重点做断层
model.create_and_add_fault(
    "fault1", ##断层名
    100,   ##插值时参与拟合曲面的数据点数量上限
    nelements=1e6, # 指定用于插值的离散网格大小
    interpolator_type="PLI",##分段线性插值 适合断层数据比较稀疏
    # ------------------或者"FDI" 有限充分，通常精度更高但计算更慢
    buffer=1,   #控制的是模型空间边界与数据边界之间的扩展比例
    major_axis=650,  #最大轴（长轴）方向上的影响范围，通常平行于断层走向
    minor_axis=550,   #中等轴，控制插值核在断层走向垂直方向的延伸程度
    intermediate_axis=100,#最小轴（厚度方向），通常是断层面垂直方向（厚度越小越尖锐）
)

model.create_and_add_fault(
    "fault2",
    50,
    nelements=1e6,
    interpolator_type="PLI",
    buffer=1,
    major_axis=650,
    minor_axis=50,
    intermediate_axis=50,
)

ground = model.create_and_add_foliation(
    "ground",
    interpolatortype="FDI",  # try changing this to 'PLI'
    nelements=1e5,  # 指定用于插值网格的离散点数
    buffer=10, #控制空间外延比例的
    solver = 'cg', #求解线性系统的方法 pyamg
    damp=True,#假如阻尼项以在插值时控制震荡，防止模型在局部高频区域过拟合
)


######################################################################
# Plot the surfaces
# ------------------------------------
# print(model.fault_names())
viewer = Loop3DView(model)## 创建三维交互式可视化器绑定model
viewer.plot_data(ground)    #显示数据点
# viewer.plot_scalar_field(ground)  #显示标量场
viewer.plot_model_surfaces(cmap="tab20")  #颜色设置tab20  Set3  Accent
model.save('./model.vtk')  #保存模型
viewer.show()  #模型展示


# viewer.interactive() #jupyter的交互模式
######################################################################
# Plot block diagram
# -------------------
viewer = Loop3DView(model)
viewer.plot_block_model(cmap="tab20")
viewer.plot_scalar_field(ground,cmap="tab20")
viewer.show()
#####################################################################
#显示梯度向量场（foliation的法向方向）
# viewer.plot_vector_field(ground) #对建好的 ground 层理场绘制其向量场，通常表示法线方向或主构造方向。
# viewer.nsteps = np.array([0,50,100])   #控制交互式显示中的标量场/层理面展示的分布断面位置
# model.save('model.vtk')

# for s in model['strati'].surfaces([330]):  #从一个连续的层理插值场中，导出某一个实际等值面的三角网
#   s.save('surface330.vtk')
#
# viewer.display()
# viewer.interactive()
# 保存 block model
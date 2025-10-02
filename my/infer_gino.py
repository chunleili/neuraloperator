"""
GINO Car CFD Model Inference
============================
该脚本用于加载一个预训练的 GINO 模型，对一个汽车样本进行压力预测，
并将预测结果与真实值进行对比可视化。
"""


# 步骤 1: 导入依赖
# --------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# 切换路径到项目根目录
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
os.chdir(proj_dir)


def load_GT():
    # 加载Ground Truth
    # --------------------------
    # 加载一个小汽车数据集中的一个样本作为我们模型的输入
    print("正在加载数据样本...")
    from neuralop.data.datasets import load_mini_car
    data_list = load_mini_car()
    sample = data_list[0]  # 使用数据集中的第一辆车作为例子
    print(f"加载完成")
    return sample

def infer(sample):
    # 步骤 2: 导入模型
    # -----------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from neuralop.models import GINO
    model = GINO.from_checkpoint(save_folder='./checkpoints/car-pressure/',save_name="model")
    
    # 从训练脚本中使用的相同配置文件导入 MyConfig
    # 实例化一个与训练时结构完全相同的模型骨架
    # print("正在创建 GINO 模型架构...")
    # from config.gino_carcfd_config import MyConfig
    # from neuralop import get_model
    # config = MyConfig()
    # model = get_model(config.to_dict())
    # print("模型架构: GINO")

    # # 将保存在 checkpoint 的权重加载到模型骨架中
    # print("正在加载模型权重...")
    # checkpoint_path = "./checkpoints/car-pressure/model_state_dict.pt"
    # # 使用 weights_only=False 来加载您自己训练的可信模型文件
    # state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # 切换到评估模式，这对于推理很重要
    print("模型权重加载成功。")


    # 步骤 3: 准备模型输入
    # ---------------------------------
    # 参照训练脚本中的 GINOCFDDataProcessor，手动构建模型期望的输入字典
    print("正在准备模型输入...")

    # 为所有张量增加一个批处理维度 (batch dimension = 1)，并移动到正确的设备
    model_input = {
        'input_geom': sample['vertices'].unsqueeze(0).to(device),
        'latent_queries': sample['query_points'].unsqueeze(0).to(device),
        'output_queries': sample['vertices'].unsqueeze(0).to(device), # 在顶点上进行预测
        'latent_features': sample['distance'].unsqueeze(0).to(device),
        'x': None  # 在此配置中，GINO 不使用规则网格数据
    }
    print("模型输入准备完毕。")

    # 步骤 4: 执行推理
    # ---------------------------------
    print("正在执行模型推理...")
    with torch.no_grad():  # 在无梯度的上下文中运行，以节省计算和内存
        # 使用 ** 将字典解包为独立的关键字参数
        prediction = model(**model_input)

    # 从批处理维度中挤压掉，并移动到 CPU 以便绘图
    pred_pressure = prediction.squeeze().cpu().numpy()
    print("推理完成。")

    # save pred_pressure
    np.save("pred_pressure.npy", pred_pressure)
    print(f"存储预测压力到 pred_pressure.npy")



def plot(pred_pressure, sample):
    # 步骤 5: 可视化预测结果 vs. 真实值 vs. 误差
    # -------------------------------------------------
    print("正在可视化结果...")
    fig = plt.figure(figsize=(21, 7)) # 增大了图像尺寸以容纳三个子图

    # 获取用于比较的真实数据
    vertices = sample['vertices'].numpy()
    true_pressure = sample['press'].squeeze().numpy()

    # 计算百分比误差
    # 添加一个极小值 epsilon 来避免除以零
    epsilon = 1e-8
    percentage_error = (pred_pressure - true_pressure) / (np.abs(true_pressure) + epsilon) * 100
    print(f"最大百分比误差: {percentage_error.max():.2f}%")
    print(f"最小百分比误差: {percentage_error.min():.2f}%")


    # 确定真实值和预测值的共享颜色范围
    vmin = min(true_pressure.min(), pred_pressure.min())
    vmax = max(true_pressure.max(), pred_pressure.max())

    # 图 1: 真实压力
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    scatter1 = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2] * 2,
                        s=2, c=true_pressure, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title("Ground Truth Pressure")
    ax1.view_init(elev=20, azim=150, roll=0, vertical_axis='y')

    # 图 2: 模型预测压力
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    scatter2 = ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2] * 2,
                        s=2, c=pred_pressure, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title("GINO Model Prediction")
    ax2.view_init(elev=20, azim=150, roll=0, vertical_axis='y')

    if with_ax3:
        # 图 3: 百分比误差
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        # 使用 'coolwarm' 颜色映射，蓝色表示负误差（预测偏低），红色表示正误差（预测偏高）
        scatter3 = ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2] * 2,
                            s=2, c=percentage_error, cmap='coolwarm')
        ax3.set_title("Percentage Error (%)")
        ax3.view_init(elev=20, azim=150, roll=0, vertical_axis='y')

    axs = [ax1, ax2]
    if with_ax3:
        axs.append(ax3)

    # 调整坐标轴并为每个图添加颜色条
    for ax in axs:
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # 为真实值和预测值图添加颜色条
    cbar1 = fig.colorbar(scatter2, ax=[ax1, ax2], shrink=0.6, pad=0.08)
    cbar1.set_label("Normalized Pressure")

    if with_ax3:
        # 为误差图添加颜色条
        cbar2 = fig.colorbar(scatter3, ax=ax3, shrink=0.6, pad=0.08)
        cbar2.set_label("Error (%)")


    plt.suptitle("GINO Car Pressure: Prediction and Error Analysis", fontsize=16)
    # 更新保存的文件名
    plt.savefig("car_pressure_prediction_with_error.png", dpi=300, bbox_inches='tight')
    print("预测与误差分析图已保存至 car_pressure_prediction_with_error.png")
    plt.show()

def calc_L2_error(pred_pressure, GT):
    # 新增：计算并打印 L2 测试误差
    # -----------------------------------
    from neuralop.losses.data_losses import LpLoss
    
    # 实例化相对 L2 损失函数
    l2_error_fn = LpLoss(d=1, p=2)
    
    # 将 numpy 预测结果和 ground truth 转换为张量以进行计算
    pred_tensor = torch.from_numpy(pred_pressure)
    true_tensor = GT['press'].squeeze()

    # LpLoss 期望输入包含批处理维度，因此使用 unsqueeze(0)
    test_error = l2_error_fn(pred_tensor.unsqueeze(0), true_tensor.unsqueeze(0))
    print(f"L2 相对测试误差: {test_error.item():.4f}")
    # -----------------------------------


if __name__ == "__main__":
    hasdone = False
    with_ax3 = True # 默认显示误差图

    sample = load_GT()
    if hasdone:
        print("已经执行过推理, 直接从npy加载预测结果")
    else:
        infer(sample) # 结果保存在npy
    pred_pressure = np.load("pred_pressure.npy")
    test_error = calc_L2_error(pred_pressure, sample)

    plot(pred_pressure, sample)
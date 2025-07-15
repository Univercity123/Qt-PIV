import numpy as np
import matplotlib.pyplot as plt

# 网格大小和参数
n = 30  # 网格大小
x = np.linspace(-5, 5, n)  # x方向坐标
y = np.linspace(-5, 5, n)  # y方向坐标
X, Y = np.meshgrid(x, y)  # 生成网格

# 设置湍流的随机扰动
def generate_turbulence_field(X, Y, t, noise_amplitude=0.5):
    """
    使用正弦波与随机噪声相结合，模拟湍流的u和v分量。
    t: 时间
    noise_amplitude: 随机噪声幅度
    """
    # 基于正弦波的基本流动（模仿湍流的宏观运动）
    u_sine = np.sin(X + t) + np.cos(Y + t)  # 水平速度分量
    v_sine = np.cos(X + t) - np.sin(Y + t)  # 垂直速度分量

    # 添加随机噪声，模拟湍流的小尺度变化
    u_noise = noise_amplitude * np.random.randn(*X.shape)  # 随机噪声
    v_noise = noise_amplitude * np.random.randn(*Y.shape)  # 随机噪声

    # 总的速度场
    u = u_sine + u_noise
    v = v_sine + v_noise

    return u, v

# 创建时间序列
time_steps = 100  # 时间步数
time = np.linspace(0, 10, time_steps)

# 准备绘图
fig, ax = plt.subplots(figsize=(8, 6))

# 创建动画效果
for t in time:
    # 生成当前时间的湍流流场
    u, v = generate_turbulence_field(X, Y, t)
    
    # 清除上一帧，绘制当前帧
    ax.clear()
    
    # 绘制速度场
    ax.quiver(X, Y, u, v, scale=80, color='black')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_title(f"Turbulent Flow at time = {t:.2f}s")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 更新显示
    plt.pause(0.05)  # 控制动画速度

plt.show()

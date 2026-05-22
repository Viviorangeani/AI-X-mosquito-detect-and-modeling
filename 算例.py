import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# ================== 1. 模拟数据 ==================
np.random.seed(42)  # 保证可复现

# 网格参数
n_x, n_y = 25, 25  # 25x25 = 625 个网格点
n_grid = n_x * n_y
n_time = 36  # 3天，每天12个时段，共36个时段

# 1.1 生成不均匀的频次基底（空间分布）
# 创建两个高风险聚集中心
x_coords = np.arange(n_x)
y_coords = np.arange(n_y)
xx, yy = np.meshgrid(x_coords, y_coords)

# 第一个中心 (5, 5)，第二个中心 (18, 20)
center1 = (5, 5)
center2 = (18, 20)
dist1 = np.sqrt((xx - center1[0])**2 + (yy - center1[1])**2)
dist2 = np.sqrt((xx - center2[0])**2 + (yy - center2[1])**2)

# 基底频次：两个高斯峰 + 背景噪声
base_freq = 2 * np.exp(-dist1**2 / 30) + 3 * np.exp(-dist2**2 / 40) + 0.5 * np.random.rand(n_x, n_y)

# 1.2 时间节律（蚊虫活动在白天和傍晚较高，深夜较低）
# 为每个时间段生成时间系数 (36个)
time_factor = np.zeros(n_time)
for t in range(n_time):
    hour_of_day = (t % 12) * 2  # 每个时间段中心小时: 0,2,4,...,22
    # 模拟活动节律：清晨(6-8)和傍晚(18-20)高，深夜低
    if hour_of_day in [6, 8, 18, 20]:
        time_factor[t] = 1.5
    elif hour_of_day in [4, 10, 16, 22]:
        time_factor[t] = 1.0
    elif hour_of_day in [0, 2]:
        time_factor[t] = 0.2
    else:
        time_factor[t] = 0.8
    # 加入随机波动
    time_factor[t] *= np.random.uniform(0.8, 1.2)

# 1.3 生成每个网格每个时段的频次 (625, 36)
freq_3d = np.zeros((n_grid, n_time))
for i in range(n_grid):
    gx = i // n_y
    gy = i % n_y
    base = base_freq[gx, gy]
    for t in range(n_time):
        # 频次 = 空间基底 * 时间节律 + 泊松噪声
        lam = max(0, base * time_factor[t])
        freq_3d[i, t] = np.random.poisson(lam)

# 查看数据概览
print("频次矩阵形状:", freq_3d.shape)
print("总检测次数:", freq_3d.sum().astype(int))
print("每个网格的平均频次范围: [{:.2f}, {:.2f}]".format(freq_3d.mean(axis=1).min(), freq_3d.mean(axis=1).max()))

# ================== 2. 模拟环境因子（无锡6月）==================
# 逐时段温度 (°C) 和湿度 (%)
temperature = np.zeros(n_time)
humidity = np.zeros(n_time)
for t in range(n_time):
    hour_of_day = (t % 12) * 2
    day = t // 12  # 0,1,2
    # 温度：日变化 + 随机波动，无锡6月白天30-32，夜间22-24
    temp_base = 27 + 5 * np.sin((hour_of_day - 14) * np.pi / 12)  # 下午2点最高
    temp_base -= day * 0.5  # 3天略有下降
    temperature[t] = temp_base + np.random.normal(0, 1)
    # 湿度：相对湿度60-90%，与温度负相关
    hum_base = 75 - 0.8 * (temperature[t] - 25) + np.random.normal(0, 3)
    humidity[t] = np.clip(hum_base, 60, 95)

# 环境适宜度 (简化为两个S函数的平均)
def temp_suitability(T):
    if T < 21:
        return 0.0
    elif T > 35:
        return 0.0
    else:
        return (T - 21) / (30 - 21) if T < 30 else 1.0

def humid_suitability(RH):
    if RH < 60:
        return 0.0
    elif RH > 85:
        return 1.0
    else:
        return (RH - 60) / (85 - 60)

env_norm = np.array([(temp_suitability(temperature[t]) + humid_suitability(humidity[t]))/2 for t in range(n_time)])

print("\n环境因子示例 (前5个时段):")
for t in range(5):
    print(f"时段{t+1}: T={temperature[t]:.1f}°C, RH={humidity[t]:.1f}%, Env={env_norm[t]:.2f}")

# ================== 3. 频次标准化 ==================
# 对每个网格的36个时段做min-max标准化（按行），或者全局标准化？这里做全局标准化（所有网格所有时段一起），保持相对关系
scaler = MinMaxScaler(feature_range=(0, 1))
freq_flat = freq_3d.flatten().reshape(-1, 1)
freq_norm_flat = scaler.fit_transform(freq_flat)
freq_norm = freq_norm_flat.reshape(n_grid, n_time)

print("\n标准化后频次范围: [{:.3f}, {:.3f}]".format(freq_norm.min(), freq_norm.max()))

# ================== 4. 基于平均频次的聚类 ==================
# 特征：每个网格的平均标准化频次
avg_freq_per_grid = freq_norm.mean(axis=1).reshape(-1, 1)  # (625,1)

# 确定最优K值（轮廓系数）
from sklearn.metrics import silhouette_score
sil_scores = []
K_range = range(2, 7)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(avg_freq_per_grid)
    score = silhouette_score(avg_freq_per_grid, labels)
    sil_scores.append(score)
    print(f"K={k}, 轮廓系数={score:.3f}")

best_k = K_range[np.argmax(sil_scores)]
print(f"\n最优聚类数 K={best_k}")

# 用最优K进行最终聚类
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(avg_freq_per_grid)

# 将聚类结果映射回网格形状 (25,25)
cluster_map = cluster_labels.reshape(n_x, n_y)
avg_freq_map = avg_freq_per_grid.reshape(n_x, n_y)

# ================== 5. 可视化 ==================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 5.1 平均频次热力图
im1 = axes[0,0].imshow(avg_freq_map, cmap='YlOrRd', origin='lower', interpolation='nearest')
axes[0,0].set_title(f'Average Standardized Frequency per Grid (25x25)')
axes[0,0].set_xlabel('Grid X')
axes[0,0].set_ylabel('Grid Y')
plt.colorbar(im1, ax=axes[0,0], label='Avg Freq (norm)')

# 5.2 聚类结果图
# 使用离散 colormap
cmap_cluster = plt.cm.get_cmap('viridis', best_k)
im2 = axes[0,1].imshow(cluster_map, cmap=cmap_cluster, origin='lower', interpolation='nearest')
axes[0,1].set_title(f'K-means Clusters (K={best_k}) based on Avg Freq')
axes[0,1].set_xlabel('Grid X')
axes[0,1].set_ylabel('Grid Y')
cbar2 = plt.colorbar(im2, ax=axes[0,1], ticks=range(best_k))
cbar2.set_label('Cluster ID')

# 5.3 总频次分布（所有网格所有时段）
total_freq_per_grid = freq_3d.sum(axis=1)
axes[1,0].hist(total_freq_per_grid, bins=30, color='skyblue', edgecolor='black')
axes[1,0].set_title('Total Detection Counts per Grid (3 days)')
axes[1,0].set_xlabel('Total Count')
axes[1,0].set_ylabel('Number of Grids')

# 5.4 每个聚类的平均频次柱状图
cluster_means = []
for c in range(best_k):
    mask = cluster_labels == c
    cluster_means.append(avg_freq_per_grid[mask].mean())
axes[1,1].bar(range(best_k), cluster_means, color=plt.cm.viridis(np.linspace(0, 1, best_k)))
axes[1,1].set_title(f'Average Freq by Cluster (K={best_k})')
axes[1,1].set_xlabel('Cluster')
axes[1,1].set_ylabel('Mean Standardized Frequency')

plt.tight_layout()
plt.savefig('risk_clustering_heatmap.png', dpi=150)
plt.show()

# 输出高风险聚类ID（平均频次最高的聚类）
highest_cluster = np.argmax(cluster_means)
print(f"\n高风险聚类: Cluster {highest_cluster} (平均频次 {cluster_means[highest_cluster]:.3f})")
print("该聚类包含的网格数:", (cluster_labels == highest_cluster).sum())
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import KDTree
import networkx as nx

def simulate_largest_component_fraction(L, radii, sds, n_simulations):
    """
    模拟硬几何图中最大连通分量节点占比随radius和sd变化的情况。

    参数：
        L (float): 环面（torus）的边长。
        radii (list of float): 不同的半径值列表。
        sds (list of float): 不同的标准差值列表。
        n_simulations (int): 每个(radius, sd)组合的模拟次数。

    返回：
        dict: 一个字典，其中键是sd，值是最大连通分量节点占比的平均值列表。
    """
    def generate_points(L, sd):
        if sd == 666:
            # 生成泊松过程
            mean_points = L**2
            n_points = np.random.poisson(mean_points)
            x = np.random.uniform(0, L, n_points)
            y = np.random.uniform(0, L, n_points)
            return np.vstack((x, y)).T
        else:
            # 生成带扰动的方格点
            n_points = int(L)
            x, y = np.meshgrid(np.arange(n_points), np.arange(n_points))
            x, y = x.ravel(), y.ravel()
            points = np.vstack((x, y)).T
            perturbations = np.random.normal(0, sd, points.shape)
            return np.mod(points + perturbations, L)
    
    def compute_largest_component_fraction(points, radius, L):
        # 扩展点集以考虑环面计算
        offsets = np.array([[0, 0], [0, L], [0, -L],
                            [L, 0], [L, L], [L, -L],
                            [-L, 0], [-L, L], [-L, -L]])
        extended_points = np.vstack([points + offset for offset in offsets])
        
        # 使用KDTree找到点对
        tree = KDTree(extended_points)
        pairs = tree.query_pairs(radius)

        # 过滤原始点中的点对
        n_original = len(points)
        graph = nx.Graph()
        for i, j in pairs:
            if i < n_original and j < n_original:
                graph.add_edge(i, j)
        
        # 计算最大连通分量的节点占比
        if len(graph.nodes) == 0:
            return 0
        largest_cc = max(nx.connected_components(graph), key=len)
        return len(largest_cc) / len(points)

    # 存储结果
    results = {sd: [] for sd in sds}

    # 模拟
    for sd in tqdm(sds, desc="模拟不同SD值"):
        for radius in tqdm(radii, desc=f"处理SD={sd}的不同半径", leave=False):
            fractions = []
            for _ in range(n_simulations):
                points = generate_points(L, sd)
                fraction = compute_largest_component_fraction(points, radius, L)
                fractions.append(fraction)
            results[sd].append(np.mean(fractions))

    return results


# 参数设置
L = 50
radii = np.arange(0.8, 1.4, 0.01)
sds = [0,0.2, 0.4, 0.6, 0.8, 1, 666]
n_simulations = 100

# 运行模拟
results = simulate_largest_component_fraction(L, radii, sds, n_simulations)

# 绘图
plt.figure(figsize=(10, 6))
for sd, fractions in results.items():
    plt.plot(radii, fractions, label=f"SD={sd}")
plt.xlabel("Radius")
plt.ylabel("最大连通分量中的节点占比")
plt.title("不同SD下最大连通分量占比随Radius变化")
plt.legend()
plt.grid()
plt.show()

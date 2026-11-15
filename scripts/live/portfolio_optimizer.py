# scripts/live/portfolio_optimizer.py
import numpy as np
from scipy.optimize import linprog

def optimize_portfolio(probs, vols, max_single=0.2, max_total=0.8, max_vol=0.02):
    """
    线性规划优化仓位
    probs: 上涨概率列表（作为预期收益代理）
    vols: 波动率列表（用于风险约束）
    """
    n = len(probs)
    if n == 0:
        return []

    # 目标函数系数：负的概率（因为 linprog 默认求最小值）
    c = -np.array(probs)

    # 不等式约束矩阵 A_ub * x <= b_ub
    A_ub = []
    b_ub = []

    # 1. 单票上限约束：x_i <= max_single
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(max_single)

    # 2. 总仓位上限：sum(x_i) <= max_total
    A_ub.append(np.ones(n))
    b_ub.append(max_total)

    # 3. 组合波动上限（线性近似）：sum(vol_i * x_i) <= max_vol
    A_ub.append(vols)
    b_ub.append(max_vol)

    # 边界约束：0 <= x_i <= max_single
    bounds = [(0, max_single) for _ in range(n)]

    # 求解
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        weights = result.x / result.x.sum() * max_total  # 归一化并乘总仓位
        return weights.tolist()
    else:
        # 求解失败则等权分配（安全回退）
        return [max_total / n] * n
    
if __name__ == '__main__':
    # 模拟几个上涨概率和波动率
    probs = [0.8, 0.7, 0.6, 0.5, 0.4]
    vols = [0.02, 0.03, 0.025, 0.02, 0.04]
    weights = optimize_portfolio(probs, vols)
    print("测试结果 - 最优仓位权重：")
    for i, w in enumerate(weights):
        print(f"  股票{i+1}: {w:.4f}")
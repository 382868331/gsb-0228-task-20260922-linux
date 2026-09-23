"""独立参考实现：全二叉树枚举（不导入、不调用被测核心的任何算法）。

参考只吃原始的 axes/size 数据，独立实现子集枚举与峰值口径，用于给
tensor_peak 核心提供期望值。测试中被测核心仅通过其公开接口被调用。
"""

import math


def _subset_tables(axes_list, label_order, sizes):
    n = len(axes_list)
    N = 1 << n
    axes_of = [None] * N
    size_of = [0] * N
    for mask in range(1, N):
        count = {}
        members = mask
        while members:
            bit = members & -members
            i = bit.bit_length() - 1
            for a in axes_list[i]:
                count[a] = count.get(a, 0) + 1
            members ^= bit
        ax = tuple(a for a in label_order if count.get(a, 0) == 1)
        axes_of[mask] = ax
        size_of[mask] = math.prod(sizes[a] for a in ax)
    return axes_of, size_of


def _pareto(points):
    """points: 可迭代的 (muls, peak)；返回乘法升序、峰值严格下降的前沿。"""
    ordered = sorted(set(points))
    kept = []
    best_peak = None
    for m, p in ordered:
        if best_peak is None or p < best_peak:
            kept.append((m, p))
            best_peak = p
    return kept


def all_optimal_points(axes_list, sizes):
    """枚举所有合法二叉收缩树（含左右求值顺序），返回全集 Pareto 点。

    返回 (points, min_peak)：points 为 (muls, peak) 元组列表（乘法升序）。
    网络不连通（某子集无合法划分）时 points 为空。
    """
    n = len(axes_list)
    label_order = []
    for ax in axes_list:
        for a in ax:
            if a not in label_order:
                label_order.append(a)
    axes_of, size_of = _subset_tables(axes_list, label_order, sizes)
    N = 1 << n
    fronts = {1 << i: [(0, 0)] for i in range(n)}
    for mask in range(1, N):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        found = set()
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub & low:
                left = set(axes_of[sub])
                right = set(axes_of[other])
                shared = left & right
                if shared and sub in fronts and other in fronts:
                    s_out = size_of[mask]
                    node_muls = math.prod(sizes[a] for a in (left | right))
                    s_l, s_r = size_of[sub], size_of[other]
                    triple = s_l + s_r + s_out
                    for ml, pl in fronts[sub]:
                        for mr, pr in fronts[other]:
                            peak_l_first = max(pl, s_l + pr, triple)
                            peak_r_first = max(pr, s_r + pl, triple)
                            found.add(
                                (ml + mr + node_muls,
                                 min(peak_l_first, peak_r_first))
                            )
            sub = (sub - 1) & mask
        front = _pareto(found)
        if front:
            fronts[mask] = front
    return fronts.get(N - 1, [])


def best_muls_under_limit(axes_list, sizes, limit):
    """返回满足 peak <= limit 的最少乘法数；无可行计划返回 None。"""
    points = all_optimal_points(axes_list, sizes)
    feasible = [m for m, p in points if p <= limit]
    return min(feasible) if feasible else None

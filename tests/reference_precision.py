"""独立参考：枚举全部括号顺序 × 每步精度组合，独立计算成本/峰值/误差。

不导入被测核心；误差公式由本文件独立推导书写（标准点积舍入界 γ_K）。
"""

import math


def gamma(k, u):
    ku = k * u
    if ku >= 1:
        return None
    return ku / (1 - ku)


def reference_options(specs, label_size, levels):
    """返回所有可行（括号树，精度指派）的 dict 列表。"""
    n = len(specs)
    N = 1 << n
    labels = []
    for t in specs:
        for a in t.axes:
            if a not in labels:
                labels.append(a)

    axes_of = {}
    for mask in range(1, N):
        count = {}
        members = mask
        while members:
            bit = members & -members
            for axis in specs[bit.bit_length() - 1].axes:
                count[axis] = count.get(axis, 0) + 1
            members ^= bit
        axes_of[mask] = tuple(a for a in labels if count.get(a, 0) == 1)

    size_of = {m: math.prod(label_size[a] for a in axes_of[m])
               for m in range(1, N)}

    states = {}
    for i, t in enumerate(specs):
        states[1 << i] = [dict(cost=0, peak=0, bound=t.abs_bound,
                               error=t.abs_error, trace=("leaf", t.name))]

    for mask in range(1, N):
        if mask & (mask - 1) == 0:
            continue
        results = []
        sub = (mask - 1) & mask
        low = mask & -mask
        while sub:
            other = mask ^ sub
            if (sub & low) and sub < mask:
                al, ar = axes_of[sub], axes_of[other]
                shared = set(al) & set(ar)
                if shared:
                    kept = tuple(a for a in labels if a not in shared
                                 and a in set(al) | set(ar))
                    s_out = math.prod(label_size[a] for a in kept)
                    muls = math.prod(label_size[a]
                                     for a in set(al) | set(ar))
                    klen = muls // s_out if s_out else muls
                    sl, sr = size_of[sub], size_of[other]
                    triple = sl + sr + s_out
                    for e1 in states.get(sub, ()):
                        for e2 in states.get(other, ()):
                            p1 = max(e1["peak"], sl + e2["peak"], triple)
                            p2 = max(e2["peak"], sr + e1["peak"], triple)
                            peak = min(p1, p2)
                            for level in levels:
                                g = gamma(klen, level.u)
                                if g is None:
                                    continue
                                terms = klen * e1["bound"] * e2["bound"]
                                inerr = klen * (
                                    e1["error"] * e2["bound"]
                                    + e2["error"] * e1["bound"]
                                    + e1["error"] * e2["error"]
                                )
                                rnd = g * terms
                                results.append(dict(
                                    cost=(e1["cost"] + e2["cost"]
                                          + muls * level.mul_cost),
                                    peak=peak,
                                    bound=terms + rnd,
                                    error=inerr + rnd,
                                    trace=("node", level.name,
                                           e1["trace"], e2["trace"]),
                                ))
            sub = (sub - 1) & mask
        states[mask] = results

    return states[N - 1]


def best_under(options, mem_limit, err_limit):
    feasible = [o for o in options
                if o["peak"] <= mem_limit and o["error"] <= err_limit]
    if not feasible:
        return None
    return min(feasible, key=lambda o: (o["cost"], o["peak"], o["error"]))

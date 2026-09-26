"""奇偶标记增量演示：带符号收缩计划与执行（现场计算，无预录答案）。

运行：python demo_iteration.py（仅标准库，离线，约 1 秒内完成）。

内容：
1. 带奇偶标记的四环网络：打印每步轴排列、奇指标交换次数与符号，
   用 execute_plan 执行并与独立的全指标带符号直算逐元素核对。
2. 括号顺序不变性：同一网络在两个内存上限下选出不同括号树，
   带符号结果完全一致（转置不是免费重排）。
3. 迹收缩 A(i,j) B(j,i)：奇-奇、奇-偶、全偶三种标记的符号对比。
4. 无标记行为：同形状不标记时符号恒 +1，数值与无符号直算一致。
5. 实际触发的异常边界：同一指标的奇偶标记冲突。
"""

import itertools

from tensor_peak import (
    TensorSpec,
    find_plan,
    execute_plan,
    InvalidProblemError,
)


def make_values(spec):
    """确定性生成稠密整数张量（仓库内代码生成，无外部数据）。"""
    data = {}
    base = sum(ord(c) for c in spec.name) % 5 + 1
    for idx in itertools.product(*(range(d) for d in spec.shape)):
        data[idx] = base + sum((p + 1) * (v + 1) for p, v in enumerate(idx))
    return data


def direct_signed_sum(specs):
    """独立直算：枚举全部指标，每个张量按“自身轴序 -> 规范轴序”的
    显式置换计符号（每交换一对奇指标乘 -1），再对乘积求和。"""
    labels = []
    for t in specs:
        for a in t.axes:
            if a not in labels:
                labels.append(a)
    parity = {}
    for t in specs:
        if t.parities:
            parity.update(dict(zip(t.axes, t.parities)))
    sizes = {}
    for t in specs:
        sizes.update(dict(zip(t.axes, t.shape)))
    data = {t.name: make_values(t) for t in specs}
    total = 0
    for combo in itertools.product(*(range(sizes[a]) for a in labels)):
        pos = dict(zip(labels, combo))
        eps = 1
        prod = 1
        for t in specs:
            canon = [a for a in labels if a in t.axes]
            order = [t.axes.index(a) for a in canon]
            for x in range(len(order)):
                for y in range(x + 1, len(order)):
                    if (order[x] > order[y]
                            and parity.get(t.axes[order[x]], 0)
                            and parity.get(t.axes[order[y]], 0)):
                        eps = -eps
            prod *= data[t.name][tuple(pos[a] for a in t.axes)]
        total += eps * prod
    return total


def print_signed_plan(plan):
    print("  括号树:", plan.tree_text, " 净符号:", "%+d" % plan.sign)
    for s in plan.steps:
        print("  步骤%d: %s x %s -> %s%s" % (
            s.index, s.lhs.name, s.rhs.name, s.output_name, s.output_shape))
        print("        轴排列 lhs=%s rhs=%s 奇指标交换=%d 次 符号=%+d"
              % (s.lhs_axis_permutation, s.rhs_axis_permutation,
                 s.odd_swap_count, s.sign))


def main():
    print("=" * 70)
    print("1. 带奇偶的四环 A(a,b) B(c,b) C(c,d) D(d,a)，奇指标 a,b,d / 偶 c")
    print("   B 的 (c,b) 是奇-偶交换（无符号），D 的 (d,a) 是奇-奇交换（-1）")
    print("=" * 70)
    specs = [
        TensorSpec("A", ("a", "b"), (6, 4), (1, 1)),
        TensorSpec("B", ("c", "b"), (7, 4), (0, 1)),
        TensorSpec("C", ("c", "d"), (7, 3), (0, 1)),
        TensorSpec("D", ("d", "a"), (3, 6), (1, 1)),
    ]
    values = {t.name: make_values(t) for t in specs}
    want = direct_signed_sum(specs)

    plan = find_plan(specs, 10**9)
    print_signed_plan(plan)
    got = execute_plan(plan, values)[()]
    print("  数值核对: 计划执行 = %d；独立带符号直算 = %d；一致: %s"
          % (got, want, got == want))
    assert got == want

    print()
    print("=" * 70)
    print("2. 括号顺序不变性：上限 66 与 61 选出不同括号树，结果必须相同")
    print("=" * 70)
    for limit in (66, 61):
        p = find_plan(specs, limit)
        g = execute_plan(p, values)[()]
        print("  上限 %2d -> %s (乘法 %d, 峰值 %d, 净符号 %+d) 结果 %d 一致: %s"
              % (limit, p.tree_text, p.total_mult_count, p.peak_memory,
                 p.sign, g, g == want))
        assert g == want

    print()
    print("=" * 70)
    print("3. 迹收缩 A(i,j) B(j,i)：同一组数据，三种奇偶标记的符号")
    print("=" * 70)
    for parities in ((1, 1), (1, 0), (0, 0)):
        tr = [
            TensorSpec("A", ("i", "j"), (3, 2), parities),
            TensorSpec("B", ("j", "i"), (2, 3), (parities[1], parities[0])),
        ]
        p = find_plan(tr, 10**9)
        g = execute_plan(p, {t.name: make_values(t) for t in tr})[()]
        w = direct_signed_sum(tr)
        print("  奇偶(i,j)=%s -> 净符号 %+d，结果 %d（独立直算 %d，一致: %s）"
              % (parities, p.sign, g, w, g == w))
        assert g == w

    print()
    print("=" * 70)
    print("4. 无标记行为不变：同形状不标记，符号恒 +1")
    print("=" * 70)
    plain = [
        TensorSpec("A", ("a", "b"), (6, 4)),
        TensorSpec("B", ("c", "b"), (7, 4)),
        TensorSpec("C", ("c", "d"), (7, 3)),
        TensorSpec("D", ("d", "a"), (3, 6)),
    ]
    p = find_plan(plain, 10**9)
    g = execute_plan(p, {t.name: make_values(t) for t in plain})[()]
    w = direct_signed_sum(plain)
    print("  净符号 %+d，逐步交换次数 %s，结果 %d（一致: %s）"
          % (p.sign, [s.odd_swap_count for s in p.steps], g, g == w))
    assert g == w and p.sign == 1

    print()
    print("=" * 70)
    print("5. 实际触发的异常边界（不是预录文本）")
    print("=" * 70)
    try:
        bad = [
            TensorSpec("A", ("i", "j"), (2, 3), (1, 0)),
            TensorSpec("B", ("j", "k"), (3, 2), (1, 0)),  # j 既偶又奇
        ]
        find_plan(bad, 10**9)
    except InvalidProblemError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))


if __name__ == "__main__":
    main()

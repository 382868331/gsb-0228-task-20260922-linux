"""固定输入演示：真实运行 tensor_peak 规划器并核对数值。

运行：python demo.py（仅标准库，离线，约 1 秒内完成）。

内容：
1. 正常结果：四环矩阵网络收缩为标量，打印括号树、每步形状/乘法数/
   峰值驻留证据，并用独立的稠密执行器逐元素核对最终值。
2. 成本/内存权衡：同一网络在不同内存上限下选出不同最优计划。
3. 实际触发的拒绝边界：过紧的内存上限（经穷举证明无解）与 NaN 维大小。
"""

import itertools

from tensor_peak import (
    TensorSpec,
    find_plan,
    pareto_front,
    InvalidProblemError,
    MemoryLimitInfeasibleError,
)


def make_values(spec):
    """确定性生成稠密整数张量（仓库内代码生成，无外部数据）。"""
    data = {}
    base = sum(ord(c) for c in spec.name) % 7 + 1
    for idx in itertools.product(*(range(d) for d in spec.shape)):
        data[idx] = base + sum((p + 1) * (v + 1) for p, v in enumerate(idx))
    return data


def execute(plan):
    """按计划逐步执行稠密收缩，返回最终输出 {索引: 值}（独立于核心算法）。"""
    inputs = {t.name: (t.axes, make_values(t)) for t in plan.tensors}
    live = {}

    def view(name):
        if name in inputs:
            return inputs[name]
        return live[name]

    for step in plan.steps:
        ax_l, dat_l = view(step.lhs.name)
        ax_r, dat_r = view(step.rhs.name)
        sizes = dict(zip(step.lhs.axes, step.lhs.shape))
        sizes.update(zip(step.rhs.axes, step.rhs.shape))
        union_axes = list(ax_l) + [a for a in ax_r if a not in set(ax_l)]
        out = {}
        for combo in itertools.product(*(range(sizes[a]) for a in union_axes)):
            pos = dict(zip(union_axes, combo))
            kl = tuple(pos[a] for a in ax_l)
            kr = tuple(pos[a] for a in ax_r)
            ko = tuple(pos[a] for a in step.output_axes)
            out[ko] = out.get(ko, 0) + dat_l[kl] * dat_r[kr]
        live[step.output_name] = (step.output_axes, out)
        for op in (step.lhs, step.rhs):
            if op.kind == "intermediate":
                live.pop(op.name, None)
    last = plan.steps[-1].output_name
    return live[last][1]


def direct_sum(specs):
    """独立直算：枚举全部收缩轴，对四个原始矩阵逐项乘积求和。"""
    data = {t.name: make_values(t) for t in specs}
    sizes = {}
    for t in specs:
        sizes.update(dict(zip(t.axes, t.shape)))
    total = 0
    for combo in itertools.product(*(range(sizes[a]) for a in sizes)):
        pos = dict(zip(sizes, combo))
        prod = 1
        for t in specs:
            prod *= data[t.name][tuple(pos[a] for a in t.axes)]
        total += prod
    return total


def print_plan(plan):
    print("  括号树: ", plan.tree_text)
    print("  标量乘法总数:", plan.total_mult_count,
          " 峰值驻留(标量元素):", plan.peak_memory)
    for s in plan.steps:
        held = ", ".join("%s=%d" % (n, z) for n, z in s.residency[:-3])
        held_txt = (" 后台驻留[%s];" % held) if held else ""
        print("  步骤%d: %s x %s  (先求%s) 收缩轴%s -> %s 形状%s"
              % (s.index, s.lhs.name, s.rhs.name, s.evaluated_first,
                 s.contracted_axes, s.output_name, s.output_shape))
        print("        本步乘法=%d%s 收缩瞬间驻留%s 合计=%d"
              % (s.mult_count, held_txt,
                 [(n, z) for n, z in s.residency], s.peak_at_step))
    print("  峰值证据: 步骤%d 达到 %d <= 上限 %d"
          % (plan.peak_step, plan.peak_memory, plan.memory_limit))


def main():
    print("=" * 70)
    print("正常结果：4 个矩阵构成的环 A(a,b) B(b,c) C(c,d) D(d,a) -> 标量")
    print("=" * 70)
    specs = [
        TensorSpec("A", ("a", "b"), (2, 3)),
        TensorSpec("B", ("b", "c"), (3, 2)),
        TensorSpec("C", ("c", "d"), (2, 3)),
        TensorSpec("D", ("d", "a"), (3, 2)),
    ]
    plan = find_plan(specs, 10_000)
    print_plan(plan)

    got = execute(plan)[()]
    want = direct_sum(specs)
    print("  数值核对: 按计划逐步收缩 = %d；独立直算 = %d；一致: %s"
          % (got, want, got == want))
    assert got == want

    print()
    print("=" * 70)
    print("成本/内存权衡：a=6 b=4 c=7 d=3，Pareto 前沿")
    print("=" * 70)
    trade = [
        TensorSpec("A", ("a", "b"), (6, 4)),
        TensorSpec("B", ("b", "c"), (4, 7)),
        TensorSpec("C", ("c", "d"), (7, 3)),
        TensorSpec("D", ("d", "a"), (3, 6)),
    ]
    front = pareto_front(trade)
    print("  Pareto 前沿 (乘法数, 峰值):",
          [(p.mult_count, p.peak_memory) for p in front])
    for limit in (66, 65):
        p = find_plan(trade, limit)
        print("  上限 %d -> 选择 %d 乘法 / 峰值 %d  (%s)"
              % (limit, p.total_mult_count, p.peak_memory, p.tree_text))

    print()
    print("=" * 70)
    print("实际触发的拒绝/异常边界（不是预录文本）")
    print("=" * 70)
    try:
        find_plan(trade, 60)  # 最小可达峰值为 61，必须经穷举证明无解
    except MemoryLimitInfeasibleError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))

    try:
        TensorSpec("A", ("a", "b"), (2, float("nan")))
    except InvalidProblemError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))


if __name__ == "__main__":
    main()

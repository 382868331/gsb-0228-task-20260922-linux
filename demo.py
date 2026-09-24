"""固定输入演示：真实运行 tensor_peak 规划器并核对数值。

运行：python demo.py（仅标准库，离线，约 1 秒内完成）。

内容：
1. 正常结果：四环矩阵网络收缩为标量，打印括号树、每步形状/乘法数/
   峰值驻留证据，并用独立的稠密执行器逐元素核对最终值。
2. 成本/内存权衡：同一网络在不同内存上限下选出不同最优计划。
3. 可切片收缩轴：指定一个收缩轴与片宽，逐片沿用规划器，打印逐切片
   计划、累计乘法数、累加加法数、含常驻累加器的峰值，并逐元素核对
   “逐片收缩再累加”与独立直算一致；含单片/逐坐标边界。
4. 实际触发的拒绝边界：过紧的内存上限（经穷举证明无解）、切片上限
   不足、对输出轴切片与 NaN 维大小。
"""

import itertools

from tensor_peak import (
    TensorSpec,
    find_plan,
    find_sliced_plan,
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


def execute_sliced(sp, originals):
    """独立执行 SlicedPlan：逐片真实收缩（取该片坐标区间），输出累加。"""
    orig = {t.name: (t.axes, make_values(t)) for t in originals}
    total = {}
    for k, plan in enumerate(sp.slices):
        lo = k * sp.slice_size
        inputs = {}
        for st in plan.tensors:
            o_axes, o_dat = orig[st.name]
            pos = {a: q for q, a in enumerate(st.axes)}
            d = {}
            for idx in itertools.product(*(range(x) for x in st.shape)):
                o_idx = tuple(
                    lo + idx[pos[a]] if a == sp.sliced_axis
                    else idx[pos[a]]
                    for a in o_axes
                )
                d[idx] = o_dat[o_idx]
            inputs[st.name] = (st.axes, d)
        live = {}

        def view(name):
            return inputs[name] if name in inputs else live[name]

        for step in plan.steps:
            ax_l, dat_l = view(step.lhs.name)
            ax_r, dat_r = view(step.rhs.name)
            sizes = dict(zip(step.lhs.axes, step.lhs.shape))
            sizes.update(zip(step.rhs.axes, step.rhs.shape))
            union_axes = list(ax_l) + [a for a in ax_r if a not in set(ax_l)]
            out = {}
            for combo in itertools.product(*(range(sizes[a]) for a in union_axes)):
                p = dict(zip(union_axes, combo))
                kl = tuple(p[a] for a in ax_l)
                kr = tuple(p[a] for a in ax_r)
                ko = tuple(p[a] for a in step.output_axes)
                out[ko] = out.get(ko, 0) + dat_l[kl] * dat_r[kr]
            live[step.output_name] = (step.output_axes, out)
            for op in (step.lhs, step.rhs):
                if op.kind == "intermediate":
                    live.pop(op.name, None)
        partial = live[plan.steps[-1].output_name][1]
        for key, val in partial.items():
            total[key] = total.get(key, 0) + val
    return total


def direct_sum_general(specs, output_axes):
    """独立直算：枚举输出轴+全部收缩轴，对原始输入乘积求和（支持向量输出）。"""
    data = {t.name: make_values(t) for t in specs}
    sizes = {}
    for t in specs:
        sizes.update(dict(zip(t.axes, t.shape)))
    contracted = [
        a for a in sizes
        if sum(a in t.axes for t in specs) == 2
    ]
    result = {}
    enum_axes = list(output_axes) + contracted
    for combo in itertools.product(*(range(sizes[a]) for a in enum_axes)):
        p = dict(zip(enum_axes, combo))
        prod = 1
        for t in specs:
            prod *= data[t.name][tuple(p[a] for a in t.axes)]
        key = tuple(p[a] for a in output_axes)
        result[key] = result.get(key, 0) + prod
    return result


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
    print("可切片收缩轴：A(i,a) B(a,j) C(j,k) D(k,i)，a=4 沿 a 切 2 片")
    print("=" * 70)
    sq = [
        TensorSpec("A", ("i", "a"), (2, 4)),
        TensorSpec("B", ("a", "j"), (4, 3)),
        TensorSpec("C", ("j", "k"), (3, 2)),
        TensorSpec("D", ("k", "i"), (2, 2)),
    ]
    sp = find_sliced_plan(sq, 10_000, "a", 2)
    print("  切片轴=%s 片宽=%d 切片数=%d 最终输出轴=%s 形状=%s（%d 个元素）"
          % (sp.sliced_axis, sp.slice_size, sp.num_slices,
             sp.output_axes, sp.output_shape, sp.output_size))
    for k, p in enumerate(sp.slices):
        print("  切片%d: %s" % (k, p.tree_text))
        print("         乘法=%d 片内峰值=%d（片内上限=%d）"
              % (p.total_mult_count, p.peak_memory, p.memory_limit))
    print("  累计标量乘法=%d；累加标量加法=(%d片-1)*%d输出=%d"
          % (sp.total_mult_count, sp.num_slices, sp.output_size,
             sp.total_add_count))
    print("  整体峰值（首片峰值 vs 累加器%d+其余片峰值）=%d"
          % (sp.output_size, sp.peak_memory))
    got = execute_sliced(sp, sq)
    want = direct_sum_general(sq, sp.output_axes)
    print("  数值核对: 逐片收缩再累加 = %d；独立直算 = %d；一致: %s"
          % (got[()], want[()], got == want))
    assert got == want

    print()
    print("-" * 70)
    print("切片边界：片宽=轴维（退化为 1 片，0 加法）与片宽=1（逐坐标）")
    print("-" * 70)
    one = find_sliced_plan(sq, 10_000, "a", 4)
    print("  片宽 4：切片数=%d 加法=%d 乘法=%d（不分片乘法=%d）"
          % (one.num_slices, one.total_add_count, one.total_mult_count,
             find_plan(sq, 10_000).total_mult_count))
    each = find_sliced_plan(sq, 10_000, "a", 1)
    print("  片宽 1：切片数=%d 逐片乘法=%s 累计乘法=%d 加法=%d"
          % (each.num_slices,
             [p.total_mult_count for p in each.slices],
             each.total_mult_count, each.total_add_count))
    assert execute_sliced(each, sq) == direct_sum_general(sq, ())

    print()
    print("=" * 70)
    print("实际触发的拒绝/异常边界（不是预录文本）")
    print("=" * 70)
    try:
        find_plan(trade, 60)  # 最小可达峰值为 61，必须经穷举证明无解
    except MemoryLimitInfeasibleError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))

    # 切片：首片可行但扣除常驻累加器后第二片无可行计划。
    try:
        small = [
            TensorSpec("A", ("i", "a"), (2, 4)),
            TensorSpec("B", ("a", "j"), (4, 3)),
        ]
        free = find_sliced_plan(small, 10_000, "a", 2)
        find_sliced_plan(small, free.peak_memory - 1, "a", 2)
    except MemoryLimitInfeasibleError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))

    # 切片：对只出现一次的输出轴切片（非法输入）。
    try:
        find_sliced_plan(
            [TensorSpec("A", ("i", "a"), (2, 4)),
             TensorSpec("B", ("a", "j"), (4, 3))],
            10_000, "i", 2,
        )
    except InvalidProblemError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))

    try:
        TensorSpec("A", ("a", "b"), (2, float("nan")))
    except InvalidProblemError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))


if __name__ == "__main__":
    main()

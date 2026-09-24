"""固定输入演示：真实运行 tensor_peak 规划器并核对数值。

运行：python demo.py（仅标准库，离线，约 1 秒内完成）。

内容：
1. 正常结果：四环矩阵网络收缩为标量，打印括号树、每步形状/乘法数/
   峰值驻留证据，并用独立的稠密执行器逐元素核对最终值。
2. 成本/内存权衡：同一网络在不同内存上限下选出不同最优计划。
3. 单维切块：沿收缩轴/输出轴切块，打印逐切片计划、累计计算量（含输出
   累加成本）与峰值中间存储，逐切片执行并跨片合并后与独立直算逐元素核对；
   含单位块边界。
4. 实际触发的拒绝边界：过紧的内存上限（经穷举证明无解）、NaN 维大小，
   以及切块特有的块长和不符、某切片在紧上限下不可行。
"""

import itertools

from tensor_peak import (
    TensorSpec,
    SliceDim,
    find_plan,
    plan_slices,
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


def cell_value(name, idx):
    """与切片无关的确定性整数值（只依赖名字与全局索引）。"""
    base = sum(ord(c) for c in name) % 7 + 1
    return base + sum((p + 1) * (v + 1) for p, v in enumerate(idx))


def full_data(t):
    return {idx: cell_value(t.name, idx)
            for idx in itertools.product(*(range(d) for d in t.shape))}


def execute_on_data(tensors, steps, output_axes, data):
    """用给定输入数据按计划逐步收缩，返回 {输出索引: 值}（不调用核心算法）。"""
    by_name = {t.name: t for t in tensors}
    live = {}

    def view(name):
        if name in by_name:
            return by_name[name].axes, data[name]
        return live[name]

    for step in steps:
        ax_l, dat_l = view(step.lhs.name)
        ax_r, dat_r = view(step.rhs.name)
        sm = dict(zip(step.lhs.axes, step.lhs.shape))
        sm.update(zip(step.rhs.axes, step.rhs.shape))
        union = list(ax_l) + [a for a in ax_r if a not in set(ax_l)]
        out = {}
        for combo in itertools.product(*(range(sm[a]) for a in union)):
            pos = dict(zip(union, combo))
            kl = tuple(pos[a] for a in ax_l)
            kr = tuple(pos[a] for a in ax_r)
            ko = tuple(pos[a] for a in output_axes)
            out[ko] = out.get(ko, 0) + dat_l[kl] * dat_r[kr]
        live[step.output_name] = (output_axes, out)
        for op in (step.lhs, step.rhs):
            if op.kind == "intermediate":
                live.pop(op.name, None)
    if not steps:
        return data[tensors[0].name]
    return live[steps[-1].output_name][1]


def slice_data(specs, axis, chunk, offset):
    """取该片输入：axis 用局部索引 0..chunk-1，映射到全局 offset+local。"""
    data = {}
    for t in specs:
        shape = tuple(
            chunk if a == axis else d for a, d in zip(t.axes, t.shape)
        )
        p = t.axes.index(axis) if axis in t.axes else None
        d = {}
        for idx in itertools.product(*(range(x) for x in shape)):
            gidx = idx if p is None else tuple(
                offset + v if q == p else v for q, v in enumerate(idx)
            )
            d[idx] = cell_value(t.name, gidx)
        data[t.name] = d
    return data


def direct_full(specs, out_axes):
    """独立直算完整问题：枚举输出轴+全部收缩轴逐项乘积求和。"""
    data = {t.name: full_data(t) for t in specs}
    sizes = {}
    for t in specs:
        sizes.update(dict(zip(t.axes, t.shape)))
    contract = [a for a in sizes if sum(a in t.axes for t in specs) == 2]
    enum = list(out_axes) + contract
    res = {}
    for combo in itertools.product(*(range(sizes[a]) for a in enum)):
        pos = dict(zip(enum, combo))
        v = 1
        for t in specs:
            v *= data[t.name][tuple(pos[a] for a in t.axes)]
        key = tuple(pos[a] for a in out_axes)
        res[key] = res.get(key, 0) + v
    return res


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
    print("单维切块：A(i,k) B(k,j)，i=2 k=6 j=3，沿收缩轴 k 切成 (2,4)")
    print("=" * 70)
    mats = [
        TensorSpec("A", ("i", "k"), (2, 6)),
        TensorSpec("B", ("k", "j"), (6, 3)),
    ]
    sp = plan_slices(mats, 10_000, SliceDim("k", (2, 4)))
    for s, p in enumerate(sp.slices):
        print("  切片%d (k=%d): %s  乘法=%d 峰值中间存储=%d"
              % (s, sp.slice_sizes[s], p.tree_text,
                 p.total_mult_count, p.peak_memory))
    print("  各片乘法和=%d，输出累加成本=(2-1)*2*3=%d，累计计算量=%d"
          % (sum(p.total_mult_count for p in sp.slices),
             sp.accumulator_cost, sp.total_mult_count))
    print("  峰值中间存储=max(16,26)=%d（切片 %d 达到）"
          % (sp.peak_intermediate_storage, sp.peak_slice))

    # 逐切片执行 + 跨片累加，与独立直算逐元素 ==。
    got, offset = {}, 0
    for s, c in enumerate(sp.slice_sizes):
        data = slice_data(mats, "k", c, offset)
        offset += c
        part = execute_on_data(
            sp.slices[s].tensors, sp.slices[s].steps,
            sp.slices[s].output_axes, data,
        )
        for key, v in part.items():
            got[key] = got.get(key, 0) + v
    want = direct_full(mats, ("i", "j"))
    print("  数值核对: 逐切片收缩+跨片累加 == 独立直算: %s" % (got == want))
    assert got == want

    # 切输出轴：i 切成 (1,1)，各片不相交分块，零累加。
    spo = plan_slices(mats, 10_000, SliceDim("i", (1, 1)))
    print("  沿输出轴 i=(1,1): 各片乘法=%s 累计=%d 累加成本=%d 峰值=%d，"
          "分块形状=%s"
          % ([p.total_mult_count for p in spo.slices], spo.total_mult_count,
             spo.accumulator_cost, spo.peak_intermediate_storage,
             [p.output_shape for p in spo.slices]))

    # 边界：k 全切成单位长。
    spu = plan_slices(mats, 10_000, SliceDim("k", (1, 1, 1, 1, 1, 1)))
    print("  边界 k=6 个单位块: 片数=%d 每片乘法=%d 每片峰值=2+3+6=%d "
          "累计(含累加)=%d"
          % (len(spu.slices), spu.slices[0].total_mult_count,
             spu.slices[0].peak_memory, spu.total_mult_count))

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

    # 切块特有的非法输入：块长之和不等于维大小。
    try:
        plan_slices(mats, 10_000, SliceDim("k", (2, 3)))  # 和 5 != 6
    except InvalidProblemError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))

    # 切块特有的失败边界：某片在过紧上限下经穷举证明无解（片 k=4 峰值 26）。
    try:
        plan_slices(mats, 25, SliceDim("k", (2, 4)))
    except MemoryLimitInfeasibleError as e:
        print("  [捕获 %s] %s" % (type(e).__name__, e))


if __name__ == "__main__":
    main()

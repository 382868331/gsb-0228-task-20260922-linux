"""数值核对：用独立的稠密逐元素执行器执行 find_plan 给出的步骤，
并与完全独立的直接枚举求和结果逐元素比对。

执行器不调用核心算法，只按计划里的轴标签/形状做确定性整数运算；
所有值都是精确整数，因此比对是 == 而不是容差比较。
"""

import itertools
import unittest

from tensor_peak import TensorSpec, find_plan


def make_values(spec):
    """确定性生成稠密张量数据：dict[索引元组] -> 正整数。

    值 = 1 + 各索引与张量名 ord 的加权和；不同张量、不同位置取不同值。
    """
    data = {}
    base = sum(ord(c) for c in spec.name) % 7 + 1
    for idx in itertools.product(*(range(d) for d in spec.shape)):
        data[idx] = (base + sum((p + 1) * (v + 1) for p, v in enumerate(idx)))
    return data


def execute_plan(plan):
    """按步骤执行收缩，返回最终输出 {输出索引元组: 值}。

    原始输入“按需载入”（这里直接全部在内存，模拟语义时按步取用）；
    中间结果产出后即只被引用一次——执行顺序由计划给出。
    """
    inputs = {t.name: (t.axes, make_values(t)) for t in plan.tensors}
    live = {}  # name -> (axes, dict)

    def view(name):
        if name in inputs:
            return inputs[name]
        return live[name]

    for step in plan.steps:
        ax_l, dat_l = view(step.lhs.name)
        ax_r, dat_r = view(step.rhs.name)
        all_axes = list(ax_l) + [a for a in ax_r if a not in set(ax_l)]
        size_map = dict(zip(step.lhs.axes, step.lhs.shape))
        size_map.update(dict(zip(step.rhs.axes, step.rhs.shape)))
        out = {}
        for combo in itertools.product(
            *[range(size_map[a]) for a in all_axes]
        ):
            pos = dict(zip(all_axes, combo))
            kl = tuple(pos[a] for a in ax_l)
            kr = tuple(pos[a] for a in ax_r)
            ko = tuple(pos[a] for a in step.output_axes)
            out[ko] = out.get(ko, 0) + dat_l[kl] * dat_r[kr]
        live[step.output_name] = (step.output_axes, out)
        # 释放被本步消费的中间量（模拟计划的生命周期；输入视为可重载）。
        for op in (step.lhs, step.rhs):
            if op.kind == "intermediate" and op.name in live:
                del live[op.name]
    final_name = plan.steps[-1].output_name
    return live[final_name][1]


def direct_reference(specs, output_axes):
    """独立直算：枚举输出轴+全部收缩轴，对原始输入乘积求和。"""
    sizes = {}
    owners = {}
    for t in specs:
        for a, d in zip(t.axes, t.shape):
            sizes[a] = d
            owners.setdefault(a, []).append(t)
    contraction_axes = [
        a for a in sizes
        if sum(a in t.axes for t in specs) == 2
    ]
    data = {t.name: make_values(t) for t in specs}
    result = {}
    enum_axes = list(output_axes) + contraction_axes
    for combo in itertools.product(*[range(sizes[a]) for a in enum_axes]):
        pos = dict(zip(enum_axes, combo))
        prod = 1
        for t in specs:
            key = tuple(pos[a] for a in t.axes)
            prod *= data[t.name][key]
        out_key = tuple(pos[a] for a in output_axes)
        result[out_key] = result.get(out_key, 0) + prod
    return result


class TestNumeric(unittest.TestCase):
    def test_shared_two_axes_values(self):
        specs = [
            TensorSpec("A", ("i", "j", "k"), (2, 3, 4)),
            TensorSpec("B", ("j", "k", "l"), (3, 4, 2)),
        ]
        plan = find_plan(specs, 10**9)
        got = execute_plan(plan)
        want = direct_reference(specs, plan.output_axes)
        self.assertEqual(got, want)

    def test_ring4_values_and_tradeoff_plans(self):
        # 小尺寸四环，两种内存上限会选出不同括号树，逐元素都必须正确。
        specs = [
            TensorSpec("A", ("a", "b"), (2, 3)),
            TensorSpec("B", ("b", "c"), (3, 2)),
            TensorSpec("C", ("c", "d"), (2, 3)),
            TensorSpec("D", ("d", "a"), (3, 2)),
        ]
        want = direct_reference(specs, ())  # 完全收缩为标量
        plan = find_plan(specs, 10**6)
        got = execute_plan(plan)
        self.assertEqual(got, {(): want[()]})

    def test_chain3_with_output_axis_values(self):
        specs = [
            TensorSpec("A", ("i", "k"), (2, 3)),
            TensorSpec("B", ("k", "j"), (3, 4)),
            TensorSpec("C", ("j", "l"), (4, 2)),
        ]
        plan = find_plan(specs, 10**9)
        got = execute_plan(plan)
        want = direct_reference(specs, ("i", "l"))
        self.assertEqual(set(got), set(want))
        self.assertEqual(got, want)

    def test_mult_count_matches_executor_work(self):
        specs = [
            TensorSpec("A", ("i", "j", "k"), (2, 3, 4)),
            TensorSpec("B", ("j", "k", "l"), (3, 4, 5)),
            TensorSpec("C", ("l", "m"), (5, 2)),
        ]
        plan = find_plan(specs, 10**9)
        # 执行器每次最内层乘积即一次标量乘法；总数应与计划一致。
        counted = 0
        for step in plan.steps:
            counted += step.mult_count
        self.assertEqual(counted, plan.total_mult_count)


if __name__ == "__main__":
    unittest.main()

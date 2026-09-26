"""指标奇偶标记的测试：带符号收缩的独立参考交叉验证与异常路径。

参考实现 reference_signed_sum 在本文件内独立写成：对全部指标做原始
全指标求和，每个输入张量按“自身轴序 -> 规范轴序”的显式置换计符号
（每交换一对奇指标乘 -1），不调用 tensor_peak 的任何内部算法；被测
核心仅通过公开接口 find_plan / execute_plan 被调用。
"""

import itertools
import unittest

from tensor_peak import (
    TensorSpec,
    find_plan,
    pareto_front,
    execute_plan,
    InvalidProblemError,
)


def make_values(spec):
    """确定性生成稠密整数数据（与核心无关的独立生成器）。"""
    data = {}
    base = sum(ord(c) for c in spec.name) % 5 + 1
    for idx in itertools.product(*(range(d) for d in spec.shape)):
        data[idx] = base + sum((p + 1) * (v + 1) for p, v in enumerate(idx))
    return data


def values_of(specs):
    return {t.name: make_values(t) for t in specs}


def canonical_order(specs):
    order = []
    for t in specs:
        for a in t.axes:
            if a not in order:
                order.append(a)
    return order


def explicit_perm_sign(axes, canon, parity):
    """显式置换符号：把 axes 排成 canon 所需交换中，奇-奇指标对次数的
    奇偶性；偶指标参与或奇-偶交换均不计。"""
    order = [axes.index(a) for a in canon]
    swaps = 0
    for x in range(len(order)):
        for y in range(x + 1, len(order)):
            if (order[x] > order[y] and parity.get(axes[order[x]], 0)
                    and parity.get(axes[order[y]], 0)):
                swaps += 1
    return -1 if swaps % 2 else 1


def reference_signed_sum(specs, parity):
    """独立参考：枚举全部指标，对带显式置换符号的输入乘积求和。

    返回 {规范输出轴序索引元组: 值}。不调用被测核心。
    """
    labels = canonical_order(specs)
    sizes = {}
    for t in specs:
        sizes.update(dict(zip(t.axes, t.shape)))
    counts = {a: sum(a in t.axes for t in specs) for a in labels}
    out_axes = tuple(a for a in labels if counts[a] == 1)
    data = values_of(specs)
    result = {}
    for combo in itertools.product(*(range(sizes[a]) for a in labels)):
        pos = dict(zip(labels, combo))
        eps = 1
        prod = 1
        for t in specs:
            canon = [a for a in labels if a in t.axes]
            eps *= explicit_perm_sign(t.axes, canon, parity)
            prod *= data[t.name][tuple(pos[a] for a in t.axes)]
        key = tuple(pos[a] for a in out_axes)
        result[key] = result.get(key, 0) + eps * prod
    return result


def parity_map(specs):
    parity = {}
    for t in specs:
        if t.parities:
            parity.update(dict(zip(t.axes, t.parities)))
    return parity


def check_matches_reference(test, specs, plan):
    got = execute_plan(plan, values_of(specs))
    want = reference_signed_sum(specs, parity_map(specs))
    test.assertEqual(got, want)
    return got


class TestParitySigns(unittest.TestCase):
    def test_two_odd_swaps_give_plus(self):
        # 规范序由先出现的 B 建立为 (b,d,c,a,e)；A 的轴序 (c,a,b) 相对
        # 规范子序 (b,c,a) 需交换两对奇指标，符号 (-1)^2 = +1。
        parity = {"a": 1, "b": 1, "c": 1, "d": 0, "e": 0}
        specs = [
            TensorSpec("B", ("b", "d"), (2, 2), (parity["b"], parity["d"])),
            TensorSpec("A", ("c", "a", "b"), (2, 3, 2),
                       (parity["c"], parity["a"], parity["b"])),
            TensorSpec("C", ("d", "a", "e"), (2, 3, 2),
                       (parity["d"], parity["a"], parity["e"])),
            TensorSpec("F", ("e", "c"), (2, 2), (parity["e"], parity["c"])),
        ]
        plan = find_plan(specs, 10**9)
        # 含 A 的收缩步必须记录 2 次奇指标交换、符号 +1。
        step_with_a = next(
            s for s in plan.steps if s.lhs.name == "A" or s.rhs.name == "A"
        )
        if step_with_a.lhs.name == "A":
            ref_a = step_with_a.lhs
            perm_a = step_with_a.lhs_axis_permutation
        else:
            ref_a = step_with_a.rhs
            perm_a = step_with_a.rhs_axis_permutation
        self.assertEqual(tuple(ref_a.axes[i] for i in perm_a), ("b", "c", "a"))
        self.assertEqual(step_with_a.odd_swap_count, 2)
        self.assertEqual(step_with_a.sign, 1)
        check_matches_reference(self, specs, plan)

    def test_even_interleave_no_sign(self):
        # 规范序 (i,j,x)；A 轴序 (j,x,i)：两次相邻交换中只有一次是
        # 奇-奇交换（偶指标 x 穿插不计），符号 -1 而非免费重排。
        parity = {"i": 1, "j": 1, "x": 0, "k": 0}
        specs = [
            TensorSpec("B", ("i", "j", "x", "k"), (2, 2, 2, 2),
                       (parity["i"], parity["j"], parity["x"], parity["k"])),
            TensorSpec("A", ("j", "x", "i"), (2, 2, 2),
                       (parity["j"], parity["x"], parity["i"])),
            TensorSpec("C", ("k",), (2,), (parity["k"],)),
        ]
        plan = find_plan(specs, 10**9)
        step_with_a = next(
            s for s in plan.steps if s.lhs.name == "A" or s.rhs.name == "A"
        )
        self.assertEqual(step_with_a.odd_swap_count, 1)
        self.assertEqual(step_with_a.sign, -1)
        check_matches_reference(self, specs, plan)

    def test_trace_contraction_odd(self):
        # 迹收缩 A(i,j) B(j,i)，i,j 皆奇：B 需一次奇-奇交换，符号 -1。
        specs = [
            TensorSpec("A", ("i", "j"), (3, 2), (1, 1)),
            TensorSpec("B", ("j", "i"), (2, 3), (1, 1)),
        ]
        plan = find_plan(specs, 10**9)
        self.assertEqual(plan.output_axes, ())
        self.assertEqual(plan.steps[0].odd_swap_count, 1)
        self.assertEqual(plan.steps[0].sign, -1)
        self.assertEqual(plan.sign, -1)
        got = check_matches_reference(self, specs, plan)
        data = values_of(specs)
        unsigned = sum(
            data["A"][(i, j)] * data["B"][(j, i)]
            for i in range(3) for j in range(2)
        )
        self.assertEqual(got[()], -unsigned)

    def test_trace_contraction_even_and_mixed(self):
        # 全偶：符号 +1；奇-偶混合：交换的不是两个奇指标，符号仍 +1。
        for parities, want_sign in (((0, 0), 1), ((1, 0), 1), ((1, 1), -1)):
            specs = [
                TensorSpec("A", ("i", "j"), (3, 2), parities),
                TensorSpec("B", ("j", "i"), (2, 3),
                           (parities[1], parities[0])),
            ]
            plan = find_plan(specs, 10**9)
            self.assertEqual(plan.sign, want_sign)
            check_matches_reference(self, specs, plan)

    def test_bracket_orders_agree(self):
        # 同一带奇偶网络在不同内存上限下选出不同括号树，带符号执行
        # 结果必须与独立参考完全一致（转置不是免费重排）。
        # 四环：B 的 (c,b) 是奇-偶交换（无符号），D 的 (d,a) 是奇-奇
        # 交换（符号 -1）；a=6 b=4 c=7 d=3 时 Pareto 前沿有两个点。
        parity = {"a": 1, "b": 1, "c": 0, "d": 1}
        specs = [
            TensorSpec("A", ("a", "b"), (6, 4),
                       (parity["a"], parity["b"])),
            TensorSpec("B", ("c", "b"), (7, 4),
                       (parity["c"], parity["b"])),
            TensorSpec("C", ("c", "d"), (7, 3),
                       (parity["c"], parity["d"])),
            TensorSpec("D", ("d", "a"), (3, 6),
                       (parity["d"], parity["a"])),
        ]
        front = pareto_front(specs)
        self.assertGreaterEqual(len(front), 2)
        plans = {}
        first = find_plan(specs, 10**9)
        plans[first.tree_text] = first
        for point in front:
            p = find_plan(specs, point.peak_memory)
            plans[p.tree_text] = p
        self.assertGreaterEqual(len(plans), 2)
        want = reference_signed_sum(specs, parity)
        for text, p in plans.items():
            got = execute_plan(p, values_of(specs))
            self.assertEqual(got, want, "括号树 %s 的带符号结果不一致" % text)
            # 计划净符号 = 各步符号之积。
            prod = 1
            for s in p.steps:
                prod *= s.sign
            self.assertEqual(p.sign, prod)

    def test_unmarked_behavior_unchanged(self):
        # 完全不做奇偶标记：符号恒 +1、交换次数恒 0，数值等于无符号直算。
        specs = [
            TensorSpec("A", ("c", "a", "b"), (2, 3, 2)),
            TensorSpec("B", ("b", "d"), (2, 2)),
            TensorSpec("C", ("d", "a", "e"), (2, 3, 2)),
            TensorSpec("F", ("e", "c"), (2, 2)),
        ]
        plan = find_plan(specs, 10**9)
        self.assertEqual(plan.sign, 1)
        for s in plan.steps:
            self.assertEqual(s.sign, 1)
            self.assertEqual(s.odd_swap_count, 0)
        got = execute_plan(plan, values_of(specs))
        want = reference_signed_sum(specs, {})
        self.assertEqual(got, want)

    def test_single_tensor_trivial_plan(self):
        t = TensorSpec("S", ("b", "a"), (2, 3), (1, 1))
        plan = find_plan([t], 1)
        self.assertEqual(plan.steps, ())
        got = execute_plan(plan, values_of([t]))
        # 单张量时规范序即其自身轴序，无转置、无符号。
        self.assertEqual(got, make_values(t))


class TestParityInvalidInputs(unittest.TestCase):
    def test_parity_length_mismatch(self):
        with self.assertRaises(InvalidProblemError):
            TensorSpec("A", ("i", "j"), (2, 3), (1,))

    def test_parity_value_rejected(self):
        for bad in (2, -1, True, "x", 1.0):
            with self.assertRaises(InvalidProblemError):
                TensorSpec("A", ("i",), (2,), (bad,))

    def test_parity_not_sequence(self):
        with self.assertRaises(InvalidProblemError):
            TensorSpec("A", ("i",), (2,), 1)

    def test_conflicting_parity_across_tensors(self):
        specs = [
            TensorSpec("A", ("i", "j"), (2, 3), (1, 0)),
            TensorSpec("B", ("j", "k"), (3, 2), (1, 0)),  # j 既偶又奇
        ]
        with self.assertRaises(InvalidProblemError):
            find_plan(specs, 10**9)

    def test_partial_marking_is_allowed(self):
        # 只有部分张量提供奇偶：其余标签按偶处理，不报错。
        specs = [
            TensorSpec("A", ("i", "j"), (2, 3), (1, 1)),
            TensorSpec("B", ("j", "i"), (3, 2)),
        ]
        plan = find_plan(specs, 10**9)
        self.assertEqual(plan.sign, -1)
        check_matches_reference(self, specs, plan)

    def test_execute_plan_bad_inputs(self):
        specs = [
            TensorSpec("A", ("i", "j"), (2, 2), (1, 1)),
            TensorSpec("B", ("j", "i"), (2, 2), (1, 1)),
        ]
        plan = find_plan(specs, 10**9)
        with self.assertRaises(InvalidProblemError):
            execute_plan("not a plan", {})
        with self.assertRaises(InvalidProblemError):
            execute_plan(plan, {"A": {(0, 0): 1}})  # 缺 B
        with self.assertRaises(InvalidProblemError):
            execute_plan(plan, {"A": {(0,): 1}, "B": {(0, 0): 1}})  # 键维数错
        with self.assertRaises(InvalidProblemError):
            execute_plan(plan, {"A": {(0, 5): 1}, "B": {(0, 0): 1}})  # 越界


if __name__ == "__main__":
    unittest.main()

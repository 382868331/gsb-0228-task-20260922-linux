"""tensor_peak 的必要测试。

参考期望值来自 tests/reference_enum.py（独立全二叉树枚举，不调用被测
核心）；数值核对见 tests/test_numeric.py。被测核心只通过公开接口调用。
"""

import math
import random
import socket
import unittest

from tests import reference_enum as ref

import tensor_peak as tp
from tensor_peak import (
    TensorSpec,
    find_plan,
    pareto_front,
    InvalidProblemError,
    DisconnectedNetworkError,
    MemoryLimitInfeasibleError,
    BudgetExhaustedError,
    InfeasibleError,
)


def spec(name, axes, shape):
    return TensorSpec(name, tuple(axes), tuple(shape))


# ---------------------------------------------------------------------------
# 计划结构不变量（不参考核心内部，仅按公开字段核对）
# ---------------------------------------------------------------------------


def assert_plan_consistent(test, plan, specs, label_size):
    names = {t.name for t in specs}
    # 括号树叶子恰好覆盖全部输入且各一次。
    leaves = []

    def walk(tree):
        if tree[0] == "leaf":
            leaves.append(tree[1])
            return
        walk(tree[2])
        walk(tree[3])

    walk(plan.tree)
    test.assertEqual(sorted(leaves), sorted(names))

    produced = set()
    total = 0
    for s in plan.steps:
        # 操作数要么原始输入，要么此前步骤产出。
        for op in (s.lhs, s.rhs):
            if op.kind == "input":
                test.assertIn(op.name, names)
                src = next(t for t in specs if t.name == op.name)
                test.assertEqual(op.axes, src.axes)
                test.assertEqual(op.shape, src.shape)
                test.assertEqual(op.size, math.prod(src.shape))
            else:
                test.assertIn(op.name, produced)
                test.assertLess(op.defining_step, s.index)
            test.assertEqual(op.size, math.prod(op.shape))
            test.assertEqual(op.shape, tuple(label_size[a] for a in op.axes))
        # 每步形状：输出轴 = 两侧并集减收缩轴，收缩轴必须两侧共有。
        shared = set(s.lhs.axes) & set(s.rhs.axes)
        test.assertTrue(shared)
        test.assertEqual(set(s.contracted_axes), shared)
        test.assertEqual(
            set(s.output_axes),
            (set(s.lhs.axes) | set(s.rhs.axes)) - shared,
        )
        test.assertEqual(
            s.mult_count,
            math.prod(
                label_size[a]
                for a in set(s.lhs.axes) | set(s.rhs.axes)
            ),
        )
        test.assertEqual(s.peak_at_step, sum(z for _, z in s.residency))
        test.assertEqual(s.output_shape,
                         tuple(label_size[a] for a in s.output_axes))
        # 驻留清单中名字唯一、恰好包含两个操作数与新输出。
        rnames = [n for n, _ in s.residency]
        test.assertEqual(len(rnames), len(set(rnames)))
        self_names = {s.lhs.name, s.rhs.name, s.output_name}
        test.assertEqual(set(rnames) & self_names, self_names)
        for n, z in s.residency:
            test.assertGreaterEqual(z, 1)
        produced.add(s.output_name)
        total += s.mult_count

    test.assertEqual(total, plan.total_mult_count)
    test.assertEqual(plan.peak_memory,
                     max((s.peak_at_step for s in plan.steps), default=0))
    test.assertLessEqual(plan.peak_memory, plan.memory_limit)
    if plan.steps:
        test.assertIn(plan.peak_step,
                      {s.index for s in plan.steps
                       if s.peak_at_step == plan.peak_memory})
    # 最终输出轴恰为只出现一次的标签。
    counts = {}
    for t in specs:
        for a in t.axes:
            counts[a] = counts.get(a, 0) + 1
    test.assertEqual(set(plan.output_axes),
                     {a for a, c in counts.items() if c == 1})
    test.assertEqual(plan.output_shape,
                     tuple(label_size[a] for a in plan.output_axes))


class TestBasic(unittest.TestCase):
    def test_matrix_chain_3(self):
        # A(i,k) B(k,j) C(j,l): i=10 k=20 j=30 l=40
        dims = {"i": 10, "k": 20, "j": 30, "l": 40}
        specs = [
            spec("A", ("i", "k"), (10, 20)),
            spec("B", ("k", "j"), (20, 30)),
            spec("C", ("j", "l"), (30, 40)),
        ]
        plan = find_plan(specs, 10**9)
        # (AB)C: 10*20*30 + 10*30*40 = 6000+12000；A(BC) 更大。
        self.assertEqual(plan.total_mult_count, 18000)
        self.assertEqual(
            ref.best_muls_under_limit([t.axes for t in specs], dims, 10**9),
            18000,
        )
        assert_plan_consistent(self, plan, specs, dims)
        self.assertEqual(plan.output_axes, ("i", "l"))
        self.assertEqual(plan.output_shape, (10, 40))

    def test_scalar_dot_product(self):
        specs = [spec("v", ("i",), (6,)), spec("w", ("i",), (6,))]
        plan = find_plan(specs, 10**6)
        self.assertEqual(plan.total_mult_count, 6)
        self.assertEqual(plan.output_axes, ())
        self.assertEqual(plan.output_shape, ())
        self.assertEqual(plan.steps[0].peak_at_step, 6 + 6 + 1)

    def test_single_tensor_trivial(self):
        t = spec("S", ("a", "b"), (3, 4))
        plan = find_plan([t], 1)
        self.assertEqual(plan.total_mult_count, 0)
        self.assertEqual(plan.peak_memory, 0)
        self.assertEqual(plan.steps, ())
        self.assertEqual(plan.output_axes, ("a", "b"))
        self.assertEqual(plan.peak_step, None)

    def test_shared_multiple_axes(self):
        # A(i,j,k) B(j,k,l) 一次收缩共享 j,k 两个轴；再与 C(l,m) 相连。
        dims = {"i": 2, "j": 3, "k": 4, "l": 5, "m": 6}
        specs = [
            spec("A", ("i", "j", "k"), (2, 3, 4)),
            spec("B", ("j", "k", "l"), (3, 4, 5)),
            spec("C", ("l", "m"), (5, 6)),
        ]
        plan = find_plan(specs, 10**9)
        expect = ref.best_muls_under_limit([t.axes for t in specs], dims, 10**9)
        self.assertEqual(plan.total_mult_count, expect)
        # 手工核对：先 AB（2*3*4*5=120）再与 C（2*5*6=60）= 180。
        self.assertEqual(plan.total_mult_count, 180)
        assert_plan_consistent(self, plan, specs, dims)
        first = plan.steps[0]
        self.assertEqual(set(first.contracted_axes), {"j", "k"})
        self.assertEqual(set(first.output_axes), {"i", "l"})

    def test_disconnected_rejected(self):
        specs = [
            spec("A", ("i", "k"), (2, 3)),
            spec("B", ("k", "j"), (3, 4)),
            spec("D", ("p", "q"), (2, 3)),
            spec("E", ("q", "r"), (3, 4)),
        ]
        with self.assertRaises(DisconnectedNetworkError):
            find_plan(specs, 10**9)
        with self.assertRaises(InfeasibleError):
            find_plan(specs, 10**9)
        with self.assertRaises(tp.TensorPlanningError):
            find_plan(specs, 10**9)

    def test_cost_memory_tradeoff_ring4(self):
        # 独立枚举固化的四环实例 a=6 b=4 c=7 d=3：
        # 168 乘/峰值 66（平衡树） 与 174 乘/峰值 61（左结合）互为 Pareto。
        dims = {"a": 6, "b": 4, "c": 7, "d": 3}
        specs = [
            spec("A", ("a", "b"), (6, 4)),
            spec("B", ("b", "c"), (4, 7)),
            spec("C", ("c", "d"), (7, 3)),
            spec("D", ("d", "a"), (3, 6)),
        ]
        front = pareto_front(specs)
        self.assertEqual([(p.mult_count, p.peak_memory) for p in front],
                         [(168, 66), (174, 61)])

        plan_free = find_plan(specs, 10**9)
        self.assertEqual(plan_free.total_mult_count, 168)
        self.assertEqual(plan_free.peak_memory, 66)

        plan_66 = find_plan(specs, 66)
        self.assertEqual(plan_66.total_mult_count, 168)
        self.assertEqual(plan_66.peak_memory, 66)

        plan_65 = find_plan(specs, 65)
        self.assertEqual(plan_65.total_mult_count, 174)
        self.assertEqual(plan_65.peak_memory, 61)
        assert_plan_consistent(self, plan_65, specs, dims)

        with self.assertRaises(MemoryLimitInfeasibleError):
            find_plan(specs, 60)
        # 上限低于最小可达峰值 61 => 经完整穷举证明无解（边界值精确）。
        with self.assertRaises(MemoryLimitInfeasibleError):
            find_plan(specs, 61 - 1)

    def test_budget_exhausted_distinct_from_infeasible(self):
        dims = {"a": 3, "b": 4, "c": 5, "d": 2, "e": 3}
        specs = [
            spec("A", ("a", "b"), (3, 4)),
            spec("B", ("b", "c"), (4, 5)),
            spec("C", ("c", "d"), (5, 2)),
            spec("D", ("d", "e"), (2, 3)),
            spec("E", ("e", "a"), (3, 3)),
        ]
        with self.assertRaises(BudgetExhaustedError):
            find_plan(specs, 10**9, search_budget=1)
        # 预算给足后同一问题成功，证明预算异常不等于无解。
        plan = find_plan(specs, 10**9, search_budget=10**7)
        expect = ref.best_muls_under_limit([t.axes for t in specs], dims, 10**9)
        self.assertEqual(plan.total_mult_count, expect)
        # BudgetExhaustedError 不是 InfeasibleError 的子类。
        self.assertFalse(issubclass(BudgetExhaustedError, InfeasibleError))

    def test_works_with_network_disabled(self):
        # 断开全部 socket：规划器必须纯离线完成。
        def boom(*_a, **_k):
            raise RuntimeError("network access attempted")

        old = socket.socket
        socket.socket = boom
        try:
            specs = [
                spec("A", ("i", "k"), (2, 3)),
                spec("B", ("k", "j"), (3, 4)),
                spec("C", ("j", "l"), (4, 5)),
            ]
            plan = find_plan(specs, 10**9)
            self.assertEqual(plan.total_mult_count,
                             ref.best_muls_under_limit(
                                 [t.axes for t in specs],
                                 {"i": 2, "k": 3, "j": 4, "l": 5}, 10**9))
        finally:
            socket.socket = old

    def test_deterministic(self):
        specs = [
            spec("A", ("a", "b"), (2, 3)),
            spec("B", ("b", "c"), (3, 4)),
            spec("C", ("c", "a"), (4, 2)),
        ]
        p1 = find_plan(specs, 10**9)
        p2 = find_plan(specs, 10**9)
        self.assertEqual(p1.tree, p2.tree)
        self.assertEqual(p1.steps, p2.steps)


class TestInvalidInputs(unittest.TestCase):
    def expect(self, *a, **k):
        return self.assertRaises(InvalidProblemError, find_plan, *a, **k)

    def test_nan_inf_dimensions_rejected(self):
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), (float("nan"),))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), (float("inf"),))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), (-float("inf"),))

    def test_bool_integer_rejected(self):
        # bool 是 int 子类，但契约明确拒绝。
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), (True,))
        specs = [spec("A", ("i", "k"), (2, 3)), spec("B", ("k", "j"), (3, 4))]
        with self.assertRaises(InvalidProblemError):
            find_plan(specs, True)
        with self.assertRaises(InvalidProblemError):
            find_plan(specs, 100, search_budget=False)

    def test_float_int_rejected(self):
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), (2.0,))
        specs = [spec("A", ("i", "k"), (2, 3)), spec("B", ("k", "j"), (3, 4))]
        with self.assertRaises(InvalidProblemError):
            find_plan(specs, 100.0)

    def test_non_positive_and_types(self):
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), (0,))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), (-3,))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), ("4",))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i",), [None])

    def test_bad_labels_and_names(self):
        with self.assertRaises(InvalidProblemError):
            spec("", ("i",), (2,))
        with self.assertRaises(InvalidProblemError):
            spec(3, ("i",), (2,))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("",), (2,))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i", 7), (2, 3))
        with self.assertRaises(InvalidProblemError):
            spec("A", ("i", "i"), (2, 3))
        with self.assertRaises(InvalidProblemError):
            TensorSpec("A", "ij", (2, 3))

    def test_problem_level_errors(self):
        good = [spec("A", ("i", "k"), (2, 3)),
                spec("B", ("k", "j"), (3, 4))]
        with self.assertRaises(InvalidProblemError):
            find_plan([], 100)
        with self.assertRaises(InvalidProblemError):
            find_plan(good, 0)
        with self.assertRaises(InvalidProblemError):
            find_plan(good, -1)
        with self.assertRaises(InvalidProblemError):
            find_plan(good, float("nan"))
        with self.assertRaises(InvalidProblemError):
            find_plan(good, 100, search_budget=0)
        # 重名。
        with self.assertRaises(InvalidProblemError):
            find_plan([spec("A", ("i",), (2,)), spec("A", ("j",), (3,))], 100)
        # 保留名冲突。
        with self.assertRaises(InvalidProblemError):
            find_plan([spec("T0", ("i",), (2,)), spec("X", ("i",), (2,))], 100)
        # 标签出现 3 次。
        with self.assertRaises(InvalidProblemError):
            find_plan([
                spec("A", ("i",), (2,)),
                spec("B", ("i",), (2,)),
                spec("C", ("i",), (2,)),
            ], 100)
        # 同名标签维大小冲突。
        with self.assertRaises(InvalidProblemError):
            find_plan([spec("A", ("i", "k"), (2, 3)),
                       spec("B", ("k", "j"), (4, 5))], 100)
        # 非 TensorSpec 元素 / 非序列。
        with self.assertRaises(InvalidProblemError):
            find_plan([good[0], "B"], 100)
        with self.assertRaises(InvalidProblemError):
            find_plan("AB", 100)

    def test_too_many_tensors(self):
        specs = [spec("T%02d" % i, ("a%d" % i, "b"), (2, 2))
                 for i in range(tp.MAX_TENSORS + 1)]
        with self.assertRaises(InvalidProblemError):
            find_plan(specs, 10**9)

    def test_error_messages_locatable(self):
        try:
            spec("A", ("i",), (float("nan"),))
        except InvalidProblemError as e:
            self.assertIn("A", str(e))
        try:
            specs = [spec("A", ("i", "k"), (2, 3)),
                     spec("B", ("k", "j"), (4, 5))]
            find_plan(specs, 100)
        except InvalidProblemError as e:
            self.assertIn("k", str(e))


class TestMaxBoundary(unittest.TestCase):
    def test_ten_tensor_chain(self):
        # 10 个张量的链：接口上限规模，默认无预算必须完整穷举。
        n = tp.MAX_TENSORS
        labels = ["l%d" % i for i in range(n + 1)]
        sizes = {a: 2 for a in labels}
        specs = []
        for i in range(n):
            specs.append(spec("X%d" % i, (labels[i], labels[i + 1]), (2, 2)))
        plan = find_plan(specs, 10**9)
        expect = ref.best_muls_under_limit([t.axes for t in specs], sizes, 10**9)
        self.assertEqual(plan.total_mult_count, expect)
        assert_plan_consistent(self, plan, specs, sizes)
        # 每步都是 2x2 @ 2x2 => 8 乘，共 9 步。
        self.assertEqual(plan.total_mult_count, 8 * (n - 1))


# ---------------------------------------------------------------------------
# n<=6 随机网络：独立全树枚举与核心逐一交叉验证（确定性生成）
# ---------------------------------------------------------------------------


def gen_network(rng, n):
    """生成连通网络：链状收缩轴保证连通，再随机加共享轴/输出轴。"""
    axes = [[] for _ in range(n)]
    sizes = {}
    nxt = [0]

    def new_label():
        a = "z%d" % nxt[0]
        nxt[0] += 1
        sizes[a] = rng.randint(2, 5)
        return a

    for i in range(1, n):  # 链：保证连通
        a = new_label()
        axes[i - 1].append(a)
        axes[i].append(a)
    for _ in range(rng.randint(0, n)):  # 额外共享轴
        i, j = rng.sample(range(n), 2)
        a = new_label()
        axes[i].append(a)
        axes[j].append(a)
    for _ in range(rng.randint(0, n)):  # 输出轴（出现一次）
        i = rng.randrange(n)
        axes[i].append(new_label())
    rng.shuffle(axes)
    return [tuple(a) for a in axes], sizes


class TestExhaustiveCrossCheck(unittest.TestCase):
    def test_random_networks_n1_to_6(self):
        rng = random.Random(20260923)
        trials = {1: 4, 2: 12, 3: 30, 4: 30, 5: 20, 6: 12}
        for n, count in trials.items():
            for _ in range(count):
                if n == 1:
                    ax = (("o0", "o1"),)
                    sizes = {"o0": rng.randint(2, 5),
                             "o1": rng.randint(2, 5)}
                    axes_list = list(ax)
                else:
                    axes_list, sizes = gen_network(rng, n)
                specs = [
                    spec("S%d" % i, a, tuple(sizes[x] for x in a))
                    for i, a in enumerate(axes_list)
                ]
                points = ref.all_optimal_points(
                    [t.axes for t in specs], sizes
                )
                self.assertTrue(points, "生成器应只产出连通网络")
                # Pareto 前沿完全一致。
                front = pareto_front(specs)
                self.assertEqual(
                    [(p.mult_count, p.peak_memory) for p in front],
                    points,
                )
                if n == 1:
                    # 单张量无收缩：零乘法零峰值；limit 仍须为正整数。
                    trivial = find_plan(specs, 1)
                    self.assertEqual(trivial.total_mult_count, 0)
                    self.assertEqual(trivial.peak_memory, 0)
                    self.assertEqual(trivial.steps, ())
                    continue
                # 前沿上每个点作为紧上限时，核心必须选出该乘法数与峰值。
                for m, p in points:
                    plan = find_plan(specs, p)
                    self.assertEqual(plan.total_mult_count, m)
                    self.assertEqual(plan.peak_memory, p)
                    self.assertLessEqual(plan.peak_memory, p)
                    assert_plan_consistent(self, plan, specs, sizes)
                # 低于最小峰值 => 经穷举证明无解（与预算异常分开）。
                min_peak = min(p for _, p in points)
                with self.assertRaises(MemoryLimitInfeasibleError):
                    find_plan(specs, min_peak - 1)
                # 宽松上限 => 全局最少乘法。
                plan = find_plan(specs, 10**18)
                self.assertEqual(plan.total_mult_count, points[0][0])


if __name__ == "__main__":
    unittest.main()

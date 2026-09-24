"""plan_slices（单维切块逐切片规划）的必要测试。

期望值来源：
- 逐切片的最少乘法数/峰值来自 tests/reference_enum.py（独立全二叉树枚举，
  不导入被测核心），切块后的子问题直接喂给独立参考；
- 总乘法数/峰值中间存储由参考逐片结果 + 手算解析关系（累加成本）组合；
- 数值正确性来自独立稠密逐元素执行器 + 独立直算枚举求和（见本文件末尾）。
被测核心只通过公开接口调用。
"""

import itertools
import random
import unittest

from tests import reference_enum as ref

import tensor_peak as tp
from tensor_peak import (
    TensorSpec,
    SliceDim,
    find_plan,
    plan_slices,
    InvalidProblemError,
    MemoryLimitInfeasibleError,
    BudgetExhaustedError,
)


def spec(name, axes, shape):
    return TensorSpec(name, tuple(axes), tuple(shape))


def slice_subproblem(specs, axis, chunk):
    """构造轴 axis 维长改为 chunk 的子问题（独立于被测核心）。"""
    axes_list, sizes = [], {}
    for t in specs:
        ax = tuple(t.axes)
        axes_list.append(ax)
        for a, d in zip(t.axes, t.shape):
            sizes[a] = chunk if a == axis else d
    return axes_list, sizes


def expect_slice(specs, axis, chunk, limit):
    """独立参考给出该切片在 limit 下的 (最少乘法, 对应峰值)。"""
    axes_list, sizes = slice_subproblem(specs, axis, chunk)
    points = ref.all_optimal_points(axes_list, sizes)
    feasible = [(m, p) for m, p in points if p <= limit]
    return min(feasible)  # 核心按 (乘法, 峰值) 字典序取最优点


def output_size_of(specs):
    counts, sizes = {}, {}
    for t in specs:
        for a, d in zip(t.axes, t.shape):
            counts[a] = counts.get(a, 0) + 1
            sizes[a] = d
    out_axes = [a for a, c in counts.items() if c == 1]
    return math_prod(sizes[a] for a in out_axes)


def math_prod(xs):
    r = 1
    for x in xs:
        r *= x
    return r


class TestSlicedContractAxis(unittest.TestCase):
    def setUp(self):
        # A(i,k) B(k,j)：i=2 k=6 j=3，收缩轴 k 切成 (2,4)。
        self.specs = [
            spec("A", ("i", "k"), (2, 6)),
            spec("B", ("k", "j"), (6, 3)),
        ]
        self.limit = 10**9

    def test_per_slice_and_totals_hand_computed(self):
        sp = plan_slices(self.specs, self.limit, SliceDim("k", (2, 4)))
        self.assertEqual(len(sp.slices), 2)
        # 手算：片0 k=2：2*2*3=12 乘，峰值 2*2+2*3+2*3=16；
        #       片1 k=4：2*4*3=24 乘，峰值 2*4+2*3+2*3=26。
        self.assertEqual(sp.slices[0].total_mult_count, 12)
        self.assertEqual(sp.slices[0].peak_memory, 16)
        self.assertEqual(sp.slices[1].total_mult_count, 24)
        self.assertEqual(sp.slices[1].peak_memory, 26)
        # 输出 2*3=6 个元素，2 片 => 累加 6 次标量加法。
        self.assertEqual(sp.accumulator_cost, 6)
        self.assertEqual(sp.total_mult_count, 12 + 24 + 6)
        self.assertEqual(sp.peak_intermediate_storage, 26)
        self.assertEqual(sp.peak_slice, 1)
        self.assertEqual(sp.slice_axis, "k")
        self.assertEqual(sp.slice_sizes, (2, 4))

    def test_totals_match_independent_reference(self):
        chunks = (2, 4)
        sp = plan_slices(self.specs, self.limit, SliceDim("k", chunks))
        muls = peak = 0
        for s, c in enumerate(chunks):
            em, ep = expect_slice(self.specs, "k", c, self.limit)
            self.assertEqual(sp.slices[s].total_mult_count, em)
            self.assertEqual(sp.slices[s].peak_memory, ep)
            muls += em
            peak = max(peak, ep)
        self.assertEqual(sp.accumulator_cost,
                         (len(chunks) - 1) * output_size_of(self.specs))
        self.assertEqual(sp.total_mult_count,
                         muls + (len(chunks) - 1) * output_size_of(self.specs))
        self.assertEqual(sp.peak_intermediate_storage, peak)

    def test_unit_chunks(self):
        # k 全切成单位长：6 片，每片 6 乘 / 峰值 2+3+6=11；累加 5*6=30。
        chunks = (1,) * 6
        sp = plan_slices(self.specs, self.limit, SliceDim("k", chunks))
        self.assertEqual([p.total_mult_count for p in sp.slices], [6] * 6)
        self.assertEqual([p.peak_memory for p in sp.slices], [11] * 6)
        self.assertEqual(sp.accumulator_cost, 30)
        self.assertEqual(sp.total_mult_count, 36 + 30)
        self.assertEqual(sp.peak_intermediate_storage, 11)
        self.assertEqual(sp.peak_slice, 0)
        # 逐片仍收缩 k，输出形状恒为完整 (2,3)。
        for p in sp.slices:
            self.assertEqual(p.output_axes, ("i", "j"))
            self.assertEqual(p.output_shape, (2, 3))

    def test_single_chunk_is_unsliced_plan_plus_zero_accum(self):
        # 整块一片：无累加，乘法/峰值等于普通 find_plan。
        whole = find_plan(self.specs, self.limit)
        sp = plan_slices(self.specs, self.limit, SliceDim("k", (6,)))
        self.assertEqual(sp.accumulator_cost, 0)
        self.assertEqual(sp.total_mult_count, whole.total_mult_count)
        self.assertEqual(sp.peak_intermediate_storage, whole.peak_memory)
        self.assertEqual(sp.slices[0].tree, whole.tree)


class TestSlicedOutputAxis(unittest.TestCase):
    def test_output_split_no_accumulation(self):
        # A(i,k) B(k,j)：i=2 k=6 j=3，沿输出轴 i 切成 (1,1)。
        specs = [
            spec("A", ("i", "k"), (2, 6)),
            spec("B", ("k", "j"), (6, 3)),
        ]
        sp = plan_slices(specs, 10**9, SliceDim("i", (1, 1)))
        self.assertEqual(sp.accumulator_cost, 0)
        # 各片 1*6*3=18 乘，峰值 1*6 + 6*3 + 1*3 = 6+18+3 = 27。
        self.assertEqual([p.total_mult_count for p in sp.slices], [18, 18])
        self.assertEqual(sp.total_mult_count, 36)
        self.assertEqual(sp.peak_intermediate_storage, 27)
        # 不相交分块：各产出 i 维长度 1，拼回完整 (2,3)。
        self.assertEqual([p.output_shape for p in sp.slices], [(1, 3), (1, 3)])
        # 独立参考核对。
        for s, c in enumerate((1, 1)):
            em, ep = expect_slice(specs, "i", c, 10**9)
            self.assertEqual(sp.slices[s].total_mult_count, em)
            self.assertEqual(sp.slices[s].peak_memory, ep)

    def test_single_tensor_output_slicing(self):
        one = [spec("S", ("a", "b"), (3, 4))]
        sp = plan_slices(one, 1, SliceDim("a", (1, 2)))
        self.assertEqual(len(sp.slices), 2)
        self.assertEqual(sp.total_mult_count, 0)
        self.assertEqual(sp.accumulator_cost, 0)
        self.assertEqual(sp.peak_intermediate_storage, 0)
        self.assertEqual([p.output_shape for p in sp.slices], [(1, 4), (2, 4)])
        self.assertEqual([p.steps for p in sp.slices], [(), ()])

    def test_ring4_scalar_output_axis_elsewhere(self):
        # 给四环加一个输出轴 o 再沿它切：累加为 0，逐片与参考一致。
        dims = {"a": 2, "b": 3, "c": 2, "d": 3, "o": 4}
        specs = [
            spec("A", ("o", "a", "b"), (4, 2, 3)),
            spec("B", ("b", "c"), (3, 2)),
            spec("C", ("c", "d"), (2, 3)),
            spec("D", ("d", "a"), (3, 2)),
        ]
        sp = plan_slices(specs, 10**9, SliceDim("o", (1, 3)))
        self.assertEqual(sp.accumulator_cost, 0)
        for s, c in enumerate((1, 3)):
            em, ep = expect_slice(specs, "o", c, 10**9)
            self.assertEqual(sp.slices[s].total_mult_count, em)
            self.assertEqual(sp.slices[s].peak_memory, ep)
        # 输出轴分块长度出现在输出形状首位（o 最先出现）。
        self.assertEqual(sp.slices[0].output_shape[0], 1)
        self.assertEqual(sp.slices[1].output_shape[0], 3)


class TestSlicingTagsAndBackwardCompat(unittest.TestCase):
    def test_steps_and_plans_tagged(self):
        specs = [spec("A", ("i", "k"), (2, 6)), spec("B", ("k", "j"), (6, 3))]
        sp = plan_slices(specs, 10**9, SliceDim("k", (2, 4)))
        for s, p in enumerate(sp.slices):
            self.assertEqual(p.slicing, SliceDim("k", (2, 4)))
            for step in p.steps:
                self.assertEqual(step.sliced_axis, "k")
                self.assertEqual(step.slice_index, s)

    def test_plain_plan_slicing_is_none(self):
        # 旧入口行为不变：普通计划不带任何切块标注。
        specs = [spec("A", ("i", "k"), (2, 6)), spec("B", ("k", "j"), (6, 3))]
        plan = find_plan(specs, 10**9)
        self.assertIsNone(plan.slicing)
        for step in plan.steps:
            self.assertIsNone(step.sliced_axis)
            self.assertIsNone(step.slice_index)

    def test_deterministic(self):
        specs = [spec("A", ("i", "k"), (2, 6)), spec("B", ("k", "j"), (6, 3))]
        sd = SliceDim("k", (2, 4))
        a = plan_slices(specs, 10**9, sd)
        b = plan_slices(specs, 10**9, sd)
        self.assertEqual(
            [p.steps for p in a.slices],
            [p.steps for p in b.slices],
        )
        self.assertEqual(a.total_mult_count, b.total_mult_count)


class TestSlicingFailures(unittest.TestCase):
    def setUp(self):
        self.specs = [
            spec("A", ("i", "k"), (2, 6)),
            spec("B", ("k", "j"), (6, 3)),
        ]

    def test_axis_not_present(self):
        with self.assertRaises(InvalidProblemError):
            plan_slices(self.specs, 10**9, SliceDim("z", (2, 4)))

    def test_chunk_sum_mismatch(self):
        with self.assertRaises(InvalidProblemError):
            plan_slices(self.specs, 10**9, SliceDim("k", (2, 3)))  # 和 5 != 6
        with self.assertRaises(InvalidProblemError):
            plan_slices(self.specs, 10**9, SliceDim("k", (2, 5)))  # 和 7 != 6

    def test_bad_slice_dim_object(self):
        with self.assertRaises(InvalidProblemError):
            SliceDim("", (2, 4))
        with self.assertRaises(InvalidProblemError):
            SliceDim(7, (2, 4))
        with self.assertRaises(InvalidProblemError):
            SliceDim("k", ())
        with self.assertRaises(InvalidProblemError):
            SliceDim("k", (0, 6))
        with self.assertRaises(InvalidProblemError):
            SliceDim("k", (-1, 7))
        with self.assertRaises(InvalidProblemError):
            SliceDim("k", (2, True))
        with self.assertRaises(InvalidProblemError):
            SliceDim("k", (2.0, 4.0))
        with self.assertRaises(InvalidProblemError):
            SliceDim("k", "24")
        with self.assertRaises(InvalidProblemError):
            plan_slices(self.specs, 10**9, ("k", (2, 4)))

    def test_underlying_problem_still_validated(self):
        # 切块不豁免原问题契约：不连通网络仍被拒绝。
        disc = [
            spec("A", ("i", "k"), (2, 3)),
            spec("B", ("k", "j"), (3, 4)),
            spec("D", ("p", "q"), (2, 3)),
            spec("E", ("q", "r"), (3, 4)),
        ]
        with self.assertRaises(tp.DisconnectedNetworkError):
            plan_slices(disc, 10**9, SliceDim("k", (1, 2)))
        # 非正上限仍拒绝。
        with self.assertRaises(InvalidProblemError):
            plan_slices(self.specs, 0, SliceDim("k", (2, 4)))

    def test_slice_infeasible_under_limit_propagates(self):
        # k 片长 4 时最小峰值 26；上限 25 使该片经穷举证明无解。
        with self.assertRaises(MemoryLimitInfeasibleError):
            plan_slices(self.specs, 25, SliceDim("k", (2, 4)))
        # 紧边界：上限 26 两片都可行（片0 峰值 16，片1 峰值 26）。
        sp = plan_slices(self.specs, 26, SliceDim("k", (2, 4)))
        self.assertEqual(sp.peak_intermediate_storage, 26)

    def test_budget_exhausted_propagates_without_partial_result(self):
        specs = [
            spec("A", ("a", "b"), (3, 4)),
            spec("B", ("b", "c"), (4, 5)),
            spec("C", ("c", "a"), (5, 3)),
        ]
        before = [tuple(t.shape) for t in specs]
        with self.assertRaises(BudgetExhaustedError):
            plan_slices(specs, 10**9, SliceDim("a", (1, 2)),
                        search_budget=1)
        # 失败不修改调用者输入（无半更新状态）。
        self.assertEqual([tuple(t.shape) for t in specs], before)


# ---------------------------------------------------------------------------
# 小规模随机网络：逐切片与独立全树枚举交叉验证（确定性种子）
# ---------------------------------------------------------------------------


def gen_network(rng, n):
    axes = [[] for _ in range(n)]
    sizes = {}
    nxt = [0]

    def new_label():
        a = "z%d" % nxt[0]
        nxt[0] += 1
        sizes[a] = rng.randint(2, 5)
        return a

    for i in range(1, n):
        a = new_label()
        axes[i - 1].append(a)
        axes[i].append(a)
    for _ in range(rng.randint(0, n)):
        i, j = rng.sample(range(n), 2)
        a = new_label()
        axes[i].append(a)
        axes[j].append(a)
    out_holders = []
    for _ in range(rng.randint(1, n)):  # 至少一个输出轴以便切输出轴
        i = rng.randrange(n)
        a = new_label()
        axes[i].append(a)
        out_holders.append(a)
    rng.shuffle(axes)
    return [tuple(a) for a in axes], sizes, out_holders


class TestRandomSlicedCrossCheck(unittest.TestCase):
    def test_random_networks(self):
        rng = random.Random(20260924)
        for n, count in {2: 8, 3: 10, 4: 6}.items():
            for _ in range(count):
                axes_list, sizes, out_axes = gen_network(rng, n)
                specs = [
                    spec("S%d" % i, a, tuple(sizes[x] for x in a))
                    for i, a in enumerate(axes_list)
                ]
                # 随机选一个轴，随机切成两块。
                axis = rng.choice(list(sizes))
                d = sizes[axis]
                c0 = rng.randint(1, d)
                chunks = (c0, d - c0) if d - c0 > 0 else (d,)
                sp = plan_slices(specs, 10**18, SliceDim(axis, chunks))
                em_total = epeak = 0
                for s, c in enumerate(chunks):
                    em, ep = expect_slice(specs, axis, c, 10**18)
                    self.assertEqual(sp.slices[s].total_mult_count, em)
                    self.assertEqual(sp.slices[s].peak_memory, ep)
                    em_total += em
                    epeak = max(epeak, ep)
                is_output = axis in out_axes
                accum = 0 if is_output else (len(chunks) - 1) * output_size_of(
                    specs
                )
                self.assertEqual(sp.accumulator_cost, accum)
                self.assertEqual(sp.total_mult_count, em_total + accum)
                self.assertEqual(sp.peak_intermediate_storage, epeak)


# ---------------------------------------------------------------------------
# 数值核对：独立稠密执行器逐切片执行并跨片合并，对独立直算结果逐元素 ==
# ---------------------------------------------------------------------------


def cell_value(name, idx):
    """与具体切片无关的确定性整数值（只依赖名字与全局索引）。"""
    base = sum(ord(c) for c in name) % 7 + 1
    return base + sum((p + 1) * (v + 1) for p, v in enumerate(idx))


def full_data(t):
    return {idx: cell_value(t.name, idx)
            for idx in itertools.product(*(range(d) for d in t.shape))}


def execute_on_data(tensors, steps, output_axes, data):
    """用给定输入数据按步骤收缩，返回 {输出索引: 值}（不调用核心算法）。"""
    by_name = {t.name: t for t in tensors}
    live = {}

    def view(name):
        if name in by_name:
            ax = by_name[name].axes
            return ax, data[name]
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


def direct_full(specs, out_axes):
    data = {t.name: full_data(t) for t in specs}
    sizes = {}
    for t in specs:
        sizes.update(dict(zip(t.axes, t.shape)))
    contract = [a for a in sizes
                if sum(a in t.axes for t in specs) == 2]
    res = {}
    for combo in itertools.product(
        *(range(sizes[a]) for a in list(out_axes) + contract)
    ):
        pos = dict(zip(list(out_axes) + contract, combo))
        v = 1
        for t in specs:
            v *= data[t.name][tuple(pos[a] for a in t.axes)]
        key = tuple(pos[a] for a in out_axes)
        res[key] = res.get(key, 0) + v
    return res


class TestSlicedNumeric(unittest.TestCase):
    def sliced_data(self, specs, axis, chunk, offset):
        """从完整确定性数据中取该片输入：沿 axis 用局部索引 0..chunk-1，
        映射到全局索引 offset+local（独立于被测核心）。"""
        data = {}
        for t in specs:
            shape = tuple(
                chunk if a == axis else d for a, d in zip(t.axes, t.shape)
            )
            p = t.axes.index(axis) if axis in t.axes else None
            d = {}
            for idx in itertools.product(*(range(x) for x in shape)):
                if p is None:
                    global_idx = idx
                else:
                    global_idx = tuple(
                        offset + v if q == p else v for q, v in enumerate(idx)
                    )
                d[idx] = cell_value(t.name, global_idx)
            data[t.name] = d
        return data

    def test_contract_axis_numeric(self):
        specs = [
            spec("A", ("i", "k"), (2, 6)),
            spec("B", ("k", "j"), (6, 3)),
        ]
        chunks = (2, 4)
        sp = plan_slices(specs, 10**9, SliceDim("k", chunks))
        want = direct_full(specs, ("i", "j"))
        got = {}
        offset = 0
        for s, c in enumerate(chunks):
            data = self.sliced_data(specs, "k", c, offset)
            offset += c
            part = execute_on_data(
                sp.slices[s].tensors, sp.slices[s].steps,
                sp.slices[s].output_axes, data,
            )
            for key, v in part.items():
                got[key] = got.get(key, 0) + v  # 跨片累加
        self.assertEqual(got, want)

    def test_output_axis_numeric(self):
        specs = [
            spec("A", ("i", "k"), (2, 6)),
            spec("B", ("k", "j"), (6, 3)),
        ]
        chunks = (1, 1)
        sp = plan_slices(specs, 10**9, SliceDim("i", chunks))
        want = direct_full(specs, ("i", "j"))
        got = {}
        offset = 0
        for s, c in enumerate(chunks):
            data = self.sliced_data(specs, "i", c, offset)
            part = execute_on_data(
                sp.slices[s].tensors, sp.slices[s].steps,
                sp.slices[s].output_axes, data,
            )
            # i 是输出轴首轴：局部索引加 offset 还原全局分块，直接拼接。
            for (li, j), v in part.items():
                got[(offset + li, j)] = v
            offset += c
        self.assertEqual(got, want)
        self.assertEqual(set(got), set(want))


if __name__ == "__main__":
    unittest.main()

"""find_sliced_plan（可切片收缩轴）的必要测试。

期望值来源：
1. 手算小例子（字面量，注释给出算式）；
2. tests/reference_enum.py 的独立全二叉树枚举（不导入被测核心）逐片交叉
   核对乘法数与峰值；
3. 本文件内独立的稠密逐元素执行器：逐片收缩并累加，与独立直算枚举
   求和结果精确 == 比对（期望值不来自被测核心）。
"""

import itertools
import math
import random
import unittest

from tests import reference_enum as ref

from tensor_peak import (
    MAX_SLICES,
    TensorSpec,
    find_plan,
    find_sliced_plan,
    InvalidProblemError,
    MemoryLimitInfeasibleError,
    BudgetExhaustedError,
)


def spec(name, axes, shape):
    return TensorSpec(name, tuple(axes), tuple(shape))


# ---------------------------------------------------------------------------
# 独立数值执行器：按逐切片计划真实收缩并累加（不调用核心算法）
# ---------------------------------------------------------------------------


def make_values(spec_t):
    data = {}
    base = sum(ord(c) for c in spec_t.name) % 7 + 1
    for idx in itertools.product(*(range(d) for d in spec_t.shape)):
        data[idx] = base + sum((p + 1) * (v + 1) for p, v in enumerate(idx))
    return data


def execute_sliced(sp, originals):
    """执行 SlicedPlan：逐片收缩（被切轴坐标取该片区间），输出累加。

    originals: 原始 TensorSpec 列表（数据按原始轴序生成）。
    返回 {输出索引元组: 累加值}。
    """
    orig_data = {t.name: (t.axes, make_values(t)) for t in originals}
    total = {}

    for k, plan in enumerate(sp.slices):
        lo = k * sp.slice_size
        # 为该片的（轴被重排、维被切窄的）输入构造数据。
        inputs = {}
        for st in plan.tensors:
            o_axes, o_dat = orig_data[st.name]
            pos_of_new = {a: q for q, a in enumerate(st.axes)}
            d = {}
            for idx in itertools.product(*(range(x) for x in st.shape)):
                o_idx = tuple(
                    (lo + idx[pos_of_new[a]]) if a == sp.sliced_axis
                    else idx[pos_of_new[a]]
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
            union_axes = list(ax_l) + [
                a for a in ax_r if a not in set(ax_l)
            ]
            out = {}
            for combo in itertools.product(
                *(range(sizes[a]) for a in union_axes)
            ):
                pos = dict(zip(union_axes, combo))
                kl = tuple(pos[a] for a in ax_l)
                kr = tuple(pos[a] for a in ax_r)
                ko = tuple(pos[a] for a in step.output_axes)
                out[ko] = out.get(ko, 0) + dat_l[kl] * dat_r[kr]
            live[step.output_name] = (step.output_axes, out)
            for op in (step.lhs, step.rhs):
                if op.kind == "intermediate":
                    live.pop(op.name, None)

        partial = live[plan.steps[-1].output_name][1]
        # 首片初始化累加器，之后逐输出元素做一次加法。
        for key, val in partial.items():
            total[key] = total.get(key, 0) + val

    return total


def direct_reference(specs, output_axes):
    """独立直算：枚举输出轴+全部收缩轴，对原始输入乘积求和。"""
    sizes = {}
    for t in specs:
        for a, d in zip(t.axes, t.shape):
            sizes[a] = d
    data = {t.name: make_values(t) for t in specs}
    result = {}
    enum_axes = list(output_axes) + [
        a for a in sizes
        if sum(a in t.axes for t in specs) == 2
    ]
    for combo in itertools.product(*(range(sizes[a]) for a in enum_axes)):
        pos = dict(zip(enum_axes, combo))
        prod = 1
        for t in specs:
            prod *= data[t.name][tuple(pos[a] for a in t.axes)]
        key = tuple(pos[a] for a in output_axes)
        result[key] = result.get(key, 0) + prod
    return result


# ---------------------------------------------------------------------------
# 结构不变量
# ---------------------------------------------------------------------------


def assert_sliced_consistent(test, sp, originals):
    test.assertEqual(len(sp.slices), sp.num_slices)
    axis_dim = next(
        d for t in originals for a, d in zip(t.axes, t.shape)
        if a == sp.sliced_axis
    )
    widths = [
        min(sp.slice_size, axis_dim - k * sp.slice_size)
        for k in range(sp.num_slices)
    ]
    test.assertTrue(all(w >= 1 for w in widths))
    test.assertEqual(sum(widths), axis_dim)

    mult_sum = 0
    output_size = math.prod(sp.output_shape)
    test.assertEqual(sp.output_size, output_size)
    for k, plan in enumerate(sp.slices):
        # 片计划：合法 ContractionPlan，输出轴恒为最终输出轴（被切轴片内
        # 即收缩），被切轴维宽与片区间一致。
        test.assertEqual(plan.output_axes, sp.output_axes)
        test.assertLessEqual(plan.peak_memory, plan.memory_limit)
        owners = [
            st for st in plan.tensors if sp.sliced_axis in st.axes
        ]
        test.assertEqual(len(owners), 2)
        for st in owners:
            test.assertEqual(st.axes[-1], sp.sliced_axis)
            test.assertEqual(st.shape[-1], widths[k])
        held = 0 if k == 0 else output_size
        test.assertEqual(plan.memory_limit, sp.memory_limit - held)
        mult_sum += plan.total_mult_count
    test.assertEqual(sp.total_mult_count, mult_sum)
    test.assertEqual(sp.total_add_count,
                     (sp.num_slices - 1) * output_size)
    # 峰值口径：max(首片峰值, 累加器 + 其余各片峰值)。
    expect_peak = max(
        sp.slices[0].peak_memory,
        max(
            (output_size + p.peak_memory for p in sp.slices[1:]),
            default=0,
        ),
    )
    test.assertEqual(sp.peak_memory, expect_peak)
    test.assertLessEqual(sp.peak_memory, sp.memory_limit)


# ---------------------------------------------------------------------------
# 正常用例（手算核对）
# ---------------------------------------------------------------------------


class TestSlicedHandChecked(unittest.TestCase):
    def test_ring3_scalar_output_two_equal_slices(self):
        # A(a,b) B(b,c) C(c,a)：a=4,b=3,c=2，沿 a 切 2 片（宽各 2）。
        specs = [
            spec("A", ("a", "b"), (4, 3)),
            spec("B", ("b", "c"), (3, 2)),
            spec("C", ("c", "a"), (2, 4)),
        ]
        sp = find_sliced_plan(specs, 10**9, "a", 2)
        self.assertEqual(sp.num_slices, 2)
        # 每片：(A x B) 2*3*2=12，再 x C 2*2=4 => 16；两片共 32。
        self.assertEqual([p.total_mult_count for p in sp.slices], [16, 16])
        self.assertEqual(sp.total_mult_count, 32)
        # 标量输出 1 个元素：1 片初始化 + 1 次标量加法。
        self.assertEqual(sp.total_add_count, 1)
        self.assertEqual(sp.output_axes, ())
        self.assertEqual(sp.output_size, 1)
        # 片峰值：|AB|=4 + |C_slice|=4 + 1 = 9 是收缩末步；首片整体
        # 峰值由规划器给出，第二片 +1 累加器。
        self.assertEqual(
            sp.peak_memory,
            max(sp.slices[0].peak_memory,
                1 + sp.slices[1].peak_memory),
        )
        assert_sliced_consistent(self, sp, specs)

        # 数值：逐片执行 + 累加 == 独立直算。
        got = execute_sliced(sp, specs)
        want = direct_reference(specs, ())
        self.assertEqual(got, want)
        self.assertEqual(got[()], want[()])

    def test_two_tensor_dot_chain_slice_count_axis(self):
        # A(i,k) B(k,j)：i=2,k=6,j=3，沿 k 切 3 片（宽各 2）。
        specs = [
            spec("A", ("i", "k"), (2, 6)),
            spec("B", ("k", "j"), (6, 3)),
        ]
        sp = find_sliced_plan(specs, 10**9, "k", 2)
        self.assertEqual(sp.num_slices, 3)
        # 每片一次收缩：i * w * j = 2*2*3 = 12；三片 36；
        # 不分片时同样是 2*6*3 = 36（乘法数与切块无关）。
        self.assertEqual([p.total_mult_count for p in sp.slices],
                         [12, 12, 12])
        self.assertEqual(sp.total_mult_count, 36)
        self.assertEqual(
            find_plan(specs, 10**9).total_mult_count, 36
        )
        # 输出 (i,j) 共 6 元素，两片做加法 => 12 次加法。
        self.assertEqual(sp.total_add_count, 12)
        self.assertEqual(sp.output_axes, ("i", "j"))
        self.assertEqual(sp.output_shape, (2, 3))
        assert_sliced_consistent(self, sp, specs)
        got = execute_sliced(sp, specs)
        want = direct_reference(specs, ("i", "j"))
        self.assertEqual(got, want)

    def test_uneven_last_slice(self):
        # a 维 5，每片 2 => 宽 2,2,1；乘法数随宽度变化。
        specs = [
            spec("A", ("a", "b"), (5, 3)),
            spec("B", ("b", "c"), (3, 2)),
            spec("C", ("c", "a"), (2, 5)),
        ]
        sp = find_sliced_plan(specs, 10**9, "a", 2)
        self.assertEqual(sp.num_slices, 3)
        widths = []
        for p in sp.slices:
            st = next(x for x in p.tensors if "a" in x.axes)
            widths.append(st.shape[-1])
        self.assertEqual(widths, [2, 2, 1])
        # 宽 2 的片：12+4=16；宽 1 的片：(1*3*2=6)+(1*2=2)=8。
        self.assertEqual([p.total_mult_count for p in sp.slices],
                         [16, 16, 8])
        self.assertEqual(sp.total_mult_count, 40)
        self.assertEqual(sp.total_add_count, 2)
        got = execute_sliced(sp, specs)
        self.assertEqual(got, direct_reference(specs, ()))


class TestSlicedBoundaries(unittest.TestCase):
    def test_single_slice_equivalent_to_unsliced(self):
        # slice_size == 轴维：恰好 1 片、零加法；除轴位置重排外问题等价，
        # 乘法数与 find_plan 相同，峰值即整片计划峰值（无累加器）。
        specs = [
            TensorSpec("A", ("a", "b"), (4, 3)),
            TensorSpec("B", ("b", "c"), (3, 2)),
            TensorSpec("C", ("c", "a"), (2, 4)),
        ]
        sp = find_sliced_plan(specs, 10**9, "a", 4)
        self.assertEqual(sp.num_slices, 1)
        self.assertEqual(sp.total_add_count, 0)
        full = find_plan(specs, 10**9)
        self.assertEqual(sp.total_mult_count, full.total_mult_count)
        self.assertEqual(sp.peak_memory,
                         sp.slices[0].peak_memory)
        got = execute_sliced(sp, specs)
        self.assertEqual(got, direct_reference(specs, ()))

    def test_slice_size_one(self):
        specs = [
            spec("A", ("a", "b"), (3, 2)),
            spec("B", ("b", "a"), (2, 3)),
        ]
        sp = find_sliced_plan(specs, 10**9, "a", 1)
        self.assertEqual(sp.num_slices, 3)
        widths = [
            next(st.shape[-1] for st in p.tensors if "a" in st.axes)
            for p in sp.slices
        ]
        self.assertEqual(widths, [1, 1, 1])
        # A、B 共享 a,b 全部收缩为标量；单片(w=1)并集 a,b => 1*2=2 乘，
        # 三片共 6；不分片时 a*b = 3*2 = 6（乘法数与切块无关）。
        self.assertEqual(sp.total_mult_count, 6)
        self.assertEqual(
            find_plan(specs, 10**9).total_mult_count, 3 * 2
        )
        got = execute_sliced(sp, specs)
        self.assertEqual(got, direct_reference(specs, ()))

    def test_slice_size_larger_than_dim(self):
        # slice_size 大于轴维：退化为 1 片（宽度即轴维）。
        specs = [
            spec("A", ("a", "b"), (3, 2)),
            spec("B", ("b", "a"), (2, 3)),
        ]
        sp = find_sliced_plan(specs, 10**9, "a", 100)
        self.assertEqual(sp.num_slices, 1)
        self.assertEqual(sp.total_add_count, 0)
        self.assertEqual(sp.total_mult_count,
                         find_plan(specs, 10**9).total_mult_count)

    def test_vector_output_accumulator_peak(self):
        # 非标量输出：累加器在第二片起全程驻留，手工核对峰值与加法。
        # A(i,a) B(a,j)：i=2,a=4,j=3；沿 a 切 2 片。输出 (i,j) 共 6。
        specs = [
            spec("A", ("i", "a"), (2, 4)),
            spec("B", ("a", "j"), (4, 3)),
        ]
        sp = find_sliced_plan(specs, 10**9, "a", 2)
        self.assertEqual(sp.output_size, 6)
        self.assertEqual(sp.total_add_count, 6)
        # 每片只有一步：|A_w|=i*w=4，|B_w|=w*j=6，输出=6 => 峰 16。
        self.assertEqual(sp.slices[0].peak_memory, 4 + 6 + 6)
        # 第二片：累加器 6 全程驻留 => 6 + 16 = 22。
        self.assertEqual(sp.peak_memory, 22)
        got = execute_sliced(sp, specs)
        self.assertEqual(got, direct_reference(specs, ("i", "j")))

    def test_deterministic(self):
        specs = [
            spec("A", ("a", "b"), (4, 3)),
            spec("B", ("b", "c"), (3, 2)),
            spec("C", ("c", "a"), (2, 4)),
        ]
        p1 = find_sliced_plan(specs, 10**9, "a", 2)
        p2 = find_sliced_plan(specs, 10**9, "a", 2)
        self.assertEqual(p1.slices, p2.slices)
        self.assertEqual(p1.peak_memory, p2.peak_memory)


# ---------------------------------------------------------------------------
# 失败用例
# ---------------------------------------------------------------------------


class TestSlicedInvalid(unittest.TestCase):
    def setUp(self):
        self.specs = [
            spec("A", ("a", "b"), (4, 3)),
            spec("B", ("b", "c"), (3, 2)),
            spec("C", ("c", "a"), (2, 4)),
        ]

    def test_axis_must_exist(self):
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(self.specs, 10**9, "zz", 2)

    def test_output_axis_not_sliceable(self):
        # 输出轴（只出现一次）不能切。
        specs = [
            spec("A", ("i", "k"), (2, 3)),
            spec("B", ("k", "j"), (3, 4)),
        ]
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(specs, 10**9, "i", 1)
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(specs, 10**9, "j", 1)

    def test_axis_type(self):
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(self.specs, 10**9, "", 2)
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(self.specs, 10**9, 3, 2)

    def test_slice_size_values(self):
        for bad in (0, -1, 2.0, True, False, float("nan"),
                    float("inf"), "2", None):
            with self.assertRaises(InvalidProblemError):
                find_sliced_plan(self.specs, 10**9, "a", bad)

    def test_too_many_slices(self):
        # 维 1000、每片 1 => 1000 片 > MAX_SLICES(64)。
        specs = [
            spec("A", ("a", "b"), (1000, 2)),
            spec("B", ("b", "a"), (2, 1000)),
        ]
        with self.assertRaises(InvalidProblemError) as cm:
            find_sliced_plan(specs, 10**9, "a", 1)
        self.assertIn(str(MAX_SLICES), str(cm.exception))

    def test_boundary_slice_count_at_limit(self):
        # 恰好 MAX_SLICES 片：合法；再多 1 片：拒绝。
        specs = [
            spec("A", ("a", "b"), (MAX_SLICES, 2)),
            spec("B", ("b", "a"), (2, MAX_SLICES)),
        ]
        sp = find_sliced_plan(specs, 10**9, "a", 1)
        self.assertEqual(sp.num_slices, MAX_SLICES)
        specs2 = [
            spec("A", ("a", "b"), (MAX_SLICES + 1, 2)),
            spec("B", ("b", "a"), (2, MAX_SLICES + 1)),
        ]
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(specs2, 10**9, "a", 1)

    def test_single_tensor_no_contraction_axis(self):
        t = spec("S", ("a", "b"), (3, 4))
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan([t], 10**9, "a", 1)

    def test_invalid_base_problem_still_rejected(self):
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan([], 10**9, "a", 1)
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(self.specs, 0, "a", 2)
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(self.specs, True, "a", 2)

    def test_disconnected_network_rejected(self):
        # 切片不改变连通性：断开的网络在切片 API 下同样被证明无解，
        # 且异常具体类型 DisconnectedNetworkError 在重抛后保留。
        from tensor_peak import DisconnectedNetworkError
        specs = [
            spec("A", ("i", "k"), (2, 3)),
            spec("B", ("k", "j"), (3, 4)),
            spec("D", ("p", "q"), (2, 3)),
            spec("E", ("q", "r"), (3, 4)),
        ]
        with self.assertRaises(DisconnectedNetworkError):
            find_sliced_plan(specs, 10**9, "k", 1)

    def test_tight_memory_limit_proved_infeasible(self):
        # 先求宽松结果取得真实峰值，再把上限压到整体峰值以下。
        sp = find_sliced_plan(self.specs, 10**9, "a", 2)
        with self.assertRaises(MemoryLimitInfeasibleError):
            find_sliced_plan(self.specs, sp.peak_memory - 1, "a", 2)

    def test_accumulator_leaves_no_room(self):
        # 输出 6 元素，片峰值为 16（4+6+6）。上限 16 时片 0 恰好可行，
        # 片 1 扣除常驻累加器 6 后只剩 10 < 16：必须拒绝，且消息定位到切片 1。
        specs = [
            spec("A", ("i", "a"), (2, 4)),
            spec("B", ("a", "j"), (4, 3)),
        ]
        with self.assertRaises(MemoryLimitInfeasibleError) as cm:
            find_sliced_plan(specs, 16, "a", 2)
        self.assertIn("切片 1", str(cm.exception))
        # 上限 22 时两片均可（整体峰值恰为 22）。
        sp = find_sliced_plan(specs, 22, "a", 2)
        self.assertEqual(sp.peak_memory, 22)

    def test_budget_exhausted_per_slice(self):
        with self.assertRaises(BudgetExhaustedError):
            find_sliced_plan(self.specs, 10**9, "a", 2,
                             search_budget=1)
        # 预算充足时同一问题成功。
        sp = find_sliced_plan(self.specs, 10**9, "a", 2,
                              search_budget=10**7)
        self.assertEqual(sp.num_slices, 2)

    def test_no_partial_state_on_failure(self):
        # 失败调用不返回任何对象；重跑合法调用不受影响（纯函数语义）。
        with self.assertRaises(InvalidProblemError):
            find_sliced_plan(self.specs, 10**9, "a", 0)
        sp = find_sliced_plan(self.specs, 10**9, "a", 2)
        self.assertEqual(sp.num_slices, 2)


# ---------------------------------------------------------------------------
# 独立参考逐片交叉验证（确定性随机网络，n=2..5）
# ---------------------------------------------------------------------------


def gen_network(rng, n):
    axes = [[] for _ in range(n)]
    sizes = {}
    nxt = [0]

    def new_label():
        a = "z%d" % nxt[0]
        nxt[0] += 1
        sizes[a] = rng.randint(2, 4)
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
    for _ in range(rng.randint(0, n)):
        i = rng.randrange(n)
        axes[i].append(new_label())
    rng.shuffle(axes)
    return [tuple(a) for a in axes], sizes


def slice_problem(axes_list, sizes, axis, width):
    """构造一片的独立参考问题（被切轴移到属主轴序末尾，维宽=width）。"""
    s_axes = []
    s_sizes = dict(sizes)
    s_sizes[axis] = width
    for ax in axes_list:
        if axis in ax:
            rest = [a for a in ax if a != axis]
            s_axes.append(tuple(rest + [axis]))
        else:
            s_axes.append(tuple(ax))
    return s_axes, s_sizes


class TestSlicedAgainstReference(unittest.TestCase):
    def test_random_networks(self):
        rng = random.Random(20260924)
        for n, count in {2: 6, 3: 12, 4: 10, 5: 6}.items():
            for _ in range(count):
                axes_list, sizes = gen_network(rng, n)
                count2 = [
                    a for a in sizes
                    if sum(a in ax for ax in axes_list) == 2
                ]
                axis = rng.choice(count2)
                dim = sizes[axis]
                chunk = rng.randint(1, dim)
                specs = [
                    spec("S%d" % i, ax, tuple(sizes[a] for a in ax))
                    for i, ax in enumerate(axes_list)
                ]
                sp = find_sliced_plan(specs, 10**18, axis, chunk)

                # 独立参考逐片核对乘法数与所选峰值。
                exp_mult = 0
                first_order = []
                for ax in axes_list:
                    for a in ax:
                        if a not in first_order:
                            first_order.append(a)
                output_axes = [
                    a for a in first_order
                    if sum(a in ax for ax in axes_list) == 1
                ]
                out_size = math.prod(sizes[a] for a in output_axes)
                exp_peak = 0
                k = 0
                lo = 0
                while lo < dim:
                    width = min(chunk, dim - lo)
                    s_ax, s_sz = slice_problem(axes_list, sizes,
                                               axis, width)
                    points = ref.all_optimal_points(s_ax, s_sz)
                    self.assertTrue(points)
                    held = 0 if k == 0 else out_size
                    # 宽松上限下核心选乘法最少点（峰值取该前沿点）。
                    m, p = points[0]
                    exp_mult += m
                    exp_peak = max(exp_peak, held + p)
                    lo += chunk
                    k += 1
                self.assertEqual(sp.total_mult_count, exp_mult)
                self.assertEqual(sp.peak_memory, exp_peak)
                self.assertEqual(sp.num_slices, k)
                assert_sliced_consistent(self, sp, specs)

                # 数值：逐片真实执行累加，必须等于独立直算。
                got = execute_sliced(sp, specs)
                want = direct_reference(specs, tuple(output_axes))
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()

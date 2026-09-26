"""联合精度规划的必要测试（含独立全枚举与 Decimal 舍入对照）。"""

import math
import unittest
from fractions import Fraction

from tests import reference_precision as ref

import tensor_peak as tp
from tensor_peak import (
    TensorSpec,
    find_plan,
    find_precision_plan,
    FP32,
    FP64,
    InvalidProblemError,
    ErrorLimitInfeasibleError,
    PrecisionInfeasibleError,
    BudgetExhaustedError,
    InfeasibleError,
    decimal_rounding_audit,
)


def make_specs(n, dims=6, seed=0):
    """确定性构造一个连通的小张量网络（标签出现 1 或 2 次）。"""
    axes = ["a%d" % i for i in range(n + 1)]
    axis_dim = {a: dims + (i * 3 + seed * 2) % 5 for i, a in enumerate(axes)}
    specs = []
    for i in range(n):
        specs.append(TensorSpec(
            "T%c" % chr(ord("A") + i),
            (axes[i], axes[i + 1]),
            (axis_dim[axes[i]], axis_dim[axes[i + 1]]),
            abs_bound=Fraction(1) + Fraction(i + 1, 10),
            abs_error=Fraction(seed + i, 10**6),
        ))
    return specs


def label_sizes(specs):
    sizes = {}
    for t in specs:
        for a, d in zip(t.axes, t.shape):
            sizes[a] = d
    return sizes


class TestCrossCheck(unittest.TestCase):
    def test_enumerate_all_orders_and_precisions(self):
        for n in (2, 3, 4, 5):
            specs = make_specs(n)
            opts = ref.reference_options(specs, label_sizes(specs),
                                         (FP32, FP64))
            self.assertGreater(len(opts), 0)
            mem_cap = max(o["peak"] for o in opts)
            err_cap = max(o["error"] for o in opts)
            for frac in (Fraction(1, 1), Fraction(1, 10),
                         Fraction(1, 10**3), Fraction(1, 10**7),
                         Fraction(1, 10**13), Fraction(0)):
                cap = err_cap * frac if frac else Fraction(0)
                expected = ref.best_under(opts, mem_cap, cap)
                if expected is None:
                    with self.assertRaises(ErrorLimitInfeasibleError):
                        find_precision_plan(specs, mem_cap, cap)
                else:
                    plan = find_precision_plan(specs, mem_cap, cap)
                    self.assertEqual(plan.total_cost, expected["cost"])
                    self.assertEqual(plan.peak_memory, expected["peak"])
                    self.assertEqual(plan.final_abs_error, expected["error"])
                    self.assertLessEqual(plan.final_abs_error, cap)
                    self.assertLessEqual(plan.peak_memory, mem_cap)

    def test_plan_uses_mixed_precision(self):
        # 大 K 步骤的舍入误差主导：紧误差上限应迫使部分步骤用 fp64。
        specs = [
            TensorSpec("A", ("i", "k"), (4, 40)),
            TensorSpec("B", ("k", "j"), (40, 4)),
            TensorSpec("C", ("j", "l"), (4, 40)),
        ]
        loose = find_precision_plan(specs, 10**9, Fraction(1, 10**3))
        tight = find_precision_plan(specs, 10**9, Fraction(1, 10**6))
        self.assertEqual([s.precision.name for s in loose.steps],
                         ["fp32", "fp32"])
        self.assertIn(FP64, [s.precision for s in tight.steps])
        self.assertGreater(tight.total_cost, loose.total_cost)
        self.assertLessEqual(tight.final_abs_error, Fraction(1, 10**6))

    def test_memory_and_error_joint(self):
        specs = make_specs(4, dims=5)
        opts = ref.reference_options(specs, label_sizes(specs),
                                     (FP32, FP64))
        tight_mem = min(o["peak"] for o in opts)
        tight_err = min(o["error"] for o in opts)
        # 内存很紧但误差宽松：可行。
        p1 = find_precision_plan(specs, tight_mem,
                                 max(o["error"] for o in opts))
        self.assertEqual(p1.peak_memory, tight_mem)
        # 两者同时取最紧：独立枚举判断是否可行。
        expected = ref.best_under(opts, tight_mem, tight_err)
        if expected is None:
            with self.assertRaises(ErrorLimitInfeasibleError):
                find_precision_plan(specs, tight_mem, tight_err)
        else:
            p2 = find_precision_plan(specs, tight_mem, tight_err)
            self.assertEqual(p2.total_cost, expected["cost"])


class TestLedger(unittest.TestCase):
    def test_step_ledger_consistency(self):
        specs = [
            TensorSpec("A", ("i", "k"), (3, 7),
                       abs_bound=Fraction(3, 2), abs_error=Fraction(1, 1000)),
            TensorSpec("B", ("k", "j"), (7, 5),
                       abs_bound=Fraction(4), abs_error=Fraction(2, 1000)),
        ]
        plan = find_precision_plan(specs, 10**9, Fraction(1, 10))
        self.assertEqual(len(plan.steps), 1)
        step = plan.steps[0]
        k = step.dot_length
        self.assertEqual(k, 7)
        gamma = k * step.precision.u / (1 - k * step.precision.u)
        self.assertEqual(
            step.rounding_error,
            gamma * k * Fraction(3, 2) * 4,
        )
        self.assertEqual(
            step.input_error_contribution,
            k * (
                Fraction(1, 1000) * 4
                + Fraction(2, 1000) * Fraction(3, 2)
                + Fraction(1, 1000) * Fraction(2, 1000)
            ),
        )
        self.assertEqual(
            step.total_abs_error,
            step.rounding_error + step.input_error_contribution,
        )
        self.assertEqual(
            step.output_stored_bound,
            step.term_mag_sum + step.rounding_error,
        )


class TestDecimalAudit(unittest.TestCase):
    def test_adversarial_decimal_rounding_within_ledger(self):
        # 用不精确对齐的非 2 的幂填充，真实触发舍入；逐步实测必须 <= 账本界。
        specs = [
            TensorSpec("A", ("i", "k"), (2, 17), abs_bound=10),
            TensorSpec("B", ("k", "j"), (17, 2), abs_bound=10),
            TensorSpec("C", ("j", "l"), (2, 17), abs_bound=10),
        ]
        plan = find_precision_plan(specs, 10**9, Fraction(1, 10))
        reports = decimal_rounding_audit(
            plan,
            fill_values={"A": Fraction(1, 3), "B": Fraction(1, 7),
                         "C": Fraction(1, 5)},
        )
        self.assertEqual(len(reports), len(plan.steps))
        for idx, observed, bound, ratio in reports:
            self.assertLessEqual(observed, bound)
            self.assertGreaterEqual(ratio, 0)
        # 非平凡舍入确实发生（不是全 0 的假对照）。
        self.assertTrue(any(observed > 0 for _, observed, _, _ in reports))

    def test_decimal_audit_rejects_prior_error(self):
        specs = [
            TensorSpec("A", ("i", "k"), (2, 4), abs_error=Fraction(1, 10**9)),
            TensorSpec("B", ("k", "j"), (4, 2)),
        ]
        plan = find_precision_plan(specs, 10**9, Fraction(1))
        with self.assertRaises(InvalidProblemError):
            decimal_rounding_audit(plan)


class TestFeasibilityAndBudget(unittest.TestCase):
    def test_gamma_failure_is_infeasible(self):
        # K = 2**24 使 fp32 的 K*u = 1，fp64 仍可行；只给 fp32 则证明无解。
        specs = [
            TensorSpec("A", ("k",), (2 ** 24,)),
            TensorSpec("B", ("k",), (2 ** 24,)),
        ]
        plan = find_precision_plan(specs, 10**9, Fraction(1),
                                   levels=(FP32, FP64))
        self.assertEqual(plan.steps[0].precision.name, "fp64")
        with self.assertRaises(InfeasibleError):
            find_precision_plan(specs, 10**9, Fraction(1), levels=(FP32,))

    def test_gamma_failure_propagates_through_chain(self):
        # K=2^24 出现在链的中间收缩，fp32-only 时原因必须沿子集传播。
        specs = [
            TensorSpec("X", ("a", "k"), (2, 2 ** 24)),
            TensorSpec("Y", ("k", "b"), (2 ** 24, 2)),
            TensorSpec("Z", ("b", "c"), (2, 3)),
        ]
        with self.assertRaises(PrecisionInfeasibleError):
            find_precision_plan(specs, 10**9, Fraction(1), levels=(FP32,))
        plan = find_precision_plan(specs, 10**9, Fraction(1))
        self.assertEqual([s.precision.name for s in plan.steps],
                         ["fp64", "fp64"])

    def test_budget_separate_from_infeasible(self):
        specs = make_specs(5)
        with self.assertRaises(BudgetExhaustedError):
            find_precision_plan(specs, 10**9, Fraction(1, 10**30),
                                search_budget=1)
        # 预算充足时同一问题给出正常计划。
        plan = find_precision_plan(specs, 10**9, Fraction(1))
        self.assertGreater(plan.total_cost, 0)

    def test_infeasible_error_message_distinct(self):
        specs = make_specs(3)
        opts = ref.reference_options(specs, label_sizes(specs),
                                     (FP32, FP64))
        smaller_than_all = min(o["error"] for o in opts) / 2
        with self.assertRaises(ErrorLimitInfeasibleError):
            find_precision_plan(specs, 10**9, smaller_than_all)

    def test_invalid_bounds(self):
        with self.assertRaises(InvalidProblemError):
            TensorSpec("A", ("i",), (2,), abs_bound=-1)
        with self.assertRaises(InvalidProblemError):
            TensorSpec("A", ("i",), (2,), abs_error=float("nan"))
        specs = [TensorSpec("A", ("i", "k"), (2, 3)),
                 TensorSpec("B", ("k", "j"), (3, 2))]
        with self.assertRaises(InvalidProblemError):
            find_precision_plan(specs, 10**9, -1)


class TestOldInterfaceKept(unittest.TestCase):
    def test_memory_only_plan_unchanged(self):
        specs = [
            TensorSpec("A", ("i", "k"), (10, 20)),
            TensorSpec("B", ("k", "j"), (20, 30)),
            TensorSpec("C", ("j", "l"), (30, 40)),
        ]
        plan = find_plan(specs, 10**9)
        self.assertEqual(plan.total_mult_count, 18000)
        self.assertEqual(plan.peak_memory, 1900)

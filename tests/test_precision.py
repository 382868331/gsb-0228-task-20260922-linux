"""混合精度规划的独立 Decimal 对照测试。"""

import itertools
import math
import unittest
from decimal import Decimal

from tensor_peak import (
    DEFAULT_PRECISION_LEVELS,
    BudgetExhaustedError,
    ConstraintsInfeasibleError,
    TensorSpec,
    find_precision_plan,
)


def specs_fixture():
    return [
        TensorSpec("A", ("i", "k"), (2, 3)),
        TensorSpec("B", ("k", "j"), (3, 2)),
        TensorSpec("C", ("j", "l"), (2, 4)),
    ]


def axes_tables(specs):
    order = []
    sizes = {}
    for tensor in specs:
        for axis, size in zip(tensor.axes, tensor.shape):
            sizes[axis] = size
            if axis not in order:
                order.append(axis)
    n = len(specs)
    axes_by_mask = {}
    size_by_mask = {}
    for mask in range(1, 1 << n):
        counts = {axis: 0 for axis in order}
        for bit, tensor in enumerate(specs):
            if mask & (1 << bit):
                for axis in tensor.axes:
                    counts[axis] += 1
        axes = tuple(axis for axis in order if counts[axis] == 1)
        axes_by_mask[mask] = axes
        size_by_mask[mask] = math.prod(sizes[axis] for axis in axes)
    return order, sizes, axes_by_mask, size_by_mask


def reference_candidates(specs, bounds, errors, levels):
    _order, sizes, axes_by_mask, size_by_mask = axes_tables(specs)
    n = len(specs)
    leaves = {
        1 << i: [
            {
                "mask": 1 << i,
                "tree": specs[i].name,
                "peak": 0,
                "cost": 0,
                "nominal": bounds[specs[i].name],
                "error": errors[specs[i].name],
            }
        ]
        for i in range(n)
    }

    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        candidates = []
        for left_mask in range(1, mask):
            if not (left_mask & low and (left_mask & mask) == left_mask):
                continue
            right_mask = mask ^ left_mask
            if not (set(axes_by_mask[left_mask]) & set(axes_by_mask[right_mask])):
                continue
            for left, right, level in itertools.product(
                leaves[left_mask], leaves[right_mask], levels
            ):
                shared = set(axes_by_mask[left_mask]) & set(axes_by_mask[right_mask])
                length = math.prod(sizes[axis] for axis in shared)
                union = set(axes_by_mask[left_mask]) | set(axes_by_mask[right_mask])
                muls = math.prod(sizes[axis] for axis in union)
                ku = Decimal(length) * level.unit_roundoff
                gamma = ku / (Decimal(1) - ku)
                nominal = Decimal(length) * left["nominal"] * right["nominal"]
                inherited = Decimal(length) * (
                    left["nominal"] * right["error"]
                    + right["nominal"] * left["error"]
                    + left["error"] * right["error"]
                )
                rounded = gamma * (Decimal(1) + Decimal("1e-60")) * (
                    left["nominal"] + left["error"]
                ) * (right["nominal"] + right["error"])
                error = inherited + rounded
                triple = (
                    size_by_mask[left_mask]
                    + size_by_mask[right_mask]
                    + size_by_mask[mask]
                )
                peak_l = max(left["peak"], size_by_mask[left_mask] + right["peak"], triple)
                peak_r = max(right["peak"], size_by_mask[right_mask] + left["peak"], triple)
                peak = min(peak_l, peak_r)
                candidates.append({
                    "mask": mask,
                    "tree": (left["tree"], right["tree"], level.name),
                    "peak": peak,
                    "cost": left["cost"] + right["cost"] + muls * level.cost_per_mult,
                    "nominal": nominal,
                    "error": error,
                })
        leaves[mask] = candidates
    return leaves[(1 << n) - 1]


class PrecisionPlanningTests(unittest.TestCase):
    def test_reference_enumeration_chooses_optimal_mixed_precision(self):
        specs = specs_fixture()
        bounds = {"A": Decimal(1), "B": Decimal(1), "C": Decimal(1)}
        errors = {"A": Decimal(0), "B": Decimal(0), "C": Decimal(0)}
        limit = Decimal("0.0001")
        plan = find_precision_plan(
            specs, 10**9, limit, bounds,
            input_error_bounds=errors,
        )
        candidates = reference_candidates(
            specs, bounds, errors, DEFAULT_PRECISION_LEVELS
        )
        feasible = [c for c in candidates if c["error"] <= limit]
        best = min(feasible, key=lambda c: (c["cost"], c["peak"], c["error"]))
        self.assertEqual(plan.total_cost, best["cost"])
        self.assertEqual(plan.peak_memory, best["peak"])
        self.assertEqual(plan.final_error_bound, best["error"])
        names = [step.precision.name for step in plan.steps]
        self.assertEqual(names, ["high", "high"])
        self.assertEqual(plan.total_cost, 12 * 8 + 16 * 8)
        self.assertLess(plan.final_error_bound, limit)

    def test_tight_error_boundaries_select_mixed_and_high_precision(self):
        specs = specs_fixture()
        bounds = {"A": Decimal(1), "B": Decimal(1), "C": Decimal(1)}
        errors = {"A": Decimal(0), "B": Decimal(0), "C": Decimal(0)}
        plan = find_precision_plan(specs, 10**9, Decimal("0.02"), bounds)
        self.assertEqual([s.precision.name for s in plan.steps], ["low", "low"])
        self.assertEqual(plan.total_cost, 12 + 16)
        mixed = find_precision_plan(specs, 10**9, Decimal("0.01203"), bounds)
        self.assertEqual([s.precision.name for s in mixed.steps], ["high", "low"])
        self.assertEqual(mixed.total_cost, 12 * 8 + 16)
        with self.assertRaises(ConstraintsInfeasibleError):
            find_precision_plan(specs, 10**9, Decimal("0.00001"), bounds)

    def test_input_error_is_propagated_conservatively(self):
        specs = [
            TensorSpec("A", ("k",), (3,)),
            TensorSpec("B", ("k",), (3,)),
        ]
        bounds = {"A": Decimal(2), "B": Decimal(5)}
        errors = {"A": Decimal("0.2"), "B": Decimal(0)}
        plan = find_precision_plan(
            specs, 10**9, Decimal(4), bounds,
            input_error_bounds=errors,
        )
        step = plan.steps[0]
        self.assertEqual(step.precision.name, "low")
        self.assertEqual(step.input_error_bound, Decimal(3))
        self.assertEqual(
            step.rounding_error_bound,
            step.gamma_n * Decimal(11) * (Decimal(1) + Decimal("1e-60")),
        )
        self.assertEqual(plan.final_error_bound, Decimal(3) + step.rounding_error_bound)

    def test_memory_and_budget_are_separate_from_proved_infeasibility(self):
        specs = specs_fixture()
        bounds = {name: Decimal(1) for name in "ABC"}
        with self.assertRaises(ConstraintsInfeasibleError):
            find_precision_plan(specs, 1, Decimal(1), bounds)
        with self.assertRaises(BudgetExhaustedError):
            find_precision_plan(
                specs, 10**9, Decimal("0.00001"), bounds,
                search_budget=1,
            )

    def test_every_step_has_decimal_memory_and_error_ledger(self):
        specs = specs_fixture()
        plan = find_precision_plan(
            specs,
            10**9,
            Decimal("0.0001"),
            {name: Decimal(1) for name in "ABC"},
        )
        self.assertEqual(plan.peak_memory, max(s.peak_at_step for s in plan.steps))
        for step in plan.steps:
            self.assertIsInstance(step.gamma_n, Decimal)
            self.assertIsInstance(step.output_error_bound, Decimal)
            self.assertEqual(
                step.output_error_bound,
                step.input_error_bound + step.rounding_error_bound,
            )
            self.assertEqual(
                step.output_magnitude_bound,
                step.nominal_magnitude + step.output_error_bound,
            )


if __name__ == "__main__":
    unittest.main()

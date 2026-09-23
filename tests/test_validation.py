"""Contract tests for public data structures, validation and error types."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contraction_planner import (  # noqa: E402
    InfeasiblePlanError,
    InvalidProblemError,
    SearchBudgetExhaustedError,
    Tensor,
    plan_contraction,
)

CHAIN = [
    Tensor(("i", "k"), (2, 3)),
    Tensor(("k", "l"), (3, 4)),
    Tensor(("l", "j"), (4, 5)),
]


class ValidationTests(unittest.TestCase):
    def test_bool_sizes_rejected(self):
        bad = Tensor(("a", "b"), (True, 3))
        with self.assertRaises(TypeError) as cm:
            plan_contraction([bad, Tensor(("a",), (2,))], 10**9)
        self.assertIn("bool", str(cm.exception))

    def test_bool_memory_cap_rejected(self):
        with self.assertRaises(TypeError):
            plan_contraction(CHAIN, True)

    def test_nan_and_infinity_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError) as cm:
                plan_contraction(
                    [Tensor(("a",), (bad,)), Tensor(("a",), (2,))], 10**9
                )
            self.assertIn("finite", str(cm.exception))
            with self.assertRaises(ValueError):
                plan_contraction(CHAIN, bad)

    def test_zero_and_negative_sizes_rejected(self):
        with self.assertRaises(InvalidProblemError):
            plan_contraction([Tensor(("a",), (0,))], 10**9)
        with self.assertRaises(InvalidProblemError):
            plan_contraction(
                [Tensor(("a",), (-2,)), Tensor(("a",), (2,))], 10**9
            )

    def test_errors_are_locateable(self):
        # Messages identify the offending tensor and axis / parameter.
        with self.assertRaises(InvalidProblemError) as cm:
            plan_contraction(
                [
                    Tensor(("a", "b"), (2, 3)),
                    Tensor(("a", "b"), (2, 4)),  # b size mismatch
                ],
                10**9,
            )
        msg = str(cm.exception)
        self.assertIn("'b'", msg)
        self.assertIn("tensors[1]", msg)

        with self.assertRaises(InvalidProblemError) as cm:
            plan_contraction(
                [
                    Tensor(("a", "x"), (2, 2)),
                    Tensor(("x", "y"), (2, 2)),
                    Tensor(("y", "a"), (2, 2)),
                    Tensor(("a", "q"), (2, 2)),  # a now occurs 3 times
                ],
                10**9,
            )
        self.assertIn("'a'", str(cm.exception))

    def test_axes_shape_length_must_match(self):
        with self.assertRaises(InvalidProblemError):
            plan_contraction([Tensor(("a", "b"), (2,))], 10**9)

    def test_empty_tensor_rejected(self):
        with self.assertRaises(InvalidProblemError):
            plan_contraction([Tensor((), ())], 10**9)

    def test_duplicate_label_in_one_tensor_rejected(self):
        with self.assertRaises(InvalidProblemError):
            plan_contraction([Tensor(("a", "a"), (2, 2))], 10**9)

    def test_tensor_count_limits(self):
        with self.assertRaises(InvalidProblemError):
            plan_contraction([], 10**9)
        eleven = [Tensor((f"p{i}",), (2,)) for i in range(11)]
        with self.assertRaises(InvalidProblemError):
            plan_contraction(eleven, 10**9)
        # The documented upper bound itself is accepted.
        ten = [
            Tensor((f"x{i}", f"x{i+1}"), (2, 2)) for i in range(10)
        ]
        plan = plan_contraction(ten, 10**30)
        self.assertEqual(len(plan.steps), 9)

    def test_wrong_container_type(self):
        with self.assertRaises(TypeError):
            plan_contraction((t for t in CHAIN), 10**9)

    def test_non_string_axis_rejected(self):
        with self.assertRaises(TypeError):
            plan_contraction([Tensor((1,), (2,))], 10**9)

    def test_node_budget_validation(self):
        with self.assertRaises(TypeError):
            plan_contraction(CHAIN, 10**9, node_budget=True)
        with self.assertRaises(InvalidProblemError):
            plan_contraction(CHAIN, 10**9, node_budget=0)

    def test_budget_and_infeasible_are_distinct_types(self):
        # Budget cutoff cannot be mistaken for a proof of impossibility.
        with self.assertRaises(SearchBudgetExhaustedError) as cm:
            plan_contraction(CHAIN, 10**9, node_budget=1)
        self.assertNotIsInstance(
            cm.exception, InfeasiblePlanError
        )

    def test_single_input_too_large_for_cap(self):
        with self.assertRaises(InfeasiblePlanError) as cm:
            plan_contraction([Tensor(("a",), (100,))], 50)
        self.assertIn("100", str(cm.exception))

    def test_planner_is_pure_no_state_change(self):
        before = [(t.axes, t.shape) for t in CHAIN]
        plan_contraction(CHAIN, 10**9)
        plan_contraction(CHAIN, 10**9, node_budget=10**6)
        after = [(t.axes, t.shape) for t in CHAIN]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

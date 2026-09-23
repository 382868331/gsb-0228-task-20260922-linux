"""Supplementary executor tests.

Expected values come from an independent nested-loop contraction written in
the test itself; the planner core is never used to compute expected values.
"""

from __future__ import annotations

import os
import sys
import unittest
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contraction_planner import (  # noqa: E402
    DenseArray,
    Tensor,
    execute_plan,
    plan_contraction,
)
from contraction_planner.sample_data import matrix_chain, tradeoff_network  # noqa: E402


def brute_einsum(tensors, arrays, out_axes):
    """Independent reference: sum over every contracted label by loops."""
    dims = {ax: s for t in tensors for ax, s in zip(t.axes, t.shape)}
    all_axes = []
    for t in tensors:
        for ax in t.axes:
            if ax not in all_axes:
                all_axes.append(ax)
    summed = [ax for ax in all_axes if ax not in out_axes]
    lookup = [
        {idx: v for idx, v in zip(product(*(range(s) for s in t.shape)),
                                  arr.data)}
        for t, arr in zip(tensors, arrays)
    ]
    result = {}
    for out_idx in product(*(range(dims[ax]) for ax in out_axes)):
        env = dict(zip(out_axes, out_idx))
        total = 0
        for sum_idx in product(*(range(dims[ax]) for ax in summed)):
            env.update(zip(summed, sum_idx))
            term = 1
            for t, table in zip(tensors, lookup):
                term *= table[tuple(env[ax] for ax in t.axes)]
            total += term
        result[out_idx] = total
    return result


class ExecutorTests(unittest.TestCase):
    def _run_exact(self, tensors, value_fn):
        arrays = [
            DenseArray.from_function(t.axes, t.shape, value_fn) for t in tensors
        ]
        plan = plan_contraction(tensors, 10**12)
        result, peak = execute_plan(plan, arrays)
        expected = brute_einsum(tensors, arrays, plan.output_axes)
        for idx, val in expected.items():
            self.assertEqual(result.get(dict(zip(plan.output_axes, idx))), val)
        self.assertEqual(result.axes, plan.output_axes)
        self.assertEqual(peak, plan.peak_memory)
        self.assertLessEqual(peak, plan.memory_cap)
        return plan, result

    def test_matrix_chain_exact_values(self):
        net = matrix_chain()
        plan, _ = self._run_exact(
            net, lambda idx: (idx[0] + 1) * 7 + idx[1] * 3 + 1
        )
        self.assertEqual(plan.total_multiplications, 64)

    def test_tradeoff_network_executes_under_cap(self):
        net = tradeoff_network()
        plan = plan_contraction(net, 202)  # forces the 812/182 tree
        arrays = [
            DenseArray.from_function(
                t.axes,
                t.shape,
                lambda idx, base=i: base
                + sum((k + 1) * (v + 1) for k, v in enumerate(idx)),
            )
            for i, t in enumerate(net)
        ]
        result, peak = execute_plan(plan, arrays)
        expected = brute_einsum(net, arrays, plan.output_axes)
        for idx, val in expected.items():
            self.assertEqual(result.get(dict(zip(plan.output_axes, idx))), val)
        self.assertEqual(peak, 182)

    def test_array_plan_mismatch_rejected(self):
        net = matrix_chain()
        plan = plan_contraction(net, 10**9)
        arrays = [
            DenseArray.from_function(t.axes, (s + 1 for s in t.shape),
                                     lambda idx: 1)  # wrong shape
            for t in net
        ]
        with self.assertRaises(ValueError):
            execute_plan(plan, arrays)

    def test_wrong_array_count_rejected(self):
        net = matrix_chain()
        plan = plan_contraction(net, 10**9)
        with self.assertRaises(ValueError):
            execute_plan(plan, [])

    def test_single_tensor_plan(self):
        net = [Tensor(("a", "b"), (3, 4))]
        plan = plan_contraction(net, 10**9)
        arr = DenseArray.from_function(("a", "b"), (3, 4),
                                       lambda idx: idx[0] * 10 + idx[1])
        result, peak = execute_plan(plan, [arr])
        self.assertEqual(result.data, arr.data)
        self.assertEqual(peak, 12)


if __name__ == "__main__":
    unittest.main()

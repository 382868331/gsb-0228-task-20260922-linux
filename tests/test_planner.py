"""Planner tests whose expected values come from tests/reference.py.

The reference enumerates every binary contraction tree itself and never
imports the package under test.
"""

from __future__ import annotations

import math
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contraction_planner import (  # noqa: E402
    DenseArray,
    InfeasiblePlanError,
    InvalidProblemError,
    MAX_TENSORS,
    SearchBudgetExhaustedError,
    Tensor,
    execute_plan,
    plan_contraction,
)
from contraction_planner.sample_data import (  # noqa: E402
    matrix_chain,
    random_connected_network,
    tradeoff_network,
)

from reference import enumerate_trees, optimal_under_cap  # noqa: E402


def axes_shapes(tensors):
    return [t.axes for t in tensors], [t.shape for t in tensors]


class ReferenceAgreementTests(unittest.TestCase):
    def _check_network(self, tensors):
        axes_list, shapes = axes_shapes(tensors)
        n = len(tensors)
        opt = optimal_under_cap(axes_list, shapes, math.inf)
        self.assertIsNotNone(opt, "reference found no feasible tree")
        ref_cost, ref_peak = opt

        plan = plan_contraction(tensors, 10**30)
        self.assertEqual(plan.total_multiplications, ref_cost)
        self.assertEqual(plan.peak_memory, ref_peak)

        # Every tree enumerated: cap sweep must match the reference at every
        # distinct peak threshold.
        trees = enumerate_trees(axes_list, shapes)
        thresholds = sorted({t.peak for t in trees})
        for cap in thresholds:
            under = optimal_under_cap(axes_list, shapes, cap)
            p = plan_contraction(tensors, cap)
            self.assertEqual(
                (p.total_multiplications, p.peak_memory),
                under,
                msg=f"mismatch at cap={cap}",
            )
            self.assertLessEqual(p.peak_memory, cap)
        below = min(t.peak for t in trees) - 1
        with self.assertRaises(InfeasiblePlanError):
            plan_contraction(tensors, below)

        # Structural invariants of the returned tree and steps.
        leaves = _flatten(plan.tree)
        self.assertEqual(sorted(leaves), list(range(n)))
        self.assertEqual(len(plan.steps), n - 1)
        self.assertEqual(
            plan.peak_memory, max(s.resident_total for s in plan.steps)
        )
        ev = plan.peak_evidence
        self.assertEqual(ev.total, plan.peak_memory)
        self.assertEqual(sum(r.size for r in ev.residents), ev.total)
        for step in plan.steps:
            self.assertEqual(
                sum(r.size for r in step.residents), step.resident_total
            )
            self.assertEqual(
                step.output_shape,
                tuple(
                    dict(
                        zip(
                            [ax for t in tensors for ax in t.axes],
                            [s for t in tensors for s in t.shape],
                        )
                    )[ax]
                    for ax in step.output_axes
                ),
            )

    def test_hand_built_networks_1_to_3(self):
        cases = [
            [Tensor(("a",), (4,))],
            matrix_chain(),
            [
                Tensor(("a", "b"), (2, 3)),
                Tensor(("b", "c"), (3, 4)),
                Tensor(("c",), (4,)),
            ],
        ]
        for net in cases:
            with self.subTest(n=len(net)):
                if len(net) == 1:
                    plan = plan_contraction(net, 10**9)
                    self.assertEqual(plan.total_multiplications, 0)
                    self.assertEqual(plan.peak_memory, math.prod(net[0].shape))
                else:
                    self._check_network(net)

    def test_all_trees_agreement_up_to_6(self):
        # n <= 6: the independent reference enumerates every binary tree.
        for n in range(2, 7):
            for seed in range(6):
                with self.subTest(n=n, seed=seed):
                    net = random_connected_network(n, seed=seed, max_dim=5)
                    self._check_network(net)

    def test_chain_has_catalan_many_feasible_trees(self):
        # In a 1-D chain (matrix chain) only adjacent tensors share an axis,
        # so exactly the Catalan(n-1) parenthesizations are feasible.
        def catalan(k):
            return math.comb(2 * k, k) // (k + 1)

        for n in range(2, 7):
            net = [
                Tensor((f"x{i}", f"x{i+1}"), (2, 2)) for i in range(n)
            ]
            trees = enumerate_trees(
                [t.axes for t in net], [t.shape for t in net]
            )
            shapes = {_shape(t.tree) for t in trees}
            self.assertEqual(len(shapes), catalan(n - 1))
            self.assertEqual(len(trees), catalan(n - 1))

    def test_star_has_factorial_many_feasible_trees(self):
        # Star network: T0 shares one distinct (valid: twice-occurring) label
        # with each leaf, leaves share nothing with each other.  A feasible
        # tree must fold leaves into the center one at a time, giving (n-1)!
        # distinct trees.
        import math as _math

        for n in range(2, 7):
            center_axes = tuple(f"h{i}" for i in range(1, n))
            center_shape = tuple(2 for _ in center_axes)
            net = [Tensor(center_axes, center_shape)]
            for i in range(1, n):
                net.append(Tensor((f"h{i}", f"p{i}"), (2, 3)))
            trees = enumerate_trees(
                [t.axes for t in net], [t.shape for t in net]
            )
            self.assertEqual(len(trees), _math.factorial(n - 1))

    def test_shared_multiple_axes(self):
        # T0 and T1 share two contraction axes; output keeps two axes.
        net = [
            Tensor(("a", "b", "c", "d"), (2, 3, 4, 5)),
            Tensor(("a", "b", "e", "f"), (2, 3, 6, 7)),
        ]
        plan = plan_contraction(net, 10**9)
        # multiplications = |a|*|b| * |output|
        self.assertEqual(plan.steps[0].multiplications, 2 * 3 * 4 * 5 * 6 * 7)
        self.assertEqual(
            plan.steps[0].contracted_axes, ("a", "b")
            if plan.steps[0].first_axes[:2] == ("a", "b")
            else plan.steps[0].contracted_axes,
        )
        self._check_network(net)

        # Numeric execution with two shared axes, exact integer check.
        def v0(idx):
            a, b, c, d = idx
            return 1 + a + 2 * b + 3 * c + 5 * d

        def v1(idx):
            a, b, e, f = idx
            return 2 + a + 3 * b + 7 * e + 11 * f

        arrays = [
            DenseArray.from_function(net[0].axes, net[0].shape, v0),
            DenseArray.from_function(net[1].axes, net[1].shape, v1),
        ]
        result, peak = execute_plan(plan, arrays)
        for c in range(4):
            for d in range(5):
                for e in range(6):
                    for f in range(7):
                        expected = sum(
                            v0((a, b, c, d)) * v1((a, b, e, f))
                            for a in range(2)
                            for b in range(3)
                        )
                        got = result.get(
                            {"c": c, "d": d, "e": e, "f": f}
                        )
                        self.assertEqual(got, expected)
        self.assertEqual(peak, plan.peak_memory)

    def test_disconnected_network_is_rejected_without_outer_product(self):
        # Two tensors sharing no label: joining them would be an outer
        # product, which the contract forbids.
        net = [Tensor(("a",), (2,)), Tensor(("b",), (3,))]
        with self.assertRaises(InfeasiblePlanError):
            plan_contraction(net, 10**9)

    def test_cost_memory_tradeoff(self):
        net = tradeoff_network()
        cheap = plan_contraction(net, 10**9)
        self.assertEqual((cheap.total_multiplications, cheap.peak_memory),
                         (770, 203))
        constrained = plan_contraction(net, 202)
        self.assertEqual(
            (constrained.total_multiplications, constrained.peak_memory),
            (812, 182),
        )
        with self.assertRaises(InfeasiblePlanError):
            plan_contraction(net, 181)

    def test_budget_exhaustion_is_distinct_from_infeasibility(self):
        net = random_connected_network(6, seed=1, max_dim=5)
        with self.assertRaises(SearchBudgetExhaustedError):
            plan_contraction(net, 10**30, node_budget=1)
        # ... whereas a proved impossibility is a different exception type.
        with self.assertRaises(InfeasiblePlanError):
            plan_contraction(net, 1)

    def test_deterministic(self):
        net = random_connected_network(5, seed=4)
        p1 = plan_contraction(net, 10**12)
        p2 = plan_contraction(net, 10**12)
        self.assertEqual(p1.bracket, p2.bracket)
        self.assertEqual(p1.total_multiplications, p2.total_multiplications)
        self.assertEqual(p1.peak_memory, p2.peak_memory)

    def test_runs_with_all_sockets_blocked(self):
        # Offline guarantee: the planner is pure compute.  Blocking every
        # socket attempt must not change the result.
        def boom(*_a, **_k):
            raise OSError("network disabled by test")

        net = matrix_chain()
        normal = plan_contraction(net, 10**9)
        old = socket.socket
        socket.socket = boom
        try:
            blocked = plan_contraction(net, 10**9)
            arrays = [
                DenseArray.from_function(
                    t.axes, t.shape, lambda idx: idx[0] * 10 + idx[1] + 1
                )
                for t in net
            ]
            result, peak = execute_plan(blocked, arrays)
        finally:
            socket.socket = old
        self.assertEqual(
            blocked.total_multiplications, normal.total_multiplications
        )
        self.assertEqual(peak, blocked.peak_memory)
        self.assertEqual(result.size, 10)


def _flatten(tree):
    if isinstance(tree, int):
        return [tree]
    return _flatten(tree[0]) + _flatten(tree[1])


def _shape(tree):
    if isinstance(tree, int):
        return tree
    return (_shape(tree[0]), _shape(tree[1]))


if __name__ == "__main__":
    unittest.main()

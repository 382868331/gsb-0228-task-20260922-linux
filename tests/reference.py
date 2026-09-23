"""Independent reference: exhaustive enumeration of every binary tree.

This module deliberately does **not** import the planner core.  It restates
the problem's own math (cost of one pairwise contraction and the resident
memory model) and enumerates every full parenthesization for networks of up
to ~8 tensors.  Tests use its results as expected values; the planner must
agree without the reference ever calling planner code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ReferenceTree", "enumerate_trees", "optimal_under_cap"]


@dataclass(frozen=True)
class ReferenceTree:
    tree: object           # int leaf or (left, right)
    cost: int              # total scalar multiplications
    peak: int              # peak resident scalars
    mult: int              # multiplications at the root
    out_axes: tuple[str, ...]


def _enumerate(mask, axes_list, dim_of, label_masks, memo):
    if mask in memo:
        return memo[mask]
    if mask & (mask - 1) == 0:
        i = mask.bit_length() - 1
        memo[mask] = (ReferenceTree(i, 0, 0, 0, tuple(axes_list[i])),)
        return memo[mask]

    pivot = mask & -mask
    rest = mask ^ pivot
    results = []
    sub = rest
    while True:
        left_mask = sub | pivot
        right_mask = mask ^ left_mask
        if right_mask != 0:  # non-trivial bipartitions only
            for lft in _enumerate(left_mask, axes_list, dim_of, label_masks, memo):
                for rgt in _enumerate(
                    right_mask, axes_list, dim_of, label_masks, memo
                ):
                    shared = tuple(
                        ax
                        for ax in lft.out_axes
                        if ax in set(rgt.out_axes)
                    )
                    if not shared:
                        continue  # outer products are forbidden
                    out_axes = tuple(
                        ax
                        for ax in lft.out_axes + rgt.out_axes
                        if ax not in shared
                    )
                    out_size = math.prod(dim_of[ax] for ax in out_axes)
                    mult = out_size * math.prod(dim_of[ax] for ax in shared)
                    cost = lft.cost + rgt.cost + mult

                    s_l = math.prod(dim_of[ax] for ax in lft.out_axes)
                    s_r = math.prod(dim_of[ax] for ax in rgt.out_axes)
                    hold_l = 0 if isinstance(lft.tree, int) else s_l
                    hold_r = 0 if isinstance(rgt.tree, int) else s_r
                    triple = s_l + s_r + out_size
                    # The executor may build either operand first; take the order
                    # with the smaller peak, mirroring the problem's memory model.
                    peak_l_first = max(lft.peak, hold_l + rgt.peak, triple)
                    peak_r_first = max(rgt.peak, hold_r + lft.peak, triple)
                    peak = min(peak_l_first, peak_r_first)

                    results.append(
                        ReferenceTree(
                            (lft.tree, rgt.tree), cost, peak, mult, out_axes
                        )
                    )
        if sub == 0:
            break
        sub = (sub - 1) & rest
    memo[mask] = tuple(results)
    return memo[mask]


def enumerate_trees(axes_list, shapes):
    """Return every feasible contraction tree with cost and peak."""
    n = len(axes_list)
    dim_of: dict[str, int] = {}
    label_masks: dict[str, int] = {}
    for i, (axes, shape) in enumerate(zip(axes_list, shapes)):
        for ax, size in zip(axes, shape):
            dim_of[ax] = size
            label_masks[ax] = label_masks.get(ax, 0) | (1 << i)
    memo: dict[int, tuple[ReferenceTree, ...]] = {}
    return _enumerate((1 << n) - 1, axes_list, dim_of, label_masks, memo)


def optimal_under_cap(axes_list, shapes, cap):
    """Return (min_cost, min_peak_at_that_cost) among trees with peak<=cap.

    Returns ``None`` when no tree fits the cap.
    """
    trees = enumerate_trees(axes_list, shapes)
    feasible = [t for t in trees if t.peak <= cap]
    if not feasible:
        return None
    best_cost = min(t.cost for t in feasible)
    best_peak = min(t.peak for t in feasible if t.cost == best_cost)
    return best_cost, best_peak

"""Tiny dense-array executor that runs a :class:`ContractionPlan`.

This is *not* part of the planning algorithm: it exists so a plan can be
executed on small tensors to verify both the numbers and the peak-memory
claim.  Arrays are flat Python lists of numbers (exact ``int`` or ``float``);
nothing here requires NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

from .core import ContractionPlan, OperandRef

__all__ = ["DenseArray", "contract", "execute_plan"]


@dataclass(frozen=True)
class DenseArray:
    """A row-major dense tensor stored as a flat list of scalars."""

    axes: tuple[str, ...]
    shape: tuple[int, ...]
    data: list

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @classmethod
    def from_function(
        cls,
        axes: Sequence[str],
        shape: Sequence[int],
        fn: Callable[[tuple[int, ...]], object],
    ) -> "DenseArray":
        axes = tuple(axes)
        shape = tuple(shape)
        data = [fn(idx) for idx in product(*(range(s) for s in shape))]
        return cls(axes, shape, data)

    def get(self, index: dict[str, int]):
        flat = 0
        for ax, s in zip(self.axes, self.shape):
            flat = flat * s + index[ax]
        return self.data[flat]

    def allclose(self, other: "DenseArray", *, rtol: float = 1e-12,
                 atol: float = 1e-12) -> bool:
        if self.axes != other.axes or self.shape != other.shape:
            return False
        for x, y in zip(self.data, other.data):
            if abs(x - y) > atol + rtol * abs(y):
                return False
        return True


def _strides(shape):
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    return tuple(reversed(strides))


def contract(a: DenseArray, b: DenseArray) -> DenseArray:
    """Multiply two arrays over their shared axes.

    Output axes are ``a``'s axes followed by ``b``'s non-shared axes, which is
    the ordering the planner assigns to a contraction result.
    """
    bpos = {ax: j for j, ax in enumerate(b.axes)}
    shared = tuple(ax for ax in a.axes if ax in bpos)
    if not shared:
        raise ValueError("contract: operands share no axis (outer product)")
    out_axes = tuple(ax for ax in a.axes if ax not in shared) + tuple(
        ax for ax in b.axes if ax not in shared
    )
    dim = dict(zip(a.axes, a.shape))
    for ax, s in zip(b.axes, b.shape):
        dim.setdefault(ax, s)
    out_shape = tuple(dim[ax] for ax in out_axes)

    sa = _strides(a.shape)
    sb = _strides(b.shape)
    stride_a = dict(zip(a.axes, sa))
    stride_b = dict(zip(b.axes, sb))
    base_b_shared = sum(stride_b[ax] for ax in shared)  # removed per coord below

    contraction_ranges = [range(b.shape[bpos[ax]]) for ax in shared]
    out = []
    a_index0 = {ax: 0 for ax in a.axes}
    for out_idx in product(*(range(s) for s in out_shape)):
        o = dict(zip(out_axes, out_idx))
        fa = sum(stride_a[ax] * o[ax] for ax in a.axes if ax not in shared)
        fb = sum(stride_b[ax] * o[ax] for ax in b.axes if ax not in shared)
        total = 0
        for kv in product(*contraction_ranges):
            ia = fa
            ib = fb
            for ax, v in zip(shared, kv):
                ia += stride_a[ax] * v
                ib += stride_b[ax] * v
            total += a.data[ia] * b.data[ib]
        out.append(total)
    return DenseArray(out_axes, out_shape, out)


def _resolve(ref: OperandRef, loaded: dict, produced: dict) -> DenseArray:
    return loaded[ref.index] if ref.kind == "input" else produced[ref.index]


def _reorder_axes(arr: DenseArray, target_axes: Sequence[str]) -> DenseArray:
    """Permute ``arr`` into ``target_axes`` order (same labels, same values)."""
    target_axes = tuple(target_axes)
    if arr.axes == target_axes:
        return arr
    dim = dict(zip(arr.axes, arr.shape))
    data = [
        arr.get(dict(zip(target_axes, idx)))
        for idx in product(*(range(dim[ax]) for ax in target_axes))
    ]
    return DenseArray(
        target_axes, tuple(dim[ax] for ax in target_axes), data
    )


def execute_plan(
    plan: ContractionPlan, arrays: Sequence[DenseArray]
) -> tuple[DenseArray, int]:
    """Execute ``plan`` on aligned input arrays.

    Inputs are loaded lazily at their consuming step and consumed operands are
    released immediately, matching the planner's memory model.  The resident
    scalar count observed at every contraction is checked against the plan's
    own per-step record, so an executed plan validates its peak evidence.

    Returns ``(result_array, observed_peak)``.
    """
    if len(arrays) != len(plan.inputs):
        raise ValueError(
            f"expected {len(plan.inputs)} input arrays, got {len(arrays)}"
        )
    for i, (spec, arr) in enumerate(zip(plan.inputs, arrays)):
        if arr.axes != spec.axes or arr.shape != spec.shape:
            raise ValueError(
                f"input {i} ({spec.name}): plan expects axes/shape "
                f"{spec.axes}/{spec.shape}, array has {arr.axes}/{arr.shape}"
            )

    # Degenerate single-input plan: the input itself is the result.
    if not plan.steps:
        arr = arrays[plan.tree]
        return arr, arr.size

    loaded: dict[int, DenseArray] = {}
    produced: dict[int, DenseArray] = {}
    live: set[str] = set()          # "in:<i>" / "step:<j>"
    observed_peak = 0

    for step in plan.steps:
        for ref in (step.first, step.second):
            if ref.kind == "input" and ref.index not in loaded:
                loaded[ref.index] = arrays[ref.index]
                live.add(f"in:{ref.index}")
        a = _resolve(step.first, loaded, produced)
        b = _resolve(step.second, loaded, produced)
        out = contract(a, b)
        # Planner tensors carry a canonical axis order (axes ordered by the
        # lowest input index); normalize the computed result to it.
        out = _reorder_axes(out, step.output_axes)

        # Explicit resident-size accounting: everything still live plus the
        # new output, at the contraction instant.
        resident_size = out.size
        for key in live:
            tag, idx = key.split(":", 1)
            idx = int(idx)
            resident_size += (
                loaded[idx].size if tag == "in" else produced[idx].size
            )
        if resident_size != step.resident_total:
            raise AssertionError(
                f"step {step.index}: executor observed {resident_size} "
                f"resident scalars but plan recorded {step.resident_total}"
            )
        observed_peak = max(observed_peak, resident_size)

        # Release the two consumed operands, then publish the new output.
        for ref in (step.first, step.second):
            key = f"{'in' if ref.kind == 'input' else 'step'}:{ref.index}"
            live.discard(key)
        produced[step.index] = out
        live.add(f"step:{step.index}")

    if observed_peak != plan.peak_memory:
        raise AssertionError(
            f"executor peak {observed_peak} != planned peak {plan.peak_memory}"
        )
    result = produced[plan.steps[-1].index]
    return _reorder_axes(result, plan.output_axes), observed_peak

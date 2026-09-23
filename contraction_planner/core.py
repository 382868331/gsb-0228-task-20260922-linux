"""Peak-memory planning for dense tensor contraction networks.

Public API (see module README)::

    Tensor(axes, shape, name=None)
    plan_contraction(tensors, memory_cap, *, node_budget=None) -> ContractionPlan

Standard library only, Python 3.14.  The planner is a pure function: it keeps no
state between calls and performs no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "MAX_TENSORS",
    "Tensor",
    "InputTensor",
    "Resident",
    "OperandRef",
    "PlanStep",
    "PeakEvidence",
    "ContractionPlan",
    "ContractionPlannerError",
    "InvalidProblemError",
    "InfeasiblePlanError",
    "SearchBudgetExhaustedError",
    "plan_contraction",
]

MAX_TENSORS = 10


class ContractionPlannerError(ValueError):
    """Base class for every error raised by this package."""


class InvalidProblemError(ContractionPlannerError):
    """The contraction problem itself violates the documented contract."""


class InfeasiblePlanError(ContractionPlannerError):
    """The search completed and *proved* that no plan satisfies the cap."""


class SearchBudgetExhaustedError(ContractionPlannerError):
    """The search stopped after ``node_budget`` candidates, without a proof."""


@dataclass(frozen=True)
class Tensor:
    """One dense input tensor.

    axes:
        Ordered dimension labels, e.g. ``("i", "j")``.  Labels are strings;
        each label must occur exactly once or twice across all tensors.
    shape:
        Positive integer size for every axis; a label must have the same size
        everywhere it occurs.
    name:
        Optional human-readable name (defaults to ``T0, T1, ...``).
    """

    axes: tuple[str, ...]
    shape: tuple[int, ...]
    name: str | None = None


@dataclass(frozen=True)
class InputTensor:
    """Validated view of one :class:`Tensor` in a solved plan."""

    index: int
    name: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class Resident:
    """One memory-resident tensor in a peak-memory evidence record.

    kind is ``"input"`` (``ref`` is the input index), ``"intermediate"``
    (``ref`` is the 1-based step that produced it) or ``"output"``.
    """

    kind: str
    ref: int | None
    axes: tuple[str, ...]
    size: int


@dataclass(frozen=True)
class OperandRef:
    """Where a contraction step gets one operand.

    kind is ``"input"`` with the input index, or ``"step"`` with the 1-based
    index of the producing step.
    """

    kind: str
    index: int

    def label(self) -> str:
        return f"T{self.index}" if self.kind == "input" else f"step{self.index}"


@dataclass(frozen=True)
class PlanStep:
    """One pairwise contraction in execution order."""

    index: int
    first: OperandRef
    second: OperandRef
    first_axes: tuple[str, ...]
    second_axes: tuple[str, ...]
    contracted_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    output_shape: tuple[int, ...]
    multiplications: int
    resident_total: int
    residents: tuple[Resident, ...]


@dataclass(frozen=True)
class PeakEvidence:
    """The attaining step and exactly what is resident there (in scalars)."""

    step_index: int
    total: int
    residents: tuple[Resident, ...]


@dataclass(frozen=True)
class ContractionPlan:
    """A complete contraction plan.

    tree:
        Nested tuples over input indices: leaves are ``int`` inputs, internal
        nodes are ``(left, right)``.
    bracket:
        The same tree rendered as e.g. ``"((T0,T1),T2)"``.
    steps:
        Pairwise contractions in the chosen evaluation order.
    """

    inputs: tuple[InputTensor, ...]
    output_axes: tuple[str, ...]
    output_shape: tuple[int, ...]
    memory_cap: int
    total_multiplications: int
    peak_memory: int
    peak_evidence: PeakEvidence
    tree: tuple
    bracket: str
    steps: tuple[PlanStep, ...]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _checked_int(param: str, value) -> int:
    """Reject bool explicitly; floats are rejected with NaN/Inf called out."""
    if isinstance(value, bool):
        raise TypeError(f"{param} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"{param} must be finite, got NaN")
        if math.isinf(value):
            raise ValueError(f"{param} must be finite, got infinity")
        raise TypeError(f"{param} must be an integer, got float {value!r}")
    raise TypeError(
        f"{param} must be an integer, got {type(value).__name__}"
    )


def _checked_positive_int(param: str, value) -> int:
    ivalue = _checked_int(param, value)
    if ivalue <= 0:
        raise InvalidProblemError(f"{param} must be >= 1, got {ivalue}")
    return ivalue


def _validate(tensors: Sequence[Tensor], memory_cap, node_budget):
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "tensors must be a list or tuple of Tensor, got "
            f"{type(tensors).__name__}"
        )
    if not 1 <= len(tensors) <= MAX_TENSORS:
        raise InvalidProblemError(
            f"number of tensors must be in 1..{MAX_TENSORS}, got {len(tensors)}"
        )

    axes_list: list[tuple[str, ...]] = []
    shape_list: list[tuple[int, ...]] = []
    names: list[str] = []
    dim_of_label: dict[str, int] = {}
    occurrences: dict[str, list[int]] = {}

    for i, t in enumerate(tensors):
        where = f"tensors[{i}]"
        if not isinstance(t, Tensor):
            raise TypeError(
                f"{where} must be a Tensor, got {type(t).__name__}"
            )
        if not isinstance(t.axes, (list, tuple)) or not isinstance(
            t.shape, (list, tuple)
        ):
            raise TypeError(f"{where}.axes/.shape must be tuples or lists")
        axes = tuple(t.axes)
        shape = tuple(t.shape)
        if len(axes) == 0:
            raise InvalidProblemError(f"{where} must have at least one axis")
        if len(axes) != len(shape):
            raise InvalidProblemError(
                f"{where}: {len(axes)} axes but {len(shape)} sizes"
            )
        seen: set[str] = set()
        for ax, sz in zip(axes, shape):
            if not isinstance(ax, str) or ax == "":
                raise TypeError(f"{where}: axis labels must be non-empty strings")
            if ax in seen:
                raise InvalidProblemError(f"{where}: repeated axis label {ax!r}")
            seen.add(ax)
            size = _checked_positive_int(f"{where} size of axis {ax!r}", sz)
            if ax in dim_of_label:
                if dim_of_label[ax] != size:
                    raise InvalidProblemError(
                        f"axis {ax!r} has inconsistent sizes: "
                        f"{dim_of_label[ax]} vs {size} at {where}"
                    )
            else:
                dim_of_label[ax] = size
            occurrences.setdefault(ax, []).append(i)
        axes_list.append(axes)
        shape_list.append(shape)
        names.append(t.name if isinstance(t.name, str) and t.name else f"T{i}")

    bad = {lab: len(oc) for lab, oc in occurrences.items() if len(oc) > 2}
    if bad:
        detail = ", ".join(f"{lab!r}x{cnt}" for lab, cnt in sorted(bad.items()))
        raise InvalidProblemError(
            f"every axis label must occur 1 or 2 times, got 3+: {detail}"
        )

    cap = _checked_positive_int("memory_cap", memory_cap)
    budget = None
    if node_budget is not None:
        budget = _checked_positive_int("node_budget", node_budget)

    return axes_list, shape_list, names, dim_of_label, occurrences, cap, budget


# --------------------------------------------------------------------------- #
# Subset DP
# --------------------------------------------------------------------------- #


class _Alt:
    """One Pareto alternative for a tensor subset."""

    __slots__ = (
        "mask",
        "leaf",
        "cost",
        "peak",
        "a",
        "b",
        "build_order",
        "mult",
        "contracted",
    )

    def __init__(self, mask, leaf, cost, peak, a=None, b=None,
                 build_order=0, mult=0, contracted=()):
        self.mask = mask
        self.leaf = leaf          # input index for leaves, else None
        self.cost = cost          # total scalar multiplications in subtree
        self.peak = peak          # standalone peak (scalars) for subtree
        self.a = a
        self.b = b
        self.build_order = build_order  # 0: build a then b, 1: b then a
        self.mult = mult
        self.contracted = contracted


def _solve(axes_list, dim_of_label, occurrences, cap, budget):
    """Subset DP.  Returns (dp, used_budget).

    dp[mask] is the cost/peak Pareto frontier for that subset, pruned to
    alternatives whose peak is <= cap.  Raises SearchBudgetExhaustedError when
    the candidate budget is hit.
    """
    n = len(axes_list)
    labels = sorted(dim_of_label)
    label_mask = {
        lab: sum(1 << i for i in occurrences[lab]) for lab in labels
    }

    # Size (scalar count) of the tensor produced by each subset.
    size_of: dict[int, int] = {}
    out_axes_of: dict[int, tuple[str, ...]] = {}
    for mask in range(1, 1 << n):
        prod = 1
        ordered: list[str] = []
        seen: set[str] = set()
        for lab in labels:
            if (label_mask[lab] & mask).bit_count() == 1:
                prod *= dim_of_label[lab]
        for i in range(n):
            if mask & (1 << i):
                for lab in axes_list[i]:
                    if lab not in seen and (
                        label_mask[lab] & mask
                    ).bit_count() == 1:
                        seen.add(lab)
                        ordered.append(lab)
        size_of[mask] = prod
        out_axes_of[mask] = tuple(ordered)

    dp: dict[int, list[_Alt]] = {
        1 << i: [_Alt(1 << i, i, 0, 0)] for i in range(n)
    }
    used = 0

    def prune(cands):
        kept: list[_Alt] = []
        best_peak = math.inf
        seen_pair: set[tuple[int, int]] = set()
        for c in sorted(cands, key=lambda z: (z.cost, z.peak)):
            if c.peak > cap or (c.cost, c.peak) in seen_pair:
                continue
            seen_pair.add((c.cost, c.peak))
            if c.peak < best_peak:
                kept.append(c)
                best_peak = c.peak
        return kept

    for subset_size in range(2, n + 1):
        for mask in range(3, 1 << n):
            if mask.bit_count() != subset_size:
                continue
            pivot = mask & -mask
            rest = mask ^ pivot
            cands: list[_Alt] = []
            a_sub = rest
            while a_sub:
                b_sub = mask ^ a_sub
                fa = dp.get(a_sub, ())
                fb = dp.get(b_sub, ())
                if fa and fb:
                    straddle = tuple(
                        lab
                        for lab in labels
                        if (label_mask[lab] & a_sub)
                        and (label_mask[lab] & b_sub)
                    )
                    # No outer products: the operands must share an axis.
                    if straddle:
                        mult = size_of[mask]
                        for lab in straddle:
                            mult *= dim_of_label[lab]
                        sa, sb, so = (
                            size_of[a_sub],
                            size_of[b_sub],
                            size_of[mask],
                        )
                        hold_a = 0 if a_sub.bit_count() == 1 else sa
                        hold_b = 0 if b_sub.bit_count() == 1 else sb
                        triple = sa + sb + so
                        for x in fa:
                            for y in fb:
                                used += 1
                                if budget is not None and used > budget:
                                    raise SearchBudgetExhaustedError(
                                        f"search budget exhausted after "
                                        f"{used} candidate contractions "
                                        f"(budget={budget}); no optimality "
                                        "proof was reached"
                                    )
                                p_ab = max(x.peak, hold_a + y.peak, triple)
                                p_ba = max(y.peak, hold_b + x.peak, triple)
                                if p_ab <= p_ba:
                                    peak, order = p_ab, 0
                                else:
                                    peak, order = p_ba, 1
                                cands.append(
                                    _Alt(
                                        mask,
                                        None,
                                        x.cost + y.cost + mult,
                                        peak,
                                        x,
                                        y,
                                        order,
                                        mult,
                                        straddle,
                                    )
                                )
                a_sub = (a_sub - 1) & rest
            dp[mask] = prune(cands)

    return dp, used, size_of, out_axes_of


# --------------------------------------------------------------------------- #
# Plan reconstruction
# --------------------------------------------------------------------------- #


def _bracket(alt: _Alt) -> str:
    if alt.leaf is not None:
        return f"T{alt.leaf}"
    return f"({_bracket(alt.a)},{_bracket(alt.b)})"


def _tree(alt: _Alt):
    if alt.leaf is not None:
        return alt.leaf
    return (_tree(alt.a), _tree(alt.b))


def _build_plan(alt, axes_list, shape_list, names, dim_of_label, occurrences,
                size_of, out_axes_of, cap) -> ContractionPlan:
    n = len(axes_list)
    full = (1 << n) - 1

    inputs = tuple(
        InputTensor(
            index=i,
            name=names[i],
            axes=axes_list[i],
            shape=shape_list[i],
            size=math.prod(shape_list[i]),
        )
        for i in range(n)
    )

    # Degenerate case: one tensor, nothing to contract.
    if alt.leaf is not None:
        only = inputs[alt.leaf]
        resident = Resident("input", alt.leaf, only.axes, only.size)
        evidence = PeakEvidence(-1, only.size, (resident,))
        return ContractionPlan(
            inputs=inputs,
            output_axes=only.axes,
            output_shape=only.shape,
            memory_cap=cap,
            total_multiplications=0,
            peak_memory=only.size,
            peak_evidence=evidence,
            tree=alt.leaf,
            bracket=only.name,
            steps=(),
        )

    steps: list[PlanStep] = []

    def resident_for(ref: OperandRef, mask: int) -> Resident:
        return Resident(
            ref.kind if ref.kind == "input" else "intermediate",
            ref.index,
            out_axes_of[mask],
            size_of[mask],
        )

    def emit(node: _Alt, held: tuple[Resident, ...]) -> int:
        """Emit subtree; returns the step id of its root result."""
        first, second = (node.a, node.b) if node.build_order == 0 else (
            node.b,
            node.a,
        )
        first_mask, second_mask = first.mask, second.mask

        # Build the first operand; then (if it is a temporary) hold its result
        # while the second operand is built.  Leaves are loaded on demand.
        first_id = emit(first, held) if first.leaf is None else None
        first_ref = (
            OperandRef("input", first.leaf)
            if first.leaf is not None
            else OperandRef("step", first_id)
        )
        held_for_second = held
        if first.leaf is None:
            held_for_second = held + (
                Resident("intermediate", first_id,
                         out_axes_of[first_mask], size_of[first_mask]),
            )
        second_id = emit(second, held_for_second) if second.leaf is None else None
        second_ref = (
            OperandRef("input", second.leaf)
            if second.leaf is not None
            else OperandRef("step", second_id)
        )

        step_id = len(steps) + 1
        out_axes = out_axes_of[node.mask]
        # At the contraction instant both operands and the new output coexist,
        # alongside any ancestor temporaries still held.
        residents = held + (
            resident_for(first_ref, first_mask),
            resident_for(second_ref, second_mask),
            Resident("output", step_id, out_axes, size_of[node.mask]),
        )
        total = sum(r.size for r in residents)

        # Contracted axes are reported in canonical a-operand axis order.
        contracted = tuple(
            ax for ax in out_axes_of[node.a.mask] if ax in node.contracted
        )
        steps.append(
            PlanStep(
                index=step_id,
                first=first_ref,
                second=second_ref,
                first_axes=out_axes_of[first_mask],
                second_axes=out_axes_of[second_mask],
                contracted_axes=contracted,
                output_axes=out_axes,
                output_shape=tuple(dim_of_label[lab] for lab in out_axes),
                multiplications=node.mult,
                resident_total=total,
                residents=residents,
            )
        )
        return step_id

    emit(alt, ())

    peak_step = min(steps, key=lambda s: (-s.resident_total, s.index))
    evidence = PeakEvidence(
        peak_step.index, peak_step.resident_total, peak_step.residents
    )

    output_axes = out_axes_of[full]
    return ContractionPlan(
        inputs=inputs,
        output_axes=output_axes,
        output_shape=tuple(dim_of_label[lab] for lab in output_axes),
        memory_cap=cap,
        total_multiplications=sum(s.multiplications for s in steps),
        peak_memory=max(s.resident_total for s in steps),
        peak_evidence=evidence,
        tree=_tree(alt),
        bracket=_bracket(alt),
        steps=tuple(steps),
    )


def plan_contraction(
    tensors: Sequence[Tensor],
    memory_cap: int,
    *,
    node_budget: int | None = None,
) -> ContractionPlan:
    """Plan a pairwise contraction network minimizing scalar multiplications.

    See the module README for the full contract.  Briefly:

    * each label occurs once (output axis) or twice (contraction axis);
    * only binary contractions with at least one shared axis are allowed
      (no outer products);
    * peak memory at a step counts both operands plus the new output, with
      consumed temporaries released afterwards and inputs loaded on demand;
    * the returned plan has minimum total scalar multiplications among all
      plans whose peak is <= ``memory_cap``;

    Raises :class:`InvalidProblemError`/:class:`TypeError` for bad inputs,
    :class:`InfeasiblePlanError` when no plan can meet the cap (proved by
    exhaustive subset DP), and :class:`SearchBudgetExhaustedError` when
    ``node_budget`` candidate contractions were examined first.
    """
    (
        axes_list,
        shape_list,
        names,
        dim_of_label,
        occurrences,
        cap,
        budget,
    ) = _validate(tensors, memory_cap, node_budget)
    n = len(axes_list)
    full = (1 << n) - 1

    if n == 1:
        only_size = math.prod(shape_list[0])
        if only_size > cap:
            raise InfeasiblePlanError(
                f"no plan fits memory_cap={cap}: the single input already "
                f"occupies {only_size} scalars"
            )

    dp, _used, size_of, out_axes_of = _solve(
        axes_list, dim_of_label, occurrences, cap, budget
    )

    frontier = dp[full]
    if not frontier:
        # Distinguish "proved impossible" from budget cutoff: the cap-filtered
        # search completed, so rerun unconstrained to characterize the bound.
        dp_free, _u2, _sf, _of = _solve(
            axes_list, dim_of_label, occurrences, math.inf, None
        )
        if not dp_free[full]:
            raise InfeasiblePlanError(
                "no outer-product-free contraction tree exists for this "
                "network (some tensors share no contraction path)"
            )
        best_peak = min(a.peak for a in dp_free[full])
        raise InfeasiblePlanError(
            f"no plan fits memory_cap={cap}: the provable minimum peak is "
            f"{best_peak} scalars (minimum-peak plan costs "
            f"{min(a.cost for a in dp_free[full] if a.peak == best_peak)} "
            "multiplications)"
        )

    # Frontier is cost-sorted; among equal cost take the smallest peak.
    best = min(frontier, key=lambda a: (a.cost, a.peak))
    return _build_plan(
        best,
        axes_list,
        shape_list,
        names,
        dim_of_label,
        occurrences,
        size_of,
        out_axes_of,
        cap,
    )

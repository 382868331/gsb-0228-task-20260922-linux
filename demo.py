#!/usr/bin/env python3
"""Fixed-input demonstration of the contraction peak-memory planner.

Run with::

    python demo.py

It really plans and numerically executes a small contraction network, then
exercises one rejection boundary.  Everything is deterministic and offline;
the run takes well under a second of compute.
"""

from __future__ import annotations

import time

from contraction_planner import (
    DenseArray,
    InfeasiblePlanError,
    plan_contraction,
    execute_plan,
)
from contraction_planner.sample_data import matrix_chain, tradeoff_network


def hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def show_plan(plan) -> None:
    print(f"bracket tree        : {plan.bracket}")
    print(f"nested tuple tree   : {plan.tree}")
    print(f"output axes / shape : {plan.output_axes} / {plan.output_shape}")
    print(f"total multiply count: {plan.total_multiplications}")
    print(f"peak resident memory: {plan.peak_memory} scalars "
          f"(cap = {plan.memory_cap})")
    print("steps:")
    for s in plan.steps:
        held = ", ".join(
            f"{r.kind}:{r.ref if r.kind != 'output' else s.index}={r.size}"
            for r in s.residents
        )
        print(
            f"  {s.index}. {s.first.label()} {s.first_axes} x "
            f"{s.second.label()} {s.second_axes}"
        )
        print(
            f"     contract {s.contracted_axes} -> {s.output_axes} "
            f"{s.output_shape}; mults={s.multiplications}; "
            f"resident={s.resident_total} [{held}]"
        )
    ev = plan.peak_evidence
    print(
        f"peak evidence @ step {ev.step_index}: "
        + " + ".join(f"{r.kind}({r.size})" for r in ev.residents)
        + f" = {ev.total}"
    )


def main() -> None:
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Normal result: (2x3) @ (3x4) @ (4x5), plan then execute exactly.
    # ------------------------------------------------------------------ #
    hr("1. NORMAL RESULT  -  matrix chain A(2x3) B(3x4) C(4x5)")
    net = matrix_chain()
    plan = plan_contraction(net, memory_cap=1_000_000)
    show_plan(plan)

    # Deterministic integer data generated in this file.
    arrays = [
        DenseArray.from_function(
            t.axes, t.shape, lambda idx: (idx[0] + 1) * 7 + idx[1] * 3 + 1
        )
        for t in net
    ]
    result, observed_peak = execute_plan(plan, arrays)
    print(f"\nexecuted plan numerically; executor-observed peak = "
          f"{observed_peak} (matches plan: {observed_peak == plan.peak_memory})")

    # Independent check computed right here by direct nested loops (not a
    # value baked into the source): Y[i,j] = sum_{k,l} A[i,k] B[k,l] C[l,j].
    A, B, C = arrays
    checked = 0
    for i in range(2):
        for j in range(5):
            expected = sum(
                A.get({"i": i, "k": k})
                * B.get({"k": k, "l": l})
                * C.get({"l": l, "j": j})
                for k in range(3)
                for l in range(4)
            )
            got = result.get({"i": i, "j": j})
            assert got == expected, (i, j, got, expected)
            checked += 1
    print(f"cross-checked {checked} output entries against direct "
          "k,l-summation: all exact integer matches")
    print("result row i=0:",
          [result.get({"i": 0, "j": j}) for j in range(5)])

    # ------------------------------------------------------------------ #
    # 2. Cost / memory trade-off on a 4-tensor network.
    # ------------------------------------------------------------------ #
    hr("2. COST / MEMORY TRADE-OFF  -  4 tensors, shared-axis network")
    tn = tradeoff_network()
    cheap = plan_contraction(tn, memory_cap=10**18)
    tight = plan_contraction(tn, memory_cap=202)
    print(f"cap = inf : tree {cheap.bracket}")
    print(f"            {cheap.total_multiplications} multiplications, "
          f"peak {cheap.peak_memory}")
    print(f"cap = 202 : tree {tight.bracket}")
    print(f"            {tight.total_multiplications} multiplications, "
          f"peak {tight.peak_memory}")
    print("=> tightening the cap by 1 scalar buys 21 fewer peak scalars at")
    print("   the price of 42 extra multiplications; the DP picks the")
    print("   cheapest tree that still fits.")

    # ------------------------------------------------------------------ #
    # 3. Actually triggered rejection boundary: infeasible memory cap.
    # ------------------------------------------------------------------ #
    hr("3. REJECTION BOUNDARY  -  cap below the provable minimum peak")
    impossible_cap = cheap.peak_memory - 100
    print(f"requesting memory_cap={impossible_cap} "
          f"(minimum provable peak is 182 for this network)...")
    try:
        plan_contraction(tn, memory_cap=impossible_cap)
    except InfeasiblePlanError as exc:
        print(f"caught {type(exc).__name__}:")
        print(f"  {exc}")
        print("the subset DP exhaustively proved impossibility; this is")
        print("distinct from a budget cutoff (SearchBudgetExhaustedError).")
    else:
        raise AssertionError("expected InfeasiblePlanError was not raised")

    elapsed = time.perf_counter() - t_start
    hr(f"done in {elapsed:.3f}s  -  standard library only, no network use")


if __name__ == "__main__":
    main()

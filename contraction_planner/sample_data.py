"""Deterministic in-repo sample networks.

No external data, no randomness sourced outside this module: every generator
takes an explicit integer seed and uses :mod:`random.Random`.
"""

from __future__ import annotations

import random
import string

from .core import Tensor

__all__ = [
    "matrix_chain",
    "tradeoff_network",
    "random_connected_network",
]


def matrix_chain() -> list[Tensor]:
    """A small (2x3) @ (3x4) @ (4x5) chain, output matrix 2x5."""
    return [
        Tensor(("i", "k"), (2, 3), name="A"),
        Tensor(("k", "l"), (3, 4), name="B"),
        Tensor(("l", "j"), (4, 5), name="C"),
    ]


def tradeoff_network() -> list[Tensor]:
    """4 tensors where cheapest tree is not the lowest-peak tree.

    Labels mapped from a hand-verified instance (seed 95 of the test
    generator).  Unconstrained minimum cost is 770 multiplications at peak
    203 scalars; the minimum peak is 182 scalars at cost 812.
    """
    # original: axes [(5,2,8),(6,5),(9,6),(8,2,1)]
    #           dims {5:7,2:2,8:7,6:4,9:7,1:5}
    return [
        Tensor(("a", "b", "c"), (7, 2, 7), name="N0"),
        Tensor(("d", "a"), (4, 7), name="N1"),
        Tensor(("e", "d"), (7, 4), name="N2"),
        Tensor(("c", "b", "f"), (7, 2, 5), name="N3"),
    ]


def random_connected_network(
    n: int, seed: int, *, max_dim: int = 6, extra_axes: int | None = None
) -> list[Tensor]:
    """Build a connected contraction network of ``n`` tensors.

    Connectivity is guaranteed by a chain of pairwise "bridge" labels
    (tensor i and i+1 always share one).  Each tensor also gets one private
    output label and a seeded number of extra labels, each either private or
    shared with one other tensor.  Every label ends up occurring exactly once
    or twice.
    """
    if not 1 <= n <= 10:
        raise ValueError("n must be in 1..10")
    rng = random.Random(seed)
    names = list(string.ascii_lowercase)
    dims: dict[str, int] = {}

    def fresh() -> str:
        k = len(dims)
        lab = names[k] if k < len(names) else f"z{k}"
        dims[lab] = rng.randint(2, max_dim)
        return lab

    axes: list[list[str]] = [[] for _ in range(n)]
    # One private output label per tensor.
    for i in range(n):
        axes[i].append(fresh())
    # Bridge labels along a spanning path guarantee connectivity.
    for i in range(n - 1):
        lab = fresh()
        axes[i].append(lab)
        axes[i + 1].append(lab)
    # Extra labels, kept to occurrence <= 2.
    tries = 0
    target = n if extra_axes is None else extra_axes
    placed = 0
    while placed < target and tries < target * 10:
        tries += 1
        i = rng.randrange(n)
        if rng.random() < 0.5:
            lab = fresh()
            axes[i].append(lab)
            placed += 1
        else:
            j = rng.randrange(n)
            if j == i:
                continue
            lab = fresh()
            axes[i].append(lab)
            axes[j].append(lab)
            placed += 1
    return [Tensor(tuple(a), tuple(dims[x] for x in a)) for a in axes]

"""tensor_peak — 张量收缩的峰值内存规划。

只读入稠密张量的维度标签与每维大小（标量元素个数即内存单位），用子集
DP + (标量乘法数, 峰值驻留) Pareto 前沿，在两两收缩、禁止外积的前提下，
求满足内存上限的最少标量乘法收缩计划，并输出括号树、逐步形状与峰值证据。

旧接口只做精确整数内存/成本计算；新接口用 Decimal 传播保守浮点误差界。
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, getcontext
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "MAX_TENSORS",
    "TensorSpec",
    "OperandRef",
    "ContractStep",
    "ContractionPlan",
    "FrontPoint",
    "TensorPlanningError",
    "InvalidProblemError",
    "InfeasibleError",
    "DisconnectedNetworkError",
    "MemoryLimitInfeasibleError",
    "BudgetExhaustedError",
    "PrecisionSpec",
    "PrecisionContractStep",
    "PrecisionPlan",
    "ConstraintsInfeasibleError",
    "DEFAULT_PRECISION_LEVELS",
    "find_plan",
    "find_precision_plan",
    "pareto_front",
    "format_tree",
]

MAX_TENSORS = 10


# ---------------------------------------------------------------------------
# 错误语义
# ---------------------------------------------------------------------------


class TensorPlanningError(Exception):
    """本模块所有异常的公共基类。"""


class InvalidProblemError(TensorPlanningError, ValueError):
    """输入不满足契约（可定位的非法输入），不做任何静默纠正。"""


class InfeasibleError(TensorPlanningError):
    """已（通过穷举性搜索）证明不存在满足约束的计划。"""


class DisconnectedNetworkError(InfeasibleError):
    """收缩轴构成的张量网络不连通：禁止外积时任何收缩树都不存在。"""


class MemoryLimitInfeasibleError(InfeasibleError):
    """收缩树存在，但已证明没有任何计划的峰值内存满足给定上限。"""


class BudgetExhaustedError(TensorPlanningError):
    """搜索预算耗尽，搜索停止；此时尚未证明无解，须与 InfeasibleError 区分。"""


class ConstraintsInfeasibleError(InfeasibleError):
    """已穷举顺序与精度组合，证明内存或最终误差约束无解。"""


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """一个原始稠密张量的静态描述。

    name:  唯一张量名（非空 str）。
    axes:  维度标签元组（非空 str，同一张量内不得重复）。
    shape: 与 axes 等长的正整数维大小。
    """

    name: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]

    def __init__(self, name, axes, shape):
        if not isinstance(name, str) or not name:
            raise InvalidProblemError(
                "TensorSpec.name 必须是非空 str，得到 %r" % (name,)
            )
        if isinstance(axes, str) or not isinstance(axes, Sequence):
            raise InvalidProblemError(
                "张量 %r 的 axes 必须是标签序列" % (name,)
            )
        if isinstance(shape, str) or not isinstance(shape, Sequence):
            raise InvalidProblemError(
                "张量 %r 的 shape 必须是整数序列" % (name,)
            )
        axes = tuple(axes)
        shape = tuple(shape)
        for p, label in enumerate(axes):
            if not isinstance(label, str) or not label:
                raise InvalidProblemError(
                    "张量 %r 的第 %d 个轴标签必须是非空 str，得到 %r"
                    % (name, p, label)
                )
        if len(set(axes)) != len(axes):
            raise InvalidProblemError("张量 %r 的轴标签重复: %r" % (name, axes))
        for p, d in enumerate(shape):
            _check_dimension(d, "张量 %r 的第 %d 维(%s)大小" % (name, p, axes[p]))
        if len(axes) != len(shape):
            raise InvalidProblemError(
                "张量 %r 的 axes/shape 长度不一致: %d != %d"
                % (name, len(axes), len(shape))
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shape", shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class OperandRef:
    """收缩步骤中一个操作数的描述（原始输入或中间结果）。"""

    kind: str  # 'input' | 'intermediate'
    name: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    size: int
    defining_step: Optional[int]  # 原始输入为 None，否则为产出它的步骤下标


@dataclass(frozen=True)
class ContractStep:
    """一次两两收缩。

    lhs/rhs 按括号树的左右孩子给出（与求值先后无关）；evaluated_first
    指出为了实现计划峰值，先求值哪一侧。residency 给出该步执行瞬间的
    完整驻留清单（后台保留的中间量、两个操作数、新输出），peak_at_step
    是其标量元素总数；整个计划的 peak_memory = max(peak_at_step)。
    """

    index: int
    output_name: str
    lhs: OperandRef
    rhs: OperandRef
    evaluated_first: str  # 'lhs' | 'rhs'
    contracted_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    output_shape: tuple[int, ...]
    mult_count: int
    residency: tuple[tuple[str, int], ...]
    peak_at_step: int


@dataclass(frozen=True)
class ContractionPlan:
    """完整收缩计划。"""

    tensors: tuple[TensorSpec, ...]
    memory_limit: Optional[int]
    output_axes: tuple[str, ...]
    output_shape: tuple[int, ...]
    total_mult_count: int
    peak_memory: int
    peak_step: Optional[int]
    steps: tuple[ContractStep, ...]
    tree: tuple
    tree_text: str


@dataclass(frozen=True)
class FrontPoint:
    """Pareto 前沿上的一个点：mult_count 更少与 peak_memory 更低不可兼得。"""

    mult_count: int
    peak_memory: int
    tree: tuple


@dataclass(frozen=True)
class PrecisionSpec:
    """一档模拟浮点精度及其每个标量乘法的抽象成本。"""

    name: str
    unit_roundoff: Decimal
    cost_per_mult: int

    def __init__(self, name, unit_roundoff, cost_per_mult):
        if not isinstance(name, str) or not name:
            raise InvalidProblemError("精度名称必须是非空 str")
        u = _decimal_number(unit_roundoff, "精度 %s 的 unit_roundoff" % name)
        if not (Decimal(0) < u < Decimal(1)):
            raise InvalidProblemError(
                "精度 %s 的 unit_roundoff 必须位于 (0,1)，得到 %s"
                % (name, u)
            )
        if isinstance(cost_per_mult, bool) or not isinstance(cost_per_mult, int):
            raise InvalidProblemError("精度成本必须是正整数")
        if cost_per_mult < 1:
            raise InvalidProblemError("精度成本必须 >= 1")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit_roundoff", u)
        object.__setattr__(self, "cost_per_mult", cost_per_mult)


@dataclass(frozen=True)
class PrecisionContractStep(ContractStep):
    """带精度与误差账本的一次收缩。"""

    precision: PrecisionSpec
    contraction_length: int
    gamma_n: Decimal
    lhs_magnitude: Decimal
    rhs_magnitude: Decimal
    nominal_magnitude: Decimal
    input_error_bound: Decimal
    rounding_error_bound: Decimal
    output_error_bound: Decimal
    output_magnitude_bound: Decimal
    step_cost: int


@dataclass(frozen=True)
class PrecisionPlan(ContractionPlan):
    """同时选择括号顺序、求值顺序和逐步精度的完整计划。"""

    total_cost: int = 0
    error_limit: Optional[Decimal] = None
    final_error_bound: Optional[Decimal] = None
    value_bounds: Optional[dict[str, Decimal]] = None
    input_error_bounds: Optional[dict[str, Decimal]] = None
    precision_levels: tuple[PrecisionSpec, ...] = ()


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _check_dimension(value, where: str) -> None:
    if isinstance(value, bool):
        raise InvalidProblemError("%s 不得为 bool 类型" % where)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidProblemError("%s 不得为 NaN/Infinity" % where)
        raise InvalidProblemError("%s 必须是精确正整数，得到 float %r" % (where, value))
    if not isinstance(value, int):
        raise InvalidProblemError("%s 必须是正整数，得到 %r" % (where, value))
    if value < 1:
        raise InvalidProblemError("%s 必须 >= 1，得到 %d" % (where, value))


def _check_positive_int(value, what: str, *, allow_none: bool = False):
    if allow_none and value is None:
        return None
    if isinstance(value, bool):
        raise InvalidProblemError("%s 不得为 bool 类型" % what)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidProblemError("%s 不得为 NaN/Infinity" % what)
        raise InvalidProblemError("%s 必须是正整数，得到 float %r" % (what, value))
    if not isinstance(value, int):
        raise InvalidProblemError("%s 必须是正整数，得到 %r" % (what, value))
    if value < 1:
        raise InvalidProblemError("%s 必须 >= 1，得到 %d" % (what, value))
    return value


def _decimal_number(value, what: str) -> Decimal:
    if isinstance(value, bool):
        raise InvalidProblemError("%s 不得为 bool" % what)
    if not isinstance(value, (int, Decimal, str)):
        raise InvalidProblemError("%s 必须是精确数字或 Decimal 字符串" % what)
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise InvalidProblemError("%s 不是有效 Decimal 数字" % what)
    if not result.is_finite():
        raise InvalidProblemError("%s 必须有限，得到 %s" % (what, result))
    return result


def _nonnegative_decimal(value, what: str) -> Decimal:
    result = _decimal_number(value, what)
    if result < 0:
        raise InvalidProblemError("%s 必须 >= 0，得到 %s" % (what, result))
    return result


def _positive_decimal(value, what: str) -> Decimal:
    result = _decimal_number(value, what)
    if result <= 0:
        raise InvalidProblemError("%s 必须 > 0，得到 %s" % (what, result))
    return result


DEFAULT_PRECISION_LEVELS = (
    PrecisionSpec("low", Decimal(1) / Decimal(1000), 1),
    PrecisionSpec("high", Decimal(1) / Decimal(1_000_000), 8),
)


def _validate_precision_args(
    specs,
    memory_limit,
    error_limit,
    value_bounds,
    input_error_bounds,
    precision_levels,
    search_budget,
):
    validated, label_size, label_count, limit, budget = _validate(
        specs, memory_limit, search_budget
    )
    err_limit = _positive_decimal(error_limit, "error_limit")
    if not isinstance(value_bounds, dict):
        raise InvalidProblemError("value_bounds 必须是 dict[str, 数值]")
    names = {t.name for t in validated}
    if set(value_bounds) != names:
        raise InvalidProblemError(
            "value_bounds 必须恰好给出每个输入的元素绝对值上界"
        )
    bounds = {
        name: _nonnegative_decimal(
            value_bounds[name], "输入 %r 的 value_bound" % name
        )
        for name in value_bounds
    }
    zero_bounds = {name: Decimal(0) for name in names}
    errors = dict(zero_bounds)
    if input_error_bounds is not None:
        if not isinstance(input_error_bounds, dict):
            raise InvalidProblemError("input_error_bounds 必须是 dict")
        bad = set(input_error_bounds) - names
        if bad:
            raise InvalidProblemError("input_error_bounds 含未知输入 %r" % sorted(bad))
        for name, value in input_error_bounds.items():
            errors[name] = _nonnegative_decimal(
                value, "输入 %r 的已有误差界" % name
            )

    levels = tuple(precision_levels or DEFAULT_PRECISION_LEVELS)
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise InvalidProblemError("precision_levels 必须是 PrecisionSpec 序列")
    if len(levels) != 2 or any(not isinstance(p, PrecisionSpec) for p in levels):
        raise InvalidProblemError("precision_levels 必须恰好包含两档 PrecisionSpec")
    if len({p.name for p in levels}) != 2:
        raise InvalidProblemError("两档精度名称必须不同")
    return (
        validated, label_size, label_count, limit, err_limit,
        bounds, errors, tuple(levels), budget,
    )


def _validate(tensors, memory_limit, search_budget):
    if isinstance(tensors, (str, bytes)) or not isinstance(tensors, Sequence):
        raise InvalidProblemError("tensors 必须是 TensorSpec 序列")
    if len(tensors) == 0:
        raise InvalidProblemError("tensors 至少包含 1 个张量")
    if len(tensors) > MAX_TENSORS:
        raise InvalidProblemError(
            "张量数 %d 超过接口上限 %d" % (len(tensors), MAX_TENSORS)
        )
    specs: list[TensorSpec] = []
    seen_names: set[str] = set()
    for i, t in enumerate(tensors):
        if not isinstance(t, TensorSpec):
            raise InvalidProblemError(
                "tensors[%d] 必须是 TensorSpec，得到 %r" % (i, type(t).__name__)
            )
        if t.name in seen_names:
            raise InvalidProblemError("张量名重复: %r" % t.name)
        seen_names.add(t.name)
        specs.append(t)

    reserved = {("T%d" % i) for i in range(len(specs))}
    clash = seen_names & reserved
    if clash:
        raise InvalidProblemError(
            "输入张量名 %r 与中间量保留名 T0..T%d 冲突"
            % (sorted(clash), len(specs) - 1)
        )

    # 标签出现次数 1 或 2；同名标签维大小必须一致。
    label_size: dict[str, int] = {}
    label_count: dict[str, int] = {}
    for t in specs:
        for label, d in zip(t.axes, t.shape):
            if label in label_size:
                if label_size[label] != d:
                    raise InvalidProblemError(
                        "收缩轴 %r 的维大小冲突: %d != %d（出现于张量 %r）"
                        % (label, label_size[label], d, t.name)
                    )
                label_count[label] += 1
            else:
                label_size[label] = d
                label_count[label] = 1
    bad = [lbl for lbl, c in label_count.items() if c > 2]
    if bad:
        raise InvalidProblemError(
            "标签 %r 出现 %s 次；契约要求每个标签恰出现 1 或 2 次"
            % (bad, "、".join(str(label_count[b]) for b in bad))
        )

    limit = _check_positive_int(
        memory_limit, "memory_limit", allow_none=False
    )
    budget = _check_positive_int(
        search_budget, "search_budget", allow_none=True
    )
    return tuple(specs), label_size, label_count, limit, budget


def _check_connected(specs, label_count) -> None:
    """以张量为节点、收缩轴（出现 2 次的标签）为边，检查网络连通性。"""
    n = len(specs)
    if n == 1:
        return
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owners: dict[str, int] = {}
    for i, t in enumerate(specs):
        for label in t.axes:
            if label_count[label] == 2:
                if label in owners:
                    union(owners[label], i)
                else:
                    owners[label] = i
    root = find(0)
    if any(find(i) != root for i in range(1, n)):
        groups: dict[int, list[str]] = {}
        for i, t in enumerate(specs):
            groups.setdefault(find(i), []).append(t.name)
        detail = "；".join("{" + ", ".join(g) + "}" for g in groups.values())
        raise DisconnectedNetworkError(
            "收缩网络不连通（%s）：禁止外积时不存在两两收缩计划" % detail
        )


# ---------------------------------------------------------------------------
# 子集 DP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    muls: int
    peak: int
    tree: tuple
    order: int  # 0=先求左子集，1=先求右子集


def _prune(entries, limit):
    """按 (muls, peak) 双目标保留 Pareto 前沿，并剔除 peak > limit。"""
    uniq = {(e.muls, e.peak): e for e in entries}
    ordered = sorted(uniq.values(), key=lambda e: (e.muls, e.peak))
    kept: list[_Entry] = []
    best_peak = None
    for e in ordered:
        if limit is not None and e.peak > limit:
            continue
        if best_peak is None or e.peak < best_peak:
            kept.append(e)
            best_peak = e.peak
    return kept


def _solve(specs, label_size, label_count, limit, budget):
    n = len(specs)
    N = 1 << n

    labels_order: list[str] = []
    for t in specs:
        for lbl in t.axes:
            if lbl not in labels_order:
                labels_order.append(lbl)
    index_of = {lbl: i for i, lbl in enumerate(labels_order)}

    axes_of = [None] * N
    size_of = [0] * N
    for mask in range(1, N):
        counts = [0] * len(labels_order)
        members = mask
        while members:
            b = members & -members
            j = b.bit_length() - 1
            for lbl in specs[j].axes:
                counts[index_of[lbl]] += 1
            members ^= b
        ax = tuple(lbl for lbl, c in zip(labels_order, counts) if c == 1)
        axes_of[mask] = ax
        size_of[mask] = math.prod(label_size[a] for a in ax)

    remaining = budget  # None 表示不限
    fronts: dict[int, list[_Entry]] = {}
    for i, t in enumerate(specs):
        fronts[1 << i] = [_Entry(0, 0, ("leaf", t.name), 0)]

    # 子集按基数递增处理。
    for mask in range(1, N):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        sub = (mask - 1) & mask
        collected: list[_Entry] = []
        while sub:
            other = mask ^ sub
            # 每个无序划分只处理一次：含最低位的一侧作为“左”。
            if (sub & low) and sub < mask:
                ax_l = axes_of[sub]
                ax_r = axes_of[other]
                shared = set(ax_l) & set(ax_r)
                if shared:
                    fl = fronts.get(sub)
                    fr = fronts.get(other)
                    if fl and fr:
                        set_l, set_r = set(ax_l), set(ax_r)
                        shared = set_l & set_r
                        union = set_l | set_r
                        out_axes = tuple(
                            a for a in labels_order if a in union and a not in shared
                        )
                        s_out = math.prod(label_size[a] for a in out_axes)
                        node_muls = math.prod(label_size[a] for a in union)
                        s_l, s_r = size_of[sub], size_of[other]
                        triple = s_l + s_r + s_out
                        contracted = tuple(a for a in labels_order if a in shared)
                        for el in fl:
                            for er in fr:
                                if remaining is not None:
                                    if remaining == 0:
                                        raise BudgetExhaustedError(
                                            "搜索预算（候选收缩求值次数）耗尽，"
                                            "搜索停止；未证明无解，请增大 search_budget"
                                        )
                                    remaining -= 1
                                # 先求左：先单独求 L（峰值 el.peak），保留其结果
                                # s_l，再求 R（其内部峰值整体抬高 s_l），最后收缩。
                                peak_l_first = max(
                                    el.peak, s_l + er.peak, triple
                                )
                                peak_r_first = max(
                                    er.peak, s_r + el.peak, triple
                                )
                                if peak_l_first <= peak_r_first:
                                    peak, order = peak_l_first, 0
                                else:
                                    peak, order = peak_r_first, 1
                                tree = (
                                    "node",
                                    out_axes,
                                    el.tree,
                                    er.tree,
                                    contracted,
                                    node_muls,
                                    order,
                                )
                                collected.append(
                                    _Entry(
                                        el.muls + er.muls + node_muls,
                                        peak,
                                        tree,
                                        order,
                                    )
                                )
            sub = (sub - 1) & mask

        front = _prune(collected, limit)
        if front:
            fronts[mask] = front
        # 该子集在给定 limit 下无可行计划：不放入 fronts；
        # 这是精确剪枝（父集峰值不会低于子集峰值），不影响最终“证明无解”。

    full = N - 1
    front = fronts.get(full)
    if not front:
        # 能走到这里说明连通性已通过、预算未耗尽，空缺即经过完整穷举的证明。
        raise MemoryLimitInfeasibleError(
            "已穷举全部合法收缩树：不存在峰值内存 <= %d 标量元素的计划"
            % limit
        )
    best = min(front, key=lambda e: (e.muls, e.peak))
    return best, fronts, axes_of, size_of, labels_order


@dataclass(frozen=True)
class _PrecisionEntry:
    cost: int
    peak: int
    nominal: Decimal
    error: Decimal
    tree: tuple
    order: int


def _prune_precision(entries, limit):
    ordered = sorted(
        entries,
        key=lambda e: (e.cost, e.peak, e.error),
    )
    kept = []
    for candidate in ordered:
        if candidate.peak > limit:
            continue
        dominated = False
        for prior in kept:
            if (
                prior.cost <= candidate.cost
                and prior.peak <= candidate.peak
                and prior.error <= candidate.error
            ):
                dominated = True
                break
        if not dominated:
            kept = [
                p for p in kept
                if not (
                    candidate.cost <= p.cost
                    and candidate.peak <= p.peak
                    and candidate.error <= p.error
                )
            ]
            kept.append(candidate)
    return kept


def _axis_tables(specs, label_size):
    n = len(specs)
    N = 1 << n
    labels_order = []
    for t in specs:
        for label in t.axes:
            if label not in labels_order:
                labels_order.append(label)
    index_of = {label: i for i, label in enumerate(labels_order)}
    axes_of = [None] * N
    size_of = [0] * N
    for mask in range(1, N):
        counts = [0] * len(labels_order)
        members = mask
        while members:
            bit = members & -members
            i = bit.bit_length() - 1
            for label in specs[i].axes:
                counts[index_of[label]] += 1
            members ^= bit
        axes = tuple(
            label
            for label, count in zip(labels_order, counts)
            if count == 1
        )
        axes_of[mask] = axes
        size_of[mask] = math.prod(label_size[label] for label in axes)
    return labels_order, axes_of, size_of


def _solve_precision(
    specs,
    label_size,
    limit,
    error_limit,
    value_bounds,
    input_errors,
    levels,
    budget,
):
    getcontext().prec = 80
    n = len(specs)
    N = 1 << n
    labels_order, axes_of, size_of = _axis_tables(specs, label_size)
    remaining = budget
    fronts = {}
    for i, t in enumerate(specs):
        mask = 1 << i
        leaf = ("leaf", t.name)
        fronts[mask] = [
            _PrecisionEntry(
                0,
                0,
                value_bounds[t.name],
                input_errors[t.name],
                leaf,
                0,
            )
        ]

    for mask in range(1, N):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        sub = (mask - 1) & mask
        collected = []
        while sub:
            other = mask ^ sub
            if sub & low:
                left_axes = axes_of[sub]
                right_axes = axes_of[other]
                shared = set(left_axes) & set(right_axes)
                if shared and fronts.get(sub) and fronts.get(other):
                    union = set(left_axes) | set(right_axes)
                    out_axes = tuple(
                        a for a in labels_order if a in union and a not in shared
                    )
                    contracted = tuple(a for a in labels_order if a in shared)
                    contraction_length = math.prod(label_size[a] for a in shared)
                    node_muls = math.prod(label_size[a] for a in union)
                    left_size = size_of[sub]
                    right_size = size_of[other]
                    out_size = size_of[mask]
                    triple = left_size + right_size + out_size

                    for left in fronts[sub]:
                        for right in fronts[other]:
                            for precision in levels:
                                if remaining is not None:
                                    if remaining == 0:
                                        raise BudgetExhaustedError(
                                            "搜索预算（顺序×精度候选求值次数）耗尽，"
                                            "搜索停止；未证明无解，请增大 search_budget"
                                        )
                                    remaining -= 1

                                k_dec = Decimal(contraction_length)
                                ku = k_dec * precision.unit_roundoff
                                if ku >= 1:
                                    continue
                                gamma = ku / (Decimal(1) - ku)
                                a = left.nominal
                                b = right.nominal
                                ea = left.error
                                eb = right.error
                                nominal = k_dec * a * b
                                existing_error = k_dec * (a * eb + b * ea + ea * eb)
                                rounded_operand_sum = (a + ea) * (b + eb)
                                rounding_error = (
                                    gamma
                                    * rounded_operand_sum
                                    * (Decimal(1) + Decimal("1e-60"))
                                )
                                error = existing_error + rounding_error
                                if error > error_limit:
                                    continue

                                peak_left_first = max(
                                    left.peak,
                                    left_size + right.peak,
                                    triple,
                                )
                                peak_right_first = max(
                                    right.peak,
                                    right_size + left.peak,
                                    triple,
                                )
                                if peak_left_first <= peak_right_first:
                                    peak, order = peak_left_first, 0
                                else:
                                    peak, order = peak_right_first, 1

                                ledger = {
                                    "precision": precision,
                                    "contraction_length": contraction_length,
                                    "gamma_n": gamma,
                                    "lhs_magnitude": a,
                                    "rhs_magnitude": b,
                                    "nominal_magnitude": nominal,
                                    "input_error_bound": existing_error,
                                    "rounding_error_bound": rounding_error,
                                    "output_error_bound": error,
                                    "output_magnitude_bound": nominal + error,
                                    "step_cost": node_muls * precision.cost_per_mult,
                                }
                                tree = (
                                    "pnode",
                                    out_axes,
                                    left.tree,
                                    right.tree,
                                    contracted,
                                    node_muls,
                                    order,
                                    precision,
                                    nominal,
                                    error,
                                    ledger,
                                )
                                collected.append(
                                    _PrecisionEntry(
                                        left.cost
                                        + right.cost
                                        + node_muls * precision.cost_per_mult,
                                        peak,
                                        nominal,
                                        error,
                                        tree,
                                        order,
                                    )
                                )
            sub = (sub - 1) & mask
        front = _prune_precision(collected, limit)
        if front:
            fronts[mask] = front

    front = fronts.get(N - 1)
    if not front:
        raise ConstraintsInfeasibleError(
            "已穷举全部合法顺序和两档精度组合：不存在峰值 <= %d 且最终绝对误差 <= %s 的计划"
            % (limit, format(error_limit, "f"))
        )
    return min(front, key=lambda e: (e.cost, e.peak, e.error)), labels_order


# ---------------------------------------------------------------------------
# 计划重建（线性化 + 峰值证据）
# ---------------------------------------------------------------------------


def _build_plan(specs, label_size, limit, entry, labels_order):
    by_name = {t.name: t for t in specs}
    steps: list[ContractStep] = []
    counter = [0]

    def ref_input(name):
        t = by_name[name]
        return OperandRef("input", name, t.axes, t.shape, t.size, None)

    def emit(tree, held: tuple[tuple[str, int], ...]):
        """求值 tree；held 为求值期间始终驻留的（中间量名, 大小）。"""
        kind = tree[0]
        if kind == "leaf":
            return ref_input(tree[1])
        _, out_axes, tl, tr, contracted, node_muls, order = tree
        if order == 0:
            first_tree, second_tree, first_side = tl, tr, "lhs"
        else:
            first_tree, second_tree, first_side = tr, tl, "rhs"
        op_first = emit(first_tree, held)
        held2 = held
        if op_first.kind == "intermediate":
            held2 = held + ((op_first.name, op_first.size),)
        op_second = emit(second_tree, held2)
        if order == 0:
            lhs, rhs = op_first, op_second
        else:
            lhs, rhs = op_second, op_first
        idx = counter[0]
        counter[0] += 1
        out_shape = tuple(label_size[a] for a in out_axes)
        out_size = math.prod(out_shape)
        out_name = "T%d" % idx
        out_ref = OperandRef(
            "intermediate", out_name, out_axes, out_shape, out_size, idx
        )
        residency = held + (
            (lhs.name, lhs.size),
            (rhs.name, rhs.size),
            (out_name, out_size),
        )
        step = ContractStep(
            index=idx,
            output_name=out_name,
            lhs=lhs,
            rhs=rhs,
            evaluated_first=first_side,
            contracted_axes=tuple(contracted),
            output_axes=out_axes,
            output_shape=out_shape,
            mult_count=node_muls,
            residency=residency,
            peak_at_step=sum(sz for _, sz in residency),
        )
        steps.append(step)
        return out_ref

    result = emit(entry.tree, ())
    assert steps, "internal error: plan without steps"  # n>=2 时必有收缩

    peak = max((s.peak_at_step for s in steps), default=0)
    peak_step = None
    for s in steps:
        if s.peak_at_step == peak:
            peak_step = s.index
            break

    output_axes = tuple(
        lbl for lbl in labels_order if sum(
            lbl in t.axes for t in specs
        ) == 1
    )
    output_shape = tuple(label_size[a] for a in output_axes)

    return ContractionPlan(
        tensors=tuple(specs),
        memory_limit=limit,
        output_axes=output_axes,
        output_shape=output_shape,
        total_mult_count=entry.muls,
        peak_memory=peak,
        peak_step=peak_step,
        steps=tuple(steps),
        tree=entry.tree,
        tree_text=format_tree(entry.tree),
    )


def _build_trivial_plan(specs, label_size, limit, labels_order):
    t = specs[0]
    output_axes = tuple(
        lbl for lbl in labels_order if sum(lbl in x.axes for x in specs) == 1
    )
    return ContractionPlan(
        tensors=tuple(specs),
        memory_limit=limit,
        output_axes=output_axes,
        output_shape=tuple(label_size[a] for a in output_axes),
        total_mult_count=0,
        peak_memory=0,
        peak_step=None,
        steps=(),
        tree=("leaf", t.name),
        tree_text=format_tree(("leaf", t.name)),
    )


def _build_precision_plan(
    specs,
    label_size,
    limit,
    error_limit,
    value_bounds,
    input_errors,
    levels,
    entry,
    labels_order,
):
    by_name = {t.name: t for t in specs}
    steps = []
    counter = [0]

    def ref_input(name):
        tensor = by_name[name]
        return (
            OperandRef("input", name, tensor.axes, tensor.shape, tensor.size, None),
            value_bounds[name],
            input_errors[name],
        )

    def emit(tree, held):
        kind = tree[0]
        if kind == "leaf":
            return ref_input(tree[1])
        (
            _kind,
            out_axes,
            tree_left,
            tree_right,
            contracted,
            node_muls,
            order,
            precision,
            _nominal,
            _error,
            ledger,
        ) = tree
        if order == 0:
            left_ref, left_nominal, left_error = emit(tree_left, held)
            held_left = held
            if left_ref.kind == "intermediate":
                held_left = held + ((left_ref.name, left_ref.size),)
            right_ref, right_nominal, right_error = emit(tree_right, held_left)
            first_side = "lhs"
        else:
            right_ref, right_nominal, right_error = emit(tree_right, held)
            held_right = held
            if right_ref.kind == "intermediate":
                held_right = held + ((right_ref.name, right_ref.size),)
            left_ref, left_nominal, left_error = emit(tree_left, held_right)
            first_side = "rhs"

        idx = counter[0]
        counter[0] += 1
        out_shape = tuple(label_size[a] for a in out_axes)
        out_size = math.prod(out_shape)
        out_name = "T%d" % idx
        out_ref = OperandRef(
            "intermediate", out_name, out_axes, out_shape, out_size, idx
        )
        residency = held + (
            (left_ref.name, left_ref.size),
            (right_ref.name, right_ref.size),
            (out_name, out_size),
        )
        step = PrecisionContractStep(
            index=idx,
            output_name=out_name,
            lhs=left_ref,
            rhs=right_ref,
            evaluated_first=first_side,
            contracted_axes=tuple(contracted),
            output_axes=out_axes,
            output_shape=out_shape,
            mult_count=node_muls,
            residency=residency,
            peak_at_step=sum(size for _, size in residency),
            precision=precision,
            contraction_length=ledger["contraction_length"],
            gamma_n=ledger["gamma_n"],
            lhs_magnitude=left_nominal,
            rhs_magnitude=right_nominal,
            nominal_magnitude=ledger["nominal_magnitude"],
            input_error_bound=ledger["input_error_bound"],
            rounding_error_bound=ledger["rounding_error_bound"],
            output_error_bound=ledger["output_error_bound"],
            output_magnitude_bound=ledger["output_magnitude_bound"],
            step_cost=ledger["step_cost"],
        )
        steps.append(step)
        return (
            out_ref,
            ledger["nominal_magnitude"],
            ledger["output_error_bound"],
        )

    emit(entry.tree, ())
    peak = max((s.peak_at_step for s in steps), default=0)
    peak_step = None
    for step in steps:
        if step.peak_at_step == peak:
            peak_step = step.index
            break
    output_axes = tuple(
        label for label in labels_order
        if sum(label in tensor.axes for tensor in specs) == 1
    )
    output_shape = tuple(label_size[a] for a in output_axes)
    return PrecisionPlan(
        tensors=tuple(specs),
        memory_limit=limit,
        output_axes=output_axes,
        output_shape=output_shape,
        total_mult_count=sum(s.mult_count for s in steps),
        peak_memory=peak,
        peak_step=peak_step,
        steps=tuple(steps),
        tree=entry.tree,
        tree_text=format_tree(entry.tree),
        total_cost=entry.cost,
        error_limit=error_limit,
        final_error_bound=entry.error,
        value_bounds=dict(value_bounds),
        input_error_bounds=dict(input_errors),
        precision_levels=tuple(levels),
    )


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def find_plan(tensors, memory_limit: int, *, search_budget: Optional[int] = None):
    """求满足 memory_limit 的最少标量乘法收缩计划。

    参数:
        tensors: 1..MAX_TENSORS 个 TensorSpec。
        memory_limit: 正整数，峰值驻留标量元素数上限。
        search_budget: None（默认，对 n<=10 穷举）或正整数，限定候选
            收缩求值次数；耗尽时抛 BudgetExhaustedError（不返回残缺结果）。

    返回: ContractionPlan。

    异常:
        InvalidProblemError: 输入违反契约（含 NaN/Infinity、bool 整数等）。
        DisconnectedNetworkError: 收缩网络不连通，证明无任何合法计划。
        MemoryLimitInfeasibleError: 完整穷举后证明无满足上限的计划。
        BudgetExhaustedError: 预算耗尽，尚未证明无解。
    """
    specs, label_size, label_count, limit, budget = _validate(
        tensors, memory_limit, search_budget
    )
    labels_order: list[str] = []
    for t in specs:
        for lbl in t.axes:
            if lbl not in labels_order:
                labels_order.append(lbl)
    if len(specs) == 1:
        return _build_trivial_plan(specs, label_size, limit, labels_order)
    _check_connected(specs, label_count)
    entry, _fronts, _ax, _sz, _lo = _solve(
        specs, label_size, label_count, limit, budget
    )
    plan = _build_plan(specs, label_size, limit, entry, labels_order)
    # 内部不变量：DP 中已按 limit 精确剪枝，重建出的计划不可能超限。
    if plan.peak_memory > limit:
        raise TensorPlanningError(
            "内部不变量被破坏：计划峰值 %d 超过上限 %d"
            % (plan.peak_memory, limit)
        )
    return plan


def find_precision_plan(
    tensors,
    memory_limit: int,
    error_limit,
    value_bounds,
    *,
    input_error_bounds=None,
    precision_levels=DEFAULT_PRECISION_LEVELS,
    search_budget: Optional[int] = None,
):
    """同时选择括号顺序、求值顺序和每步两档精度，最小化精度加权成本。"""
    (
        specs,
        label_size,
        label_count,
        limit,
        err_limit,
        bounds,
        input_errors,
        levels,
        budget,
    ) = _validate_precision_args(
        tensors,
        memory_limit,
        error_limit,
        value_bounds,
        input_error_bounds,
        precision_levels,
        search_budget,
    )
    if len(specs) == 1:
        tensor = specs[0]
        final_error = input_errors[tensor.name]
        if final_error > err_limit:
            raise ConstraintsInfeasibleError(
                "唯一输入已有误差 %s 超过最终误差上限 %s"
                % (format(final_error, "f"), format(err_limit, "f"))
            )
        labels_order = []
        for item in specs:
            for label in item.axes:
                if label not in labels_order:
                    labels_order.append(label)
        return PrecisionPlan(
            tensors=tuple(specs),
            memory_limit=limit,
            output_axes=tuple(
                label for label in labels_order
                if sum(label in x.axes for x in specs) == 1
            ),
            output_shape=tuple(label_size[a] for a in tensor.axes),
            total_mult_count=0,
            peak_memory=0,
            peak_step=None,
            steps=(),
            tree=("leaf", tensor.name),
            tree_text=format_tree(("leaf", tensor.name)),
            total_cost=0,
            error_limit=err_limit,
            final_error_bound=final_error,
            value_bounds=dict(bounds),
            input_error_bounds=dict(input_errors),
            precision_levels=levels,
        )

    _check_connected(specs, label_count)
    entry, labels_order = _solve_precision(
        specs,
        label_size,
        limit,
        err_limit,
        bounds,
        input_errors,
        levels,
        budget,
    )
    plan = _build_precision_plan(
        specs,
        label_size,
        limit,
        err_limit,
        bounds,
        input_errors,
        levels,
        entry,
        labels_order,
    )
    if plan.peak_memory > limit or plan.final_error_bound > err_limit:
        raise TensorPlanningError("内部不变量被破坏：重建计划不满足约束")
    return plan


def pareto_front(tensors, *, search_budget: Optional[int] = None):
    """返回无内存限制下的 (乘法数, 峰值) Pareto 前沿，按乘法数升序。

    用于观察成本/内存权衡：前沿上没有点在两个目标上同时更优。
    异常语义同 find_plan（无 memory_limit 相关拒绝）。
    """
    specs, label_size, label_count, _limit, budget = _validate(
        tensors, 10**30, search_budget
    )
    labels_order: list[str] = []
    for t in specs:
        for lbl in t.axes:
            if lbl not in labels_order:
                labels_order.append(lbl)
    if len(specs) == 1:
        return (FrontPoint(0, 0, ("leaf", specs[0].name)),)
    _check_connected(specs, label_count)
    _entry, fronts, _ax, _sz, _lo = _solve(
        specs, label_size, label_count, None, budget
    )
    full = (1 << len(specs)) - 1
    return tuple(
        FrontPoint(e.muls, e.peak, e.tree)
        for e in sorted(fronts[full], key=lambda e: (e.muls, e.peak))
    )


def format_tree(tree) -> str:
    """把括号树渲染为可读文本，叶子为张量名，节点标注输出轴。"""
    kind = tree[0]
    if kind == "leaf":
        return tree[1]
    if tree[0] == "pnode":
        _, out_axes, tl, tr, _contracted, _muls, _order, precision, *_ = tree
        return "(%s x %s:%s)->[%s]" % (
            format_tree(tl),
            format_tree(tr),
            precision.name,
            ",".join(out_axes) if out_axes else "scalar",
        )
    _, out_axes, tl, tr, _contracted, _muls, _order = tree
    return "(%s x %s)->[%s]" % (
        format_tree(tl),
        format_tree(tr),
        ",".join(out_axes) if out_axes else "scalar",
    )

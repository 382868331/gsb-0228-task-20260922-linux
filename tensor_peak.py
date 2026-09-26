"""tensor_peak — 张量收缩的峰值内存规划。

只读入稠密张量的维度标签与每维大小（标量元素个数即内存单位），用子集
DP + (标量乘法数, 峰值驻留) Pareto 前沿，在两两收缩、禁止外积的前提下，
求满足内存上限的最少标量乘法收缩计划，并输出括号树、逐步形状与峰值证据。

本模块只使用 Python 标准库，全部计算为精确整数运算，不涉及浮点误差。
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
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
    "ErrorLimitInfeasibleError",
    "PrecisionInfeasibleError",
    "PrecisionLevel",
    "FP32",
    "FP64",
    "PRECISION_LEVELS",
    "PrecisionContractStep",
    "PrecisionContractionPlan",
    "find_plan",
    "find_precision_plan",
    "pareto_front",
    "format_tree",
    "decimal_rounding_audit",
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
    abs_bound: Fraction = 1
    abs_error: Fraction = 0

    def __init__(self, name, axes, shape, abs_bound=1, abs_error=0):
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
        bound = _check_nonneg_number(abs_bound, "张量 %r 的 abs_bound" % name)
        prior = _check_nonneg_number(abs_error, "张量 %r 的 abs_error" % name)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "abs_bound", bound)
        object.__setattr__(self, "abs_error", prior)

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


# ---------------------------------------------------------------------------

# 两档模拟浮点精度
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionLevel:
    """一档模拟浮点精度。

    name:    精度名（'fp32'/'fp64'）。
    p:       二进制有效数字位数（含隐含位），fp32=24，fp64=53。
    u:       单位舍入（机器 epsilon 的一半），u = 2**-p。
    mul_cost: 一次标量乘法（连同一次累加）在该精度下的成本权重。
    """

    name: str
    p: int
    u: Fraction
    mul_cost: int

    def __str__(self):
        return self.name


FP32 = PrecisionLevel("fp32", 24, Fraction(1, 2 ** 24), 1)
FP64 = PrecisionLevel("fp64", 53, Fraction(1, 2 ** 53), 4)
PRECISION_LEVELS: tuple[PrecisionLevel, ...] = (FP32, FP64)


@dataclass(frozen=True)
class PrecisionContractStep:
    """带精度、内存与误差账本的一次两两收缩。

    precision 为本步采用的精度；dot_length K = 收缩轴长度（每输出元素的
    点积项数）。rounding_gamma = K*u/(1-K*u)（K*u >= 1 时该精度对本步
    不可用，搜索中排除）。

    误差/幅度账本分三类：
    - input_error_contribution：两侧操作数已有误差传播到本步输出的部分；
    - rounding_error：本步点积 K 项舍入的标准先验界（γ_K · 项幅度和）；
    - total_abs_error：本步输出元素的总绝对误差上界（前两者之和）。
    operand_mag_sums: (lhs 项幅度和 S_l, rhs S_r, 乘积幅度和 Σ |a||b|)，
    便于独立复核。
    """

    index: int
    output_name: str
    lhs: OperandRef
    rhs: OperandRef
    precision: PrecisionLevel
    dot_length: int
    evaluated_first: str
    contracted_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    output_shape: tuple[int, ...]
    mult_count: int
    step_cost: int
    residency: tuple[tuple[str, int], ...]
    peak_at_step: int
    lhs_bound: Fraction
    rhs_bound: Fraction
    term_mag_sum: Fraction
    output_true_bound: Fraction
    output_stored_bound: Fraction
    input_error_contribution: Fraction
    rounding_error: Fraction
    total_abs_error: Fraction


@dataclass(frozen=True)
class PrecisionContractionPlan:
    """联合选择括号顺序与每步精度后的完整计划。"""

    tensors: tuple[TensorSpec, ...]
    memory_limit: int
    error_limit: Fraction
    output_axes: tuple[str, ...]
    output_shape: tuple[int, ...]
    total_cost: int
    total_mult_count: int
    peak_memory: int
    peak_step: Optional[int]
    final_abs_error: Fraction
    final_true_bound: Fraction
    steps: tuple[PrecisionContractStep, ...]
    tree: tuple
    tree_text: str


class ErrorLimitInfeasibleError(InfeasibleError):
    """已完整穷举全部括号顺序×精度组合，证明最终误差无法满足上限。"""


class PrecisionInfeasibleError(InfeasibleError):
    """点积长度对全部给定精度都不满足 K*u < 1，该收缩无任何误差界。"""


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


def _check_nonneg_number(value, what: str) -> Fraction:
    """校验非负实数（int/Fraction/Decimal/有限 float），统一为 Fraction。"""
    if isinstance(value, bool):
        raise InvalidProblemError("%s 不得为 bool 类型" % what)
    if isinstance(value, Fraction):
        frac = value
    elif isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise InvalidProblemError("%s 不得为 NaN/Infinity" % what)
        frac = Fraction(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidProblemError("%s 不得为 NaN/Infinity" % what)
        frac = Fraction(value)
    else:
        raise InvalidProblemError(
            "%s 必须是非负实数（int/fractions.Fraction/decimal.Decimal/float），得到 %r"
            % (what, type(value).__name__)
        )
    if frac < 0:
        raise InvalidProblemError("%s 必须 >= 0，得到 %r" % (what, value))
    return frac


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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PEntry:
    cost: int
    peak: int
    muls: int
    bound: Fraction
    error: Fraction
    tree: tuple
    order: int


def _pprune(entries, limit):
    """四维 Pareto：cost/peak/error/stored_bound 互不支配；按内存上限剪枝。"""
    uniq = {}
    for e in entries:
        if limit is not None and e.peak > limit:
            continue
        key = (e.cost, e.peak, e.bound, e.error)
        uniq[key] = e
    pts = list(uniq.values())
    kept: list[_PEntry] = []
    for i, e in enumerate(pts):
        dominated = False
        for j, g in enumerate(pts):
            if i == j:
                continue
            if (
                g.cost <= e.cost
                and g.peak <= e.peak
                and g.bound <= e.bound
                and g.error <= e.error
                and (
                    g.cost < e.cost
                    or g.peak < e.peak
                    or g.bound < e.bound
                    or g.error < e.error
                )
            ):
                dominated = True
                break
        if not dominated:
            kept.append(e)
    return kept


def _step_error_bounds(M_l, e_l, M_r, e_r, k_len, level):
    """标准点积舍入界（每项一次乘法一次舍入，K 项求和）。

    返回 (stored_bound, true_bound, input_err, rounding_err, total_err, gamma)；
    K*u >= 1 时该精度不满足界的前提，返回 None。
    """
    ku = k_len * level.u
    if ku >= 1:
        return None
    gamma = ku / (1 - ku)
    term_sum = k_len * M_l * M_r
    input_err = k_len * (e_l * M_r + e_r * M_l + e_l * e_r)
    rounding_err = gamma * term_sum
    total = input_err + rounding_err
    stored = term_sum + rounding_err
    true_bound = k_len * (M_l + e_l) * (M_r + e_r)
    return stored, true_bound, input_err, rounding_err, total, gamma


def _solve_precision(specs, label_size, label_count, limit, budget, levels):
    n = len(specs)
    N = 1 << n

    labels_order: list[str] = []
    for t in specs:
        for lbl in t.axes:
            if lbl not in labels_order:
                labels_order.append(lbl)

    axes_of = [None] * N
    size_of = [0] * N
    for mask in range(1, N):
        idx_of = {lbl: i for i, lbl in enumerate(labels_order)}
        counts = [0] * len(labels_order)
        members = mask
        while members:
            b = members & -members
            j = b.bit_length() - 1
            for lbl in specs[j].axes:
                counts[idx_of[lbl]] += 1
            members ^= b
        ax = tuple(lbl for lbl, c in zip(labels_order, counts) if c == 1)
        axes_of[mask] = ax
        size_of[mask] = math.prod(label_size[a] for a in ax)

    remaining = budget
    fronts: dict[int, list[_PEntry]] = {}
    # 每个子集记录：是否存在共享轴的合法划分、是否出现过某精度下 K*u>=1。
    has_pair = set()
    has_gamma_failure = set()
    # 子集为空的原因：'precision'（某收缩对全部档 K*u>=1）或 'memory'。
    empty_reason: dict[int, str] = {}
    for i, t in enumerate(specs):
        fronts[1 << i] = [
            _PEntry(0, 0, 0, t.abs_bound, t.abs_error, ("leaf", t.name), 0)
        ]

    for mask in range(1, N):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        sub = (mask - 1) & mask
        collected: list[_PEntry] = []
        while sub:
            other = mask ^ sub
            if (sub & low) and sub < mask:
                ax_l = axes_of[sub]
                ax_r = axes_of[other]
                shared = set(ax_l) & set(ax_r)
                if shared:
                    has_pair.add(mask)
                    fl = fronts.get(sub)
                    fr = fronts.get(other)
                    if fl and fr:
                        set_l, set_r = set(ax_l), set(ax_r)
                        union = set_l | set_r
                        out_axes = tuple(
                            a for a in labels_order
                            if a in union and a not in shared
                        )
                        s_out = math.prod(label_size[a] for a in out_axes)
                        node_muls = math.prod(label_size[a] for a in union)
                        k_len = node_muls // s_out if s_out else node_muls
                        s_l, s_r = size_of[sub], size_of[other]
                        triple = s_l + s_r + s_out
                        contracted = tuple(a for a in labels_order if a in shared)
                        for el in fl:
                            for er in fr:
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
                                if limit is not None and peak > limit:
                                    continue
                                for level in levels:
                                    if remaining is not None:
                                        if remaining == 0:
                                            raise BudgetExhaustedError(
                                                "搜索预算（候选 顺序×精度 求值次数）"
                                                "耗尽，搜索停止；未证明无解，请增大 search_budget"
                                            )
                                        remaining -= 1
                                    eb = _step_error_bounds(
                                        el.bound, el.error, er.bound, er.error,
                                        k_len, level,
                                    )
                                    if eb is None:
                                        has_gamma_failure.add(mask)
                                        continue
                                    stored, true_b, in_err, rnd, total, gamma = eb
                                    tree = (
                                        "node",
                                        out_axes,
                                        el.tree,
                                        er.tree,
                                        contracted,
                                        node_muls,
                                        order,
                                        level.name,
                                    )
                                    collected.append(
                                        _PEntry(
                                            el.cost + er.cost
                                            + node_muls * level.mul_cost,
                                            peak,
                                            el.muls + er.muls + node_muls,
                                            stored,
                                            total,
                                            tree,
                                            order,
                                        )
                                    )
            sub = (sub - 1) & mask

        front = _pprune(collected, limit)
        if front:
            fronts[mask] = front
        else:
            sub2 = (mask - 1) & mask
            saw_pair = False
            saw_gamma = False
            saw_empty_child = False
            child_precision = False
            while sub2:
                other2 = mask ^ sub2
                if (sub2 & low) and sub2 < mask:
                    if set(axes_of[sub2]) & set(axes_of[other2]):
                        saw_pair = True
                        if sub2 not in fronts or other2 not in fronts:
                            saw_empty_child = True
                            if (empty_reason.get(sub2) == "precision"
                                    or empty_reason.get(other2) == "precision"):
                                child_precision = True
                sub2 = (sub2 - 1) & mask
            reason = None
            if mask in has_gamma_failure or child_precision:
                reason = "precision"
            elif saw_pair and not saw_empty_child:
                # 有合法划分、无子集失败、收集为空：只能是峰值剪枝。
                reason = "memory"
            elif saw_empty_child:
                reason = "memory"
            if reason:
                empty_reason[mask] = reason

    full = N - 1
    front = fronts.get(full)
    if not front:
        if empty_reason.get(full) == "precision":
            raise PrecisionInfeasibleError(
                "已穷举全部精度组合：存在点积长度 K 使 K*u >= 1，无法给出有界误差"
            )
        raise MemoryLimitInfeasibleError(
            "已穷举全部合法收缩树与精度组合：不存在峰值内存 <= %d 标量元素的计划"
            % limit
        )
    return front, fronts, axes_of, size_of, labels_order


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def _build_precision_plan(specs, label_size, limit, error_limit, entry,
                          labels_order, levels):
    level_by_name = {lv.name: lv for lv in levels}
    in_by_name = {t.name: t for t in specs}
    steps: list[PrecisionContractStep] = []
    counter = [0]
    # 已产出操作数的账本：name -> (axes, size, bound, error)
    ledger: dict[str, tuple] = {}

    def ref_input(name):
        t = in_by_name[name]
        return OperandRef("input", name, t.axes, t.shape, t.size, None)

    def emit(tree, held):
        kind = tree[0]
        if kind == "leaf":
            name = tree[1]
            ledger[name] = (
                in_by_name[name].axes, in_by_name[name].size,
                in_by_name[name].abs_bound, in_by_name[name].abs_error,
            )
            return ref_input(name)
        (_, out_axes, tl, tr, contracted, node_muls, order, level_name) = tree
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
        out_name = "T%d" % idx
        out_shape = tuple(label_size[a] for a in out_axes)
        out_size = math.prod(out_shape)
        level = level_by_name[level_name]
        _ax_l, _sz_l, m_l, e_l = ledger[lhs.name]
        _ax_r, _sz_r, m_r, e_r = ledger[rhs.name]
        k_len = math.prod(label_size[a] for a in contracted)
        eb = _step_error_bounds(m_l, e_l, m_r, e_r, k_len, level)
        stored, true_b, in_err, rnd, total, gamma = eb
        out_ref = OperandRef(
            "intermediate", out_name, out_axes, out_shape, out_size, idx
        )
        residency = held + (
            (lhs.name, lhs.size),
            (rhs.name, rhs.size),
            (out_name, out_size),
        )
        step = PrecisionContractStep(
            index=idx,
            output_name=out_name,
            lhs=lhs,
            rhs=rhs,
            precision=level,
            dot_length=k_len,
            evaluated_first=first_side,
            contracted_axes=tuple(contracted),
            output_axes=out_axes,
            output_shape=out_shape,
            mult_count=node_muls,
            step_cost=node_muls * level.mul_cost,
            residency=residency,
            peak_at_step=sum(z for _, z in residency),
            lhs_bound=m_l,
            rhs_bound=m_r,
            term_mag_sum=k_len * m_l * m_r,
            output_true_bound=true_b,
            output_stored_bound=stored,
            input_error_contribution=in_err,
            rounding_error=rnd,
            total_abs_error=total,
        )
        steps.append(step)
        ledger[out_name] = (out_axes, out_size, stored, total)
        return out_ref

    emit(entry.tree, ())
    assert steps
    final = steps[-1]
    peak = max(s.peak_at_step for s in steps)
    peak_step = next(s.index for s in steps if s.peak_at_step == peak)
    output_axes = tuple(
        lbl for lbl in labels_order
        if sum(lbl in t.axes for t in specs) == 1
    )
    return PrecisionContractionPlan(
        tensors=tuple(specs),
        memory_limit=limit,
        error_limit=error_limit,
        output_axes=output_axes,
        output_shape=tuple(label_size[a] for a in output_axes),
        total_cost=entry.cost,
        total_mult_count=entry.muls,
        peak_memory=peak,
        peak_step=peak_step,
        final_abs_error=final.total_abs_error,
        final_true_bound=final.output_true_bound,
        steps=tuple(steps),
        tree=entry.tree,
        tree_text=format_tree(entry.tree),
    )


def find_precision_plan(
    tensors,
    memory_limit: int,
    error_limit,
    *,
    search_budget: Optional[int] = None,
    levels: Sequence[PrecisionLevel] = PRECISION_LEVELS,
):
    """联合选择括号顺序与每步精度，在内存与最终绝对误差上限内最小化成本。

    每个输入 TensorSpec 可给 abs_bound（元素绝对值上界，默认 1）与
    abs_error（输入已有绝对误差上界，默认 0）。规划器不读取真实张量数据。

    error_limit: 非负实数（int/Fraction/Decimal/float），最终绝对误差硬上限。
    levels: 可选精度档序列，默认 (FP32, FP64)；每档给出单位舍入与乘法成本。

    异常:
        InvalidProblemError: 输入契约错误。
        DisconnectedNetworkError: 网络不连通。
        PrecisionInfeasibleError: 存在点积使 K*u>=1（所有精度都无有界误差）。
        MemoryLimitInfeasibleError: 穷举证明内存上限不可行。
        ErrorLimitInfeasibleError: 内存可行但穷举证明误差上限不可行。
        BudgetExhaustedError: 预算耗尽，未证明无解。
    """
    levels = _validate_levels(levels)
    specs, label_size, label_count, limit, budget = _validate(
        tensors, memory_limit, search_budget
    )
    err_cap = _check_nonneg_number(error_limit, "error_limit")
    labels_order: list[str] = []
    for t in specs:
        for lbl in t.axes:
            if lbl not in labels_order:
                labels_order.append(lbl)
    if len(specs) == 1:
        t = specs[0]
        if t.abs_error > err_cap:
            raise ErrorLimitInfeasibleError(
                "单输入无收缩步骤：输入已有误差 %s 超过误差上限 %s"
                % (t.abs_error, err_cap)
            )
        output_axes = tuple(
            lbl for lbl in labels_order
            if sum(lbl in x.axes for x in specs) == 1
        )
        out_shape = tuple(label_size[a] for a in output_axes)
        ref = OperandRef("input", t.name, t.axes, t.shape, t.size, None)
        return PrecisionContractionPlan(
            tensors=(t,),
            memory_limit=limit,
            error_limit=err_cap,
            output_axes=output_axes,
            output_shape=out_shape,
            total_cost=0,
            total_mult_count=0,
            peak_memory=0,
            peak_step=None,
            final_abs_error=t.abs_error,
            final_true_bound=t.abs_bound,
            steps=(),
            tree=("leaf", t.name),
            tree_text=format_tree(("leaf", t.name)),
        )
    _check_connected(specs, label_count)
    front, _fronts, _ax, _sz, _lo = _solve_precision(
        specs, label_size, label_count, limit, budget, levels
    )
    feasible = [e for e in front if e.error <= err_cap]
    if not feasible:
        best_err = min(e.error for e in front)
        raise ErrorLimitInfeasibleError(
            "已穷举全部括号顺序×精度组合：最优最终误差界 %s 仍超过上限 %s"
            % (best_err, err_cap)
        )
    entry = min(feasible, key=lambda e: (e.cost, e.peak, e.error))
    plan = _build_precision_plan(
        specs, label_size, limit, err_cap, entry, labels_order, levels
    )
    if plan.peak_memory > limit or plan.final_abs_error > err_cap:
        raise TensorPlanningError("内部不变量被破坏：计划超过内存或误差上限")
    return plan


def _validate_levels(levels):
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise InvalidProblemError("levels 必须是 PrecisionLevel 序列")
    lv_list = list(levels)
    if not lv_list:
        raise InvalidProblemError("levels 至少包含一档精度")
    names = set()
    for lv in lv_list:
        if not isinstance(lv, PrecisionLevel):
            raise InvalidProblemError(
                "levels 元素必须是 PrecisionLevel，得到 %r"
                % type(lv).__name__
            )
        if lv.name in names:
            raise InvalidProblemError("精度名重复: %r" % lv.name)
        names.add(lv.name)
        if not isinstance(lv.p, int) or lv.p < 1:
            raise InvalidProblemError("精度 %r 的 p 必须是正整数" % lv.name)
        if not isinstance(lv.mul_cost, int) or lv.mul_cost < 1:
            raise InvalidProblemError("精度 %r 的 mul_cost 必须是正整数" % lv.name)
    return tuple(lv_list)


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
    if len(tree) == 8:
        _, out_axes, tl, tr, _contracted, _muls, _order, _lvl = tree
    else:
        _, out_axes, tl, tr, _contracted, _muls, _order = tree
    return "(%s x %s)->[%s]" % (
        format_tree(tl),
        format_tree(tr),
        ",".join(out_axes) if out_axes else "scalar",
    )


# ---------------------------------------------------------------------------
# Decimal 逐步舍入模拟：独立核对误差账本的保守界
# ---------------------------------------------------------------------------


def _binary_round(x: Decimal, p: int) -> Decimal:
    """把 Decimal x 按 p 位二进制有效数字、就近偶舍入模拟。"""
    if x == 0:
        return x
    sign = -1 if x < 0 else 1
    ax = -x if x < 0 else x
    # 求 floor(log2(ax))：用 2 的幂做指数探测（Decimal 精确比较）。
    e = 0
    two = Decimal(2)
    # 先用十进制粗定位，再线性修正，避免大循环。
    approx = int(ax.log10() / Decimal(2).log10())
    e = approx
    if two ** e <= ax:
        while two ** (e + 1) <= ax:
            e += 1
    else:
        while two ** e > ax:
            e -= 1
    ulp = Decimal(2) ** (e - p + 1)
    q = (ax / ulp).to_integral_value(rounding=ROUND_HALF_EVEN)
    return sign * q * ulp


def _audit_fill_value(bound: Fraction) -> Decimal:
    """取不超过上界的最大 2 的幂（在任何二进制精度下都精确可表示）。"""
    if bound == 0:
        return Decimal(0)
    # floor(log2(bound))，对 Fraction 精确计算。
    num, den = bound.numerator, bound.denominator
    e = num.bit_length() - den.bit_length()
    # 修正到 2^e <= bound < 2^(e+1)。
    while Fraction(2 ** (e + 1)) <= bound:
        e += 1
    while Fraction(2 ** e) > bound:
        e -= 1
    return Decimal(2) ** e


def decimal_rounding_audit(plan, *, context_precision: int = 80, signs=(),
                           fill_scales=None, fill_values=None):
    """用 Decimal 逐步模拟二进制舍入，核对计划误差账本的保守界。

    输入张量不读取真实数据：按每个输入的 abs_bound 取一个不超过该上界的
    2 的幂作为元素值（在 fp32/fp64 下精确表示），并按计划逐元素执行收缩；
    每步同时用高 Decimal 精度计算精确值与“每项一次二进制舍入”的累加值。

    返回 tuple，每项为
    (step_index, observed_error, ledger_rounding_bound, ratio)，
    其中 observed_error 是该步输出上实测的舍入误差上界，恒满足
    observed <= ledger.rounding_error（账本为保守先验界）。

    signs: 可选 {输入名: +1/-1}，改变填充符号以产生不同对齐模式；默认全 +1。
    fill_scales: 可选 {输入名: Fraction(0..1)}，在该输入 abs_bound 内选取
        更小的 2 的幂填充，以制造非平凡对齐；默认取不超过上界的最大 2 的幂。
    fill_values: 可选 {输入名: Fraction}，直接指定 <= abs_bound 的常量填充值
        （用于对抗性对齐）；与 fill_scales 互斥（同名时后者优先）。
    该审计只针对输入已有误差为 0 的情形（账本中的舍入分量）。
    context_precision: Decimal 内部十进制有效数字位数。
    """
    if not isinstance(plan, PrecisionContractionPlan):
        raise InvalidProblemError(
            "decimal_rounding_audit 需要 PrecisionContractionPlan"
        )
    if any(t.abs_error != 0 for t in plan.tensors):
        raise InvalidProblemError("审计模拟仅支持输入 abs_error 均为 0 的计划")
    import itertools

    sign_map = {t.name: 1 for t in plan.tensors}
    sign_map.update(dict(signs))
    scales = {t.name: Fraction(1) for t in plan.tensors}
    if fill_scales:
        for name, sc in fill_scales.items():
            sc = _check_nonneg_number(sc, "fill_scales[%r]" % name)
            if sc > 1:
                raise InvalidProblemError(
                    "fill_scales[%r] = %r 必须 <= 1" % (name, sc)
                )
            scales[name] = sc
    custom = {}
    if fill_values:
        for name, val in fill_values.items():
            val = _check_nonneg_number(val, "fill_values[%r]" % name)
            spec = next(x for x in plan.tensors if x.name == name)
            if val > spec.abs_bound:
                raise InvalidProblemError(
                    "fill_values[%r] = %r 超过该输入 abs_bound" % (name, val)
                )
            custom[name] = val
    levels = {s.output_name: s.precision.p for s in plan.steps}
    inputs = {}
    for t in plan.tensors:
        if t.name in custom:
            v = Decimal(custom[t.name].numerator) / Decimal(
                custom[t.name].denominator)
        else:
            v = _audit_fill_value(t.abs_bound * scales[t.name])
        v = v * sign_map[t.name]
        inputs[t.name] = (
            t.axes,
            {idx: v for idx in itertools.product(
                *(range(d) for d in t.shape))},
        )
    live: dict[str, tuple] = {}
    reports = []

    def view(name):
        return inputs.get(name) or live[name]

    with localcontext() as ctx:
        ctx.prec = context_precision
        for step in plan.steps:
            ax_l, dat_l = view(step.lhs.name)
            ax_r, dat_r = view(step.rhs.name)
            union = list(ax_l) + [a for a in ax_r if a not in set(ax_l)]
            sizes = {}
            for op in (step.lhs, step.rhs):
                sizes.update(dict(zip(op.axes, op.shape)))
            exact = {}
            rounded = {}
            term_count = {}
            p = levels[step.output_name]
            for combo in itertools.product(
                *(range(sizes[a]) for a in union)
            ):
                pos = dict(zip(union, combo))
                kl = tuple(pos[a] for a in ax_l)
                kr = tuple(pos[a] for a in ax_r)
                ko = tuple(pos[a] for a in step.output_axes)
                prod_exact = dat_l[kl] * dat_r[kr]
                exact[ko] = exact.get(ko, Decimal(0)) + prod_exact
                acc = rounded.get(ko, Decimal(0))
                acc = _binary_round(
                    acc + _binary_round(prod_exact, p), p
                )
                rounded[ko] = acc
                term_count[ko] = term_count.get(ko, 0) + 1
            observed = max(
                abs(rounded[k] - exact[k]) for k in exact
            )
            bound = step.rounding_error
            bound_dec = Decimal(bound.numerator) / Decimal(bound.denominator)
            ratio = (observed / bound_dec) if bound_dec > 0 else Decimal(0)
            reports.append(
                (step.index, observed, bound_dec, ratio)
            )
            live[step.output_name] = (step.output_axes, rounded)
            for op in (step.lhs.name, step.rhs.name):
                if op in live:
                    live.pop(op, None)
    return tuple(reports)

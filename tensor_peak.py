"""tensor_peak — 张量收缩的峰值内存规划。

只读入稠密张量的维度标签与每维大小（标量元素个数即内存单位），用子集
DP + (标量乘法数, 峰值驻留) Pareto 前沿，在两两收缩、禁止外积的前提下，
求满足内存上限的最少标量乘法收缩计划，并输出括号树、逐步形状与峰值证据。

可选的指标奇偶标记：每个维度标签可标记为偶(0)或奇(1)。为收缩而把操作数
置换到规范轴序时，每交换一对奇指标贡献一个 -1 符号；计划逐步记录轴排列、
奇指标交换次数与符号，execute_plan 按此执行带符号数值收缩。不做任何
标记时全部指标为偶，符号恒为 +1，行为与无标记版本完全一致。

本模块只使用 Python 标准库，全部计算为精确整数运算，不涉及浮点误差。
"""

from __future__ import annotations

import itertools
import math
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
    "find_plan",
    "pareto_front",
    "format_tree",
    "execute_plan",
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
    parities: 可选，与 axes 等长的 0/1 元组（0=偶, 1=奇）；None 表示
        本张量不提供奇偶信息。同一标签在所有标记处必须一致。
    """

    name: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    parities: Optional[tuple[int, ...]] = None

    def __init__(self, name, axes, shape, parities=None):
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
        if parities is not None:
            if isinstance(parities, str) or not isinstance(parities, Sequence):
                raise InvalidProblemError(
                    "张量 %r 的 parities 必须是 0/1 序列或 None" % (name,)
                )
            parities = tuple(parities)
            if len(parities) != len(axes):
                raise InvalidProblemError(
                    "张量 %r 的 parities 长度 %d 与 axes 长度 %d 不一致"
                    % (name, len(parities), len(axes))
                )
            for p, m in enumerate(parities):
                if isinstance(m, bool) or not isinstance(m, int) or m not in (0, 1):
                    raise InvalidProblemError(
                        "张量 %r 的第 %d 个轴(%s)奇偶标记必须是 0 或 1，得到 %r"
                        % (name, p, axes[p], m)
                    )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "parities", parities)

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
    # 奇偶符号证据：为收缩把操作数置换到规范轴序的排列（canonical =
    # [axes[i] for i in permutation]）、其中奇指标对的交换次数及符号。
    # 无奇偶标记时排列仍为规范排列、交换次数为 0、符号恒为 +1。
    lhs_axis_permutation: tuple[int, ...] = ()
    rhs_axis_permutation: tuple[int, ...] = ()
    odd_swap_count: int = 0
    sign: int = 1


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
    # 整个计划的净符号（各步符号之积；单张量平凡计划为输入轴序
    # 置换到规范输出轴序的符号）。无奇偶标记时恒为 +1。
    sign: int = 1


@dataclass(frozen=True)
class FrontPoint:
    """Pareto 前沿上的一个点：mult_count 更少与 peak_memory 更低不可兼得。"""

    mult_count: int
    peak_memory: int
    tree: tuple


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

    # 奇偶标记按标签汇总：同一标签在所有标记处必须一致；未标记的
    # 张量不提供约束。全部未标记时 label_parity 为空（全部按偶处理）。
    label_parity: dict[str, int] = {}
    for t in specs:
        if t.parities is None:
            continue
        for lbl, m in zip(t.axes, t.parities):
            if lbl in label_parity:
                if label_parity[lbl] != m:
                    raise InvalidProblemError(
                        "指标 %r 的奇偶标记冲突: %d != %d（出现于张量 %r）"
                        % (lbl, label_parity[lbl], m, t.name)
                    )
            else:
                label_parity[lbl] = m

    limit = _check_positive_int(
        memory_limit, "memory_limit", allow_none=False
    )
    budget = _check_positive_int(
        search_budget, "search_budget", allow_none=True
    )
    return tuple(specs), label_size, label_count, label_parity, limit, budget


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


def _canonical_perm(axes, labels_order):
    """返回把 axes 置换到规范轴序（labels_order 过滤）的排列 perm，
    满足 canonical = [axes[i] for i in perm]。"""
    owned = set(axes)
    return tuple(axes.index(a) for a in labels_order if a in owned)


def _count_odd_swaps(axes, perm, label_parity):
    """置换 perm 中奇指标对的逆序数（即相邻交换展开下的奇-奇交换次数）。"""
    swaps = 0
    for x in range(len(perm)):
        lx = axes[perm[x]]
        if not label_parity.get(lx, 0):
            continue
        for y in range(x + 1, len(perm)):
            if perm[x] > perm[y] and label_parity.get(axes[perm[y]], 0):
                swaps += 1
    return swaps


def _build_plan(
    specs, label_size, limit, entry, labels_order, label_parity
):
    by_name = {t.name: t for t in specs}
    steps: list[ContractStep] = []
    counter = [0]

    def ref_input(name):
        t = by_name[name]
        return OperandRef("input", name, t.axes, t.shape, t.size, None)

    def perm_of(ref):
        return _canonical_perm(ref.axes, labels_order)

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
        lhs_perm = perm_of(lhs)
        rhs_perm = perm_of(rhs)
        odd_swaps = _count_odd_swaps(
            lhs.axes, lhs_perm, label_parity
        ) + _count_odd_swaps(rhs.axes, rhs_perm, label_parity)
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
            lhs_axis_permutation=lhs_perm,
            rhs_axis_permutation=rhs_perm,
            odd_swap_count=odd_swaps,
            sign=-1 if odd_swaps % 2 else 1,
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

    plan_sign = 1
    for s in steps:
        plan_sign *= s.sign

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
        sign=plan_sign,
    )


def _build_trivial_plan(specs, label_size, limit, labels_order, label_parity=None):
    t = specs[0]
    output_axes = tuple(
        lbl for lbl in labels_order if sum(lbl in x.axes for x in specs) == 1
    )
    perm = _canonical_perm(t.axes, labels_order)
    swaps = _count_odd_swaps(t.axes, perm, label_parity or {})
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
        sign=-1 if swaps % 2 else 1,
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
    specs, label_size, label_count, label_parity, limit, budget = _validate(
        tensors, memory_limit, search_budget
    )
    labels_order: list[str] = []
    for t in specs:
        for lbl in t.axes:
            if lbl not in labels_order:
                labels_order.append(lbl)
    if len(specs) == 1:
        return _build_trivial_plan(
            specs, label_size, limit, labels_order, label_parity
        )
    _check_connected(specs, label_count)
    entry, _fronts, _ax, _sz, _lo = _solve(
        specs, label_size, label_count, limit, budget
    )
    plan = _build_plan(
        specs, label_size, limit, entry, labels_order, label_parity
    )
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
    specs, label_size, label_count, _parity, _limit, budget = _validate(
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
    _, out_axes, tl, tr, _contracted, _muls, _order = tree
    return "(%s x %s)->[%s]" % (
        format_tree(tl),
        format_tree(tr),
        ",".join(out_axes) if out_axes else "scalar",
    )


# ---------------------------------------------------------------------------
# 带符号执行器
# ---------------------------------------------------------------------------


def _permute_data(axes, data, perm, sign, where):
    """按 perm 重排稠密数据键并乘符号；perm 为空元组视为恒等（兼容
    未记录排列的旧式手工构造步骤）。"""
    if not perm:
        perm = tuple(range(len(axes)))
    out = {}
    for key, value in data.items():
        if not isinstance(key, tuple) or len(key) != len(axes):
            raise InvalidProblemError(
                "%s 的数据键 %r 与轴数 %d 不符" % (where, key, len(axes))
            )
        out[tuple(key[i] for i in perm)] = sign * value
    return out


def execute_plan(plan, values):
    """按计划逐步执行带奇偶符号的稠密收缩，返回 {输出索引元组: 值}。

    参数:
        plan: find_plan 返回的 ContractionPlan。
        values: dict[张量名, dict[索引元组, 数值]]，按各输入张量自身的
            轴序给值；索引越界或缺失会给出可定位的 InvalidProblemError。

    每步先把两个操作数按步骤记录的轴排列置换到规范轴序（奇指标交换
    的符号已计入 step.sign），再对收缩轴求和；最终输出轴序即
    plan.output_axes 的规范顺序，净符号为 plan.sign。全部数值为精确
    整数运算（输入整数时输出必为整数）。
    """
    if not isinstance(plan, ContractionPlan):
        raise InvalidProblemError(
            "plan 必须是 ContractionPlan，得到 %r" % type(plan).__name__
        )
    if not isinstance(values, dict):
        raise InvalidProblemError("values 必须是 {张量名: {索引: 值}} 的 dict")

    def fetch(ref, perm, sign):
        if ref.kind == "input":
            if ref.name not in values:
                raise InvalidProblemError(
                    "values 缺少输入张量 %r 的数据" % ref.name
                )
            data = values[ref.name]
            if not isinstance(data, dict):
                raise InvalidProblemError(
                    "张量 %r 的数据必须是 {索引元组: 值} 的 dict" % ref.name
                )
            for key in data:
                if not isinstance(key, tuple) or len(key) != len(ref.axes):
                    raise InvalidProblemError(
                        "张量 %r 的数据键 %r 与轴数 %d 不符"
                        % (ref.name, key, len(ref.axes))
                    )
                for p, v in enumerate(key):
                    if isinstance(v, bool) or not isinstance(v, int) or not (
                        0 <= v < ref.shape[p]
                    ):
                        raise InvalidProblemError(
                            "张量 %r 的数据键 %r 第 %d 维越界（维大小 %d）"
                            % (ref.name, key, p, ref.shape[p])
                        )
            return _permute_data(ref.axes, data, perm, sign, "张量 %r" % ref.name)
        return _permute_data(
            ref.axes, live[ref.name], perm, sign, "中间量 %r" % ref.name
        )

    live: dict[str, dict] = {}
    for step in plan.steps:
        dat_l = fetch(step.lhs, step.lhs_axis_permutation, 1)
        dat_r = fetch(step.rhs, step.rhs_axis_permutation, 1)
        ax_l = [step.lhs.axes[i] for i in (
            step.lhs_axis_permutation or range(len(step.lhs.axes))
        )]
        ax_r = [step.rhs.axes[i] for i in (
            step.rhs_axis_permutation or range(len(step.rhs.axes))
        )]
        sizes = dict(zip(step.lhs.axes, step.lhs.shape))
        sizes.update(zip(step.rhs.axes, step.rhs.shape))
        union_axes = list(ax_l) + [a for a in ax_r if a not in set(ax_l)]
        out: dict = {}
        for combo in _product_ranges([sizes[a] for a in union_axes]):
            pos = dict(zip(union_axes, combo))
            kl = tuple(pos[a] for a in ax_l)
            kr = tuple(pos[a] for a in ax_r)
            ko = tuple(pos[a] for a in step.output_axes)
            out[ko] = out.get(ko, 0) + dat_l[kl] * dat_r[kr]
        if step.sign != 1:
            out = {k: step.sign * v for k, v in out.items()}
        live[step.output_name] = out
        for op in (step.lhs, step.rhs):
            if op.kind == "intermediate":
                live.pop(op.name, None)

    if not plan.steps:
        # 单张量平凡计划：仅做输入轴序 -> 规范输出轴序的带符号转置。
        t = plan.tensors[0]
        if t.name not in values:
            raise InvalidProblemError(
                "values 缺少输入张量 %r 的数据" % t.name
            )
        perm = _canonical_perm(t.axes, plan.output_axes)
        return _permute_data(
            t.axes, values[t.name], perm, plan.sign, "张量 %r" % t.name
        )
    return live[plan.steps[-1].output_name]


def _product_ranges(dims):
    """itertools.product(*(range(d) for d in dims))。"""
    return itertools.product(*(range(d) for d in dims))

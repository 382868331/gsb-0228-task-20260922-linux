"""tensor_peak — 张量收缩的峰值内存规划。

只读入稠密张量的维度标签与每维大小（标量元素个数即内存单位），用子集
DP + (标量乘法数, 峰值驻留) Pareto 前沿，在两两收缩、禁止外积的前提下，
求满足内存上限的最少标量乘法收缩计划，并输出括号树、逐步形状与峰值证据。

本模块只使用 Python 标准库，全部计算为精确整数运算，不涉及浮点误差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "MAX_TENSORS",
    "MAX_SLICES",
    "TensorSpec",
    "OperandRef",
    "ContractStep",
    "ContractionPlan",
    "SlicedPlan",
    "FrontPoint",
    "TensorPlanningError",
    "InvalidProblemError",
    "InfeasibleError",
    "DisconnectedNetworkError",
    "MemoryLimitInfeasibleError",
    "BudgetExhaustedError",
    "find_plan",
    "find_sliced_plan",
    "pareto_front",
    "format_tree",
]

MAX_TENSORS = 10
# find_sliced_plan 对每个切片都要完整调用一次规划器，切片数给出明确接口边界。
MAX_SLICES = 64


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
class SlicedPlan:
    """沿一个可切片收缩轴切块、逐切片收缩后的整体计划。

    sliced_axis:  被切片的收缩轴标签（恰出现于两个输入）。
    slice_size:   每片覆盖的该轴坐标数（最后一片可能更小）。
    num_slices:   切片数 ceil(轴维大小 / slice_size)。
    slices:       逐切片计划（S0..S{k-1} 顺序）；每个都是 find_plan
                  风格的完整 ContractionPlan，部分输出轴恒为最终输出轴
                  （被切轴在片内即被收缩，不出现在部分输出中）。
    total_mult_count: 全部切片标量乘法之和（用旧规划器逐片累计）。
    total_add_count:  把各片部分输出累加到常驻输出所需的标量加法数，
                      即 (num_slices - 1) * 输出元素数；首片初始化、
                      之后每片逐输出元素一次加法。
    peak_memory:  整体执行的峰值驻留（标量元素）。自第二片起，上一片
                  产出的完整输出作为常驻累加器（output_size 个元素）
                  在该片求值期间始终驻留；峰值 =
                  max(首片计划峰值, output_size + 其余各片计划峰值)。
    output_size:  最终输出的标量元素数（累加器大小）。
    output_axes / output_shape: 最终输出轴（出现 1 次的标签）与形状。
    """

    tensors: tuple[TensorSpec, ...]
    memory_limit: Optional[int]
    sliced_axis: str
    slice_size: int
    num_slices: int
    slices: tuple[ContractionPlan, ...]
    total_mult_count: int
    total_add_count: int
    peak_memory: int
    output_size: int
    output_axes: tuple[str, ...]
    output_shape: tuple[int, ...]


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
# 规划核心（find_plan 与逐切片规划共用）
# ---------------------------------------------------------------------------


def _plan_core(specs, label_size, label_count, limit, budget, labels_order):
    """输入已校验；返回满足 limit 的最优 ContractionPlan（n>=1）。"""
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
    return _plan_core(
        specs, label_size, label_count, limit, budget, labels_order
    )


def find_sliced_plan(
    tensors,
    memory_limit: int,
    sliced_axis: str,
    slice_size: int,
    *,
    search_budget: Optional[int] = None,
):
    """沿一个可切片的收缩轴切块，对每个切片沿用 find_plan 的规划器。

    参数:
        tensors: 1..MAX_TENSORS 个 TensorSpec（被切轴必须恰出现于两个
            输入，即为收缩轴；只出现一次的输出轴不可切）。
        memory_limit: 正整数。切片规划以“片内峰值 + 常驻累加器”为约束：
            首片无累加器，其余各片要求 plan.peak_memory + 输出元素数
            <= memory_limit；任一片不满足即 MemoryLimitInfeasibleError。
        sliced_axis: 要切片的收缩轴标签（非空 str）。
        slice_size: 正整数，每片覆盖的该轴坐标数；最后一片可能更小。
            切片数超过 MAX_SLICES 时按非法输入拒绝。
        search_budget: None 或正整数；为每片独立规划的候选求值次数上限
            （每片各自一份预算，耗尽抛 BudgetExhaustedError）。

    返回: SlicedPlan（逐切片计划、累计乘法数、累加加法数、含常驻累加器
        的整体峰值）。失败只抛异常，不产生任何半更新状态。

    异常:
        InvalidProblemError: 参数或切片设置违反契约。
        DisconnectedNetworkError / MemoryLimitInfeasibleError /
        BudgetExhaustedError: 语义同 find_plan，发生在任一切片规划中。
    """
    specs, label_size, label_count, limit, budget = _validate(
        tensors, memory_limit, search_budget
    )
    if not isinstance(sliced_axis, str) or not sliced_axis:
        raise InvalidProblemError(
            "sliced_axis 必须是非空 str，得到 %r" % (sliced_axis,)
        )
    if sliced_axis not in label_count:
        raise InvalidProblemError(
            "sliced_axis %r 不是任何输入张量的轴标签" % (sliced_axis,)
        )
    if label_count[sliced_axis] != 2:
        raise InvalidProblemError(
            "sliced_axis %r 出现 %d 次；只能对恰出现 2 次的收缩轴切片"
            % (sliced_axis, label_count[sliced_axis])
        )
    if isinstance(slice_size, bool):
        raise InvalidProblemError("slice_size 不得为 bool 类型")
    if isinstance(slice_size, float):
        if not math.isfinite(slice_size):
            raise InvalidProblemError("slice_size 不得为 NaN/Infinity")
        raise InvalidProblemError(
            "slice_size 必须是精确正整数，得到 float %r" % (slice_size,)
        )
    if not isinstance(slice_size, int) or slice_size < 1:
        raise InvalidProblemError(
            "slice_size 必须是正整数，得到 %r" % (slice_size,)
        )

    axis_dim = label_size[sliced_axis]
    num_slices = (axis_dim + slice_size - 1) // slice_size
    if num_slices > MAX_SLICES:
        raise InvalidProblemError(
            "轴 %r 维大小 %d、每片 %d 将产生 %d 个切片，超过接口上限 %d"
            % (sliced_axis, axis_dim, slice_size, num_slices, MAX_SLICES)
        )

    labels_order: list[str] = []
    for t in specs:
        for lbl in t.axes:
            if lbl not in labels_order:
                labels_order.append(lbl)
    output_axes = tuple(a for a in labels_order if label_count[a] == 1)
    output_shape = tuple(label_size[a] for a in output_axes)
    output_size = math.prod(output_shape)

    owners = [
        i for i, t in enumerate(specs) if sliced_axis in t.axes
    ]
    assert len(owners) == 2

    sliced_plans: list[ContractionPlan] = []
    total_mult = 0
    peak_overall = 0
    # 逐片规划：所有片先全部成功才构造 SlicedPlan（失败不留下半更新状态）。
    for k in range(num_slices):
        lo = k * slice_size
        width = min(slice_size, axis_dim - lo)
        owners_set = set(owners)
        slice_specs: list[TensorSpec] = []
        for i, t in enumerate(specs):
            if i not in owners_set:
                slice_specs.append(t)
                continue
            p = t.axes.index(sliced_axis)
            # 被切轴移到该张量轴序末尾：等价重排，且对两个属主一致，
            # 使两个片输入共享同一被切轴，片内即完成该块收缩。
            new_axes = tuple(a for q, a in enumerate(t.axes) if q != p)
            new_shape = tuple(
                d for q, d in enumerate(t.shape) if q != p
            )
            slice_specs.append(
                TensorSpec(
                    t.name, new_axes + (sliced_axis,), new_shape + (width,)
                )
            )
        slice_label_size = dict(label_size)
        slice_label_size[sliced_axis] = width

        # 首片无常驻累加器；自第二片起，累加器（上一片产出的完整输出）
        # 在整片求值期间驻留，片内可用峰值上限相应扣减。
        held = 0 if k == 0 else output_size
        slice_limit = limit - held
        try:
            plan_k = _plan_core(
                tuple(slice_specs),
                slice_label_size,
                label_count,
                slice_limit,
                budget,
                labels_order,
            )
        except InfeasibleError as e:
            raise type(e)(
                "切片 %d（轴 %r 坐标 [%d,%d)，片内峰值预算 %d）规划失败：%s"
                % (k, sliced_axis, lo, lo + width, slice_limit, e)
            ) from e
        except BudgetExhaustedError as e:
            raise BudgetExhaustedError(
                "切片 %d（轴 %r 坐标 [%d,%d)）规划预算耗尽：%s"
                % (k, sliced_axis, lo, lo + width, e)
            ) from e
        # 片内被切轴必须被完全收缩：部分输出轴恒等于最终输出轴。
        if plan_k.output_axes != output_axes:
            raise TensorPlanningError(
                "内部不变量被破坏：切片 %d 的部分输出轴 %r != 最终输出轴 %r"
                % (k, plan_k.output_axes, output_axes)
            )
        sliced_plans.append(plan_k)
        total_mult += plan_k.total_mult_count
        peak_overall = max(peak_overall, held + plan_k.peak_memory)

    total_add = (num_slices - 1) * output_size
    return SlicedPlan(
        tensors=tuple(specs),
        memory_limit=limit,
        sliced_axis=sliced_axis,
        slice_size=slice_size,
        num_slices=num_slices,
        slices=tuple(sliced_plans),
        total_mult_count=total_mult,
        total_add_count=total_add,
        peak_memory=peak_overall,
        output_size=output_size,
        output_axes=output_axes,
        output_shape=output_shape,
    )


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
    _, out_axes, tl, tr, _contracted, _muls, _order = tree
    return "(%s x %s)->[%s]" % (
        format_tree(tl),
        format_tree(tr),
        ",".join(out_axes) if out_axes else "scalar",
    )

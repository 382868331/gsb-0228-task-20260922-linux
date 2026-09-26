"""固定输入的奇指标收缩演示（仅标准库，离线现场计算）。"""

import itertools

from tensor_peak import (
    InvalidProblemError,
    TensorSpec,
    execute_plan,
    find_plan,
)


def scalar_value(spec, index):
    base = sum(ord(char) for char in spec.name) % 7 + 1
    return base + sum(
        (position + 1) * (value + 1)
        for position, value in enumerate(index)
    )


def direct_scalar(specs):
    sizes = {}
    counts = {}
    labels_order = []
    parity = {}
    for spec in specs:
        for axis, size, odd in zip(spec.axes, spec.shape, spec.parity):
            sizes[axis] = size
            counts[axis] = counts.get(axis, 0) + 1
            parity[axis] = odd
            if axis not in labels_order:
                labels_order.append(axis)
    source = tuple(axis for spec in specs for axis in spec.axes)
    target = tuple(
        axis
        for axis in labels_order
        if counts[axis] == 2
        for _ in range(2)
    )
    remaining = {}
    for position, axis in enumerate(target):
        remaining.setdefault(axis, []).append(position)
    target_positions = [remaining[axis].pop(0) for axis in source]
    odd_swaps = 0
    for i, left_axis in enumerate(source):
        for j in range(i + 1, len(source)):
            right_axis = source[j]
            if (
                parity[left_axis]
                and parity[right_axis]
                and target_positions[i] > target_positions[j]
            ):
                odd_swaps += 1
    sign = -1 if odd_swaps % 2 else 1

    total = 0
    for values in itertools.product(
        *(range(size) for axis, size in sizes.items() if counts[axis] == 2)
    ):
        point = dict(
            zip(
                [axis for axis in sizes if counts[axis] == 2],
                values,
            )
        )
        product = 1
        for spec in specs:
            product *= scalar_value(
                spec, tuple(point[axis] for axis in spec.axes)
            )
        total += product
    return sign * total, odd_swaps


def show_plan(title, specs, limit):
    print("=" * 72)
    print(title)
    print("=" * 72)
    plan = find_plan(specs, limit)
    print("括号树:", plan.tree_text)
    for step in plan.steps:
        print(
            "步骤%d: %s x %s -> %s" %
            (step.index, step.lhs.name, step.rhs.name, step.output_name)
        )
        print("  操作轴排列:", step.operation_axes)
        print(
            "  左置换 %s；右置换 %s；奇-奇交换 %d 次；符号 %+d"
            % (
                step.lhs_permutation,
                step.rhs_permutation,
                step.odd_swaps,
                step.sign,
            )
        )
    got = execute_plan(plan)
    want, canonical_swaps = direct_scalar(specs)
    print("计划累计符号: %+d；规范奇交换: %d" % (plan.parity_sign, canonical_swaps))
    print("执行器结果: %d；独立全求和参考: %d；一致: %s" % (got, want, got == want))
    assert got == want
    print()


def main():
    ring = [
        TensorSpec("A", ("a", "b"), (2, 3), (True, False)),
        TensorSpec("B", ("b", "c"), (3, 2), (False, True)),
        TensorSpec("C", ("c", "d"), (2, 3), (True, False)),
        TensorSpec("D", ("d", "a"), (3, 2), (False, True)),
    ]
    show_plan("四环：两个奇指标 a/c，偶指标 b/d 穿插；低峰值括号", ring, 10**9)
    show_plan("同一四环：收紧内存上限后选择另一括号顺序", ring, 19)

    trace = [
        TensorSpec("A", ("a", "b", "e"), (2, 3, 4), (True, True, False)),
        TensorSpec("B", ("b", "a", "e"), (3, 2, 4), (True, True, False)),
    ]
    show_plan("迹收缩：a/b 是奇指标，e 为穿插的偶收缩轴", trace, 10**9)

    negative = [
        TensorSpec("A", ("a", "e", "b"), (2, 3, 2), (True, False, True)),
        TensorSpec("C", ("c", "b"), (2, 2), (True, True)),
    ]
    plan = find_plan(negative, 10**9)
    print("=" * 72)
    print("保留轴案例：a,e,c 按规范顺序输出，收缩 b 前需一次奇-奇交换")
    print("=" * 72)
    print("输出轴:", plan.output_axes, "累计符号:", plan.parity_sign)
    print("结果形状:", plan.output_shape)
    assert plan.output_axes == ("a", "e", "c")
    assert plan.parity_sign == -1

    print()
    print("异常路径：同一指标在两个输入中的奇偶标记冲突")
    try:
        find_plan(
            [
                TensorSpec("A", ("a",), (2,), (True,)),
                TensorSpec("B", ("a",), (2,), (False,)),
            ],
            10**9,
        )
    except InvalidProblemError as error:
        print("[捕获 %s] %s" % (type(error).__name__, error))


if __name__ == "__main__":
    main()

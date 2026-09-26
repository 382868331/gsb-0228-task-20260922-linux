"""可选奇指标标记的数值、符号和异常测试。"""

import itertools
import unittest

from tensor_peak import (
    InvalidProblemError,
    TensorSpec,
    execute_plan,
    find_plan,
)


def make_nested(spec):
    base = sum(ord(char) for char in spec.name) % 7 + 1

    def build(axis):
        if axis == len(spec.shape):
            return 0
        return [build(axis + 1) for _ in range(spec.shape[axis])]

    result = build(0)
    for index in itertools.product(*(range(size) for size in spec.shape)):
        target = result
        for position in index[:-1]:
            target = target[position]
        target[index[-1]] = base + sum(
            (position + 1) * (value + 1)
            for position, value in enumerate(index)
        )
    return result


def nested_get(array, index):
    for position in index:
        array = array[position]
    return array


def direct_signed(specs, output_axes):
    inputs = {spec.name: make_nested(spec) for spec in specs}
    sizes = {}
    parity = {}
    counts = {}
    for spec in specs:
        for axis, size, odd in zip(spec.axes, spec.shape, spec.parity):
            sizes[axis] = size
            parity[axis] = odd
            counts[axis] = counts.get(axis, 0) + 1
    labels_order = []
    for spec in specs:
        for axis in spec.axes:
            if axis not in labels_order:
                labels_order.append(axis)
    source_tokens = tuple(axis for spec in specs for axis in spec.axes)
    target_tokens = tuple(output_axes) + tuple(
        axis
        for axis in labels_order
        if counts[axis] == 2
        for _ in range(2)
    )
    odd_swaps = 0
    target_positions = {}
    for position, axis in enumerate(target_tokens):
        target_positions.setdefault(axis, []).append(position)
    remaining = {
        axis: list(positions)
        for axis, positions in target_positions.items()
    }
    positions = [remaining[axis].pop(0) for axis in source_tokens]
    for i, left_axis in enumerate(source_tokens):
        if not parity[left_axis]:
            continue
        for j in range(i + 1, len(source_tokens)):
            right_axis = source_tokens[j]
            if parity[right_axis] and positions[i] > positions[j]:
                odd_swaps += 1
    sign = -1 if odd_swaps % 2 else 1

    contracted_axes = [axis for axis, count in counts.items() if count == 2]
    result = {}
    enum_axes = list(output_axes) + contracted_axes
    for values in itertools.product(*(range(sizes[axis]) for axis in enum_axes)):
        point = dict(zip(enum_axes, values))
        product = 1
        for spec in specs:
            product *= nested_get(
                inputs[spec.name],
                tuple(point[axis] for axis in spec.axes),
            )
        key = tuple(point[axis] for axis in output_axes)
        result[key] = result.get(key, 0) + sign * product
    if not output_axes:
        return result[()]

    def zeros(axis):
        if axis == len(output_axes) - 1:
            return [0] * sizes[output_axes[axis]]
        return [zeros(axis + 1) for _ in range(sizes[output_axes[axis]])]

    nested = zeros(0)
    for index, value in result.items():
        target = nested
        for position in index[:-1]:
            target = target[position]
        target[index[-1]] = value
    return nested


class TestParity(unittest.TestCase):
    def test_odd_ring_different_bracket_orders_match_direct_reference(self):
        specs = [
            TensorSpec("A", ("a", "b"), (2, 3), (True, False)),
            TensorSpec("B", ("b", "c"), (3, 2), (False, True)),
            TensorSpec("C", ("c", "d"), (2, 3), (True, False)),
            TensorSpec("D", ("d", "a"), (3, 2), (False, True)),
        ]
        plans = [find_plan(specs, 10**9), find_plan(specs, 19)]
        self.assertNotEqual(plans[0].tree_text, plans[1].tree_text)
        expected = direct_signed(specs, ())
        signs = set()
        recorded = set()
        for plan in plans:
            signs.add(plan.parity_sign)
            recorded.add(
                sum(step.odd_swaps for step in plan.steps) % 2
            )
            self.assertEqual(execute_plan(plan), expected)
            self.assertEqual(plan.output_axes, ())
        self.assertEqual(signs, {1})
        self.assertEqual(recorded, {0})
        self.assertIn(
            1,
            [step.odd_swaps for plan in plans for step in plan.steps],
        )

    def test_odd_swap_with_even_output_axis_is_negative(self):
        specs = [
            TensorSpec("A", ("a", "e", "b"), (2, 3, 2), (True, False, True)),
            TensorSpec("C", ("c", "b"), (2, 2), (True, True)),
        ]
        plan = find_plan(specs, 10**9)
        self.assertEqual(plan.output_axes, ("a", "e", "c"))
        self.assertEqual(plan.parity_sign, -1)
        self.assertEqual(plan.steps[0].odd_swaps, 1)
        self.assertEqual(execute_plan(plan), direct_signed(specs, plan.output_axes))

    def test_trace_like_pair_records_two_odd_swaps(self):
        specs = [
            TensorSpec("A", ("a", "b", "e"), (2, 3, 4), (True, True, False)),
            TensorSpec("B", ("b", "a", "e"), (3, 2, 4), (True, True, False)),
        ]
        plan = find_plan(specs, 10**9)
        self.assertEqual(plan.output_axes, ())
        self.assertEqual(plan.steps[0].odd_swaps, 2)
        self.assertEqual(plan.parity_sign, 1)
        self.assertEqual(execute_plan(plan), direct_signed(specs, ()))

    def test_default_parity_unchanged_and_plan_metadata_present(self):
        specs = [
            TensorSpec("A", ("i", "k"), (2, 3)),
            TensorSpec("B", ("k", "j"), (3, 2)),
        ]
        plan = find_plan(specs, 10**9)
        self.assertEqual(plan.tensors[0].parity, (False, False))
        self.assertEqual(plan.parity_sign, 1)
        self.assertTrue(all(step.sign == 1 for step in plan.steps))
        self.assertTrue(all(step.odd_swaps == 0 for step in plan.steps))

    def test_invalid_parity(self):
        with self.assertRaisesRegex(InvalidProblemError, "parity/axes"):
            TensorSpec("A", ("a", "b"), (2, 2), (True,))
        with self.assertRaisesRegex(InvalidProblemError, "必须是 bool"):
            TensorSpec("A", ("a",), (2,), (1,))
        with self.assertRaisesRegex(InvalidProblemError, "奇偶标记冲突"):
            find_plan(
                [
                    TensorSpec("A", ("a",), (2,), (True,)),
                    TensorSpec("B", ("a",), (2,), (False,)),
                ],
                10**9,
            )

if __name__ == "__main__":
    unittest.main()

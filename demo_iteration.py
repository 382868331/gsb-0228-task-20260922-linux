"""混合精度收缩规划演示：离线、确定性、真实计算。

运行：python demo_iteration.py
"""

from decimal import Decimal

from tensor_peak import (
    BudgetExhaustedError,
    ConstraintsInfeasibleError,
    TensorSpec,
    find_precision_plan,
)


def make_problem():
    specs = [
        TensorSpec("A", ("i", "k"), (2, 3)),
        TensorSpec("B", ("k", "j"), (3, 2)),
        TensorSpec("C", ("j", "l"), (2, 4)),
    ]
    bounds = {"A": Decimal(1), "B": Decimal(1), "C": Decimal(1)}
    input_errors = {
        "A": Decimal("0.0001"),
        "B": Decimal(0),
        "C": Decimal(0),
    }
    return specs, bounds, input_errors


def print_plan(title, plan):
    print(title)
    print("  括号树:", plan.tree_text)
    print(
        "  成本:", plan.total_cost,
        " 乘法数:", plan.total_mult_count,
        " 峰值元素:", plan.peak_memory,
    )
    print("  最终绝对误差保守界:", plan.final_error_bound)
    print("  上限:", plan.error_limit)
    for step in plan.steps:
        print(
            "  步骤%d %s x %s -> %s | 精度=%s | 乘法=%d 成本=%d | 峰值=%d"
            % (
                step.index,
                step.lhs.name,
                step.rhs.name,
                step.output_name,
                step.precision.name,
                step.mult_count,
                step.step_cost,
                step.peak_at_step,
            )
        )
        print(
            "    K=%d gamma=%s | 幅度=%s | 传入误差=%s | 舍入=%s | 输出误差=%s"
            % (
                step.contraction_length,
                step.gamma_n,
                step.nominal_magnitude,
                step.input_error_bound,
                step.rounding_error_bound,
                step.output_error_bound,
            )
        )


def main():
    specs, bounds, input_errors = make_problem()

    normal = find_precision_plan(
        specs,
        memory_limit=100,
        error_limit=Decimal("0.007"),
        value_bounds=bounds,
        input_error_bounds=input_errors,
    )
    print_plan("正常结果：在内存和误差约束内选择括号顺序与混合精度", normal)

    strict = find_precision_plan(
        specs,
        memory_limit=100,
        error_limit=Decimal("0.0003"),
        value_bounds=bounds,
        input_error_bounds={name: Decimal(0) for name in bounds},
    )
    print_plan("更紧误差结果：需要高档精度", strict)

    print("边界结果 1：证明无可行计划（不是样本偶然失败）")
    try:
        find_precision_plan(
            specs,
            memory_limit=100,
            error_limit=Decimal("0.000001"),
            value_bounds=bounds,
            input_error_bounds=input_errors,
        )
    except ConstraintsInfeasibleError as exc:
        print(" ", type(exc).__name__, "-", exc)

    print("边界结果 2：搜索预算不足（不声称无解）")
    try:
        find_precision_plan(
            specs,
            memory_limit=100,
            error_limit=Decimal("0.000001"),
            value_bounds=bounds,
            input_error_bounds=input_errors,
            search_budget=1,
        )
    except BudgetExhaustedError as exc:
        print(" ", type(exc).__name__, "-", exc)


if __name__ == "__main__":
    main()

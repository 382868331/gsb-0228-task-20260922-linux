"""本轮迭代演示：括号顺序 × 双档浮点精度联合规划。

运行：python demo_iteration.py
仅标准库、完全离线；真实执行子集 DP 与 Decimal 逐步舍入对照，
正常情况下远小于 90 秒。
"""

from fractions import Fraction

from tensor_peak import (
    FP32,
    FP64,
    TensorSpec,
    BudgetExhaustedError,
    ErrorLimitInfeasibleError,
    decimal_rounding_audit,
    find_precision_plan,
)


def show_plan(title, plan):
    print("=" * 68)
    print(title)
    print("括号树   :", plan.tree_text)
    print("总成本   :", plan.total_cost,
          "（fp32 乘法权重 1，fp64 权重 4）")
    print("峰值驻留 :", plan.peak_memory, "标量元素（上限",
          plan.memory_limit, "）")
    print("最终误差界: %.6e（上限 %s）" % (
        float(plan.final_abs_error), plan.error_limit))
    for step in plan.steps:
        print(
            "  T%d %s x=%d 项数K=%d 乘法=%d 成本=%d 峰值=%d"
            " 输入误差=%.3e 舍入界=%.3e 总误差=%.3e 幅度界=%.3f"
            % (
                step.index,
                step.precision.name,
                step.mult_count,
                step.dot_length,
                step.mult_count,
                step.step_cost,
                step.peak_at_step,
                float(step.input_error_contribution),
                float(step.rounding_error),
                float(step.total_abs_error),
                float(step.output_stored_bound),
            )
        )


def main():
    # 三矩阵链 A(i,k) B(k,j) C(j,l)，输入仅给出元素绝对值上界与已有误差。
    specs = [
        TensorSpec("A", ("i", "k"), (8, 10), abs_bound=Fraction(3, 2)),
        TensorSpec("B", ("k", "j"), (10, 8), abs_bound=Fraction(2)),
        TensorSpec("C", ("j", "l"), (8, 10), abs_bound=Fraction(5, 2)),
    ]
    mem_cap = 10**9

    loose = find_precision_plan(specs, mem_cap, Fraction(1, 10**2))
    show_plan("正常结果 1：宽松误差上限（倾向便宜的 fp32）", loose)

    tight = find_precision_plan(specs, mem_cap, Fraction(4, 10**4))
    show_plan("正常结果 2：收紧误差上限（自动混入 fp64）", tight)

    # Decimal 逐步模拟二进制舍入（对抗性非 2 的幂填充），核对保守界。
    print("=" * 68)
    print("Decimal 逐步舍入对照（填充 1/3、1/7、1/5，真实触发舍入）")
    audit_specs = [
        TensorSpec("A", ("i", "k"), (2, 17), abs_bound=10),
        TensorSpec("B", ("k", "j"), (17, 2), abs_bound=10),
        TensorSpec("C", ("j", "l"), (2, 17), abs_bound=10),
    ]
    audit_plan = find_precision_plan(audit_specs, mem_cap, Fraction(1, 10))
    reports = decimal_rounding_audit(
        audit_plan,
        fill_values={"A": Fraction(1, 3), "B": Fraction(1, 7),
                     "C": Fraction(1, 5)},
    )
    for idx, observed, bound, ratio in reports:
        print("  T%d 实测舍入误差 %.6e <= 账本先验界 %.6e（占用界的 %.4f%%）"
              % (idx, observed, bound, float(ratio) * 100))

    # 边界 1：误差上限严到任何顺序×精度组合都不满足 —— 证明无解。
    print("=" * 68)
    print("边界 1：误差上限不可能满足（穷举证明）")
    try:
        find_precision_plan(specs, mem_cap, Fraction(1, 10**14))
    except ErrorLimitInfeasibleError as exc:
        print("  捕获 ErrorLimitInfeasibleError:", str(exc)[:72], "...")

    # 边界 2：搜索预算不足 —— 与“无解”严格分开。
    print("=" * 68)
    print("边界 2：搜索预算耗尽（未证明无解）")
    try:
        find_precision_plan(specs, mem_cap, Fraction(1), search_budget=1)
    except BudgetExhaustedError as exc:
        print("  捕获 BudgetExhaustedError:", str(exc)[:72], "...")
    print("=" * 68)
    print("完成。")


if __name__ == "__main__":
    main()

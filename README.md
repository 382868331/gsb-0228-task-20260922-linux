# 张量收缩的峰值内存规划（contraction_planner）

为离线科学计算编译工程提供的可复用 Python 模块：给定最多 **10 个稠密张量**的
维度标签与大小，求在给定**内存标量上限**内、**标量乘法次数最少**的两两收缩
（括号化）计划，并输出括号树、每步形状与峰值内存证据。

- Python **3.14.7**，仅标准库，零第三方依赖，完全离线。
- 不调用任何第三方同类收缩优化实现；核心算法为本仓库内的子集 DP。
- 所有成本/内存量均为**精确整数运算**（`int` 无溢出问题）。

## 运行命令

```bash
# 必要测试（30 个用例，约 1–2 秒）
python -m unittest discover -s tests -v

# 固定输入演示（真实规划 + 真实数值执行 + 一个实际触发的拒绝，约 0.05 秒）
python demo.py
```

两条命令都不需要网络、账号或任何额外安装（仅有 Python 3.14 即可）。

## 问题模型（本模块实现的契约子集）

- 每个维度标签（非空字符串）在所有输入张量中**恰好出现 1 次或 2 次**：
  - 出现 1 次：输出轴（保留到最终结果）；
  - 出现 2 次：收缩轴（在某次两两收缩中求和）。
- 同一标签在各处的维度大小必须一致，且为正整数。
- 只允许**两两收缩**；一次收缩的两个操作数**必须至少共享一个轴**，
  即**禁止外积**。不连通的网络（任何收缩树都需要外积才能合并）被判无解。
- 内存模型（峰值按标量个数计）：
  - 某一步收缩瞬间，**两个操作数 + 新输出**同时驻留，计入峰值；
  - 收缩完成后立即释放两个已消耗的操作数（临时量），发布新输出；
  - 原始输入按需载入（conservative：只在使用它的那一步载入），不长期占内存；
  - 两个子树谁先构造也可选择，构造第二个子树期间第一个子树的结果驻留。
- 一次收缩的标量乘法次数 =
  `Π(被收缩轴大小) × Π(结果保留轴大小)`。

这是 einsum 收缩路径优化的一个**明确数学子集**，不宣称兼容 NumPy/PyTorch
einsum 的完整语法或行业标准（不支持广播、对角/迹、省略号、一标签多次以上
出现等）。

## 公开接口

```python
from contraction_planner import Tensor, plan_contraction

tensors = [
    Tensor(axes=("i", "k"), shape=(2, 3), name="A"),
    Tensor(axes=("k", "l"), shape=(3, 4), name="B"),
    Tensor(axes=("l", "j"), shape=(4, 5), name="C"),
]
plan = plan_contraction(tensors, memory_cap=1_000_000)
```

`plan_contraction(tensors, memory_cap, *, node_budget=None) -> ContractionPlan`

- `tensors`：`Tensor` 列表/元组，长度 1..10；
- `memory_cap`：正整数标量上限；
- `node_budget`：可选正整数，检查的候选收缩数上限（搜索预算）。

返回的不可变数据结构（`dataclass`）：

| 字段 | 含义 |
| --- | --- |
| `inputs` | 每个输入的索引、名字、轴、形状、标量数 |
| `output_axes` / `output_shape` | 最终输出轴顺序与形状 |
| `total_multiplications` | 计划总标量乘法次数（上限内最少） |
| `peak_memory` | 计划峰值驻留标量数（含完整证据） |
| `peak_evidence` | 达到峰值的步骤号、总量与当时驻留明细 `Resident` |
| `tree` | 嵌套括号树，叶子为输入索引，如 `(2, (1, 0))` |
| `bracket` | 同一棵树的字符串形式，如 `(T2,(T1,T0))` |
| `steps` | 按执行顺序的每一步 `PlanStep` |

`PlanStep` 含：操作数引用（`OperandRef`：输入或上一步产物）、双方轴、
`contracted_axes`、输出轴/形状、本步乘法数、`resident_total` 与当时驻留的
`Resident` 明细（可直接验算 `sum(r.size for r in residents)`）。

数值核对可用内置小执行器（不是规划算法的一部分，仅供验证）：

```python
from contraction_planner import DenseArray, execute_plan
arrays = [DenseArray.from_function(t.axes, t.shape, fn) for t in tensors]
result, observed_peak = execute_plan(plan, arrays)
```

执行器按计划惰性载入输入、即时释放消耗项，并逐步断言其观测到的驻留标量数
与计划记录一致（不一致会抛 `AssertionError`）。

## 错误语义

| 异常 | 触发条件 |
| --- | --- |
| `TypeError` | 参数类型错误；**布尔值冒充整数一律拒绝**（含 `shape`、`memory_cap`、`node_budget`） |
| `ValueError` | 数值为 **NaN / ±Infinity**（浮点）等非有限值 |
| `InvalidProblemError` | 张量数越界、空轴、轴数与形状数不符、标签出现 ≥3 次、同标签维度不一致、大小 ≤0 等；错误信息包含 `tensors[i]` 与轴名，便于定位 |
| `InfeasiblePlanError` | 子集 DP **穷尽并证明**不存在满足上限的计划（错误信息给出可证明的最小峰值）；也用于不连通网络 |
| `SearchBudgetExhaustedError` | `node_budget` 个候选已检查完仍未得出结论——**预算耗尽不等于证明无解**，与上一个异常严格区分 |

规划器是纯函数：不修改输入、不保留跨调用状态、不做 I/O；不存在“部分变更”。

## 算法与设计取舍

- **子集 DP**：对每个输入子集（bitmask）保存其收缩方式的
  **(总乘法数, 峰值) Pareto 前沿**，合并时枚举所有非平凡二分划与前沿组合。
  同一子集的输出张量大小固定，峰值关于子方案峰值单调，故被支配点可安全剪枝；
  再按 `memory_cap` 过滤不可行点。
- 对每个内部节点同时计算“先构造左子树/先构造右子树”两种执行序的峰值并取优，
  执行序与树一起记录，保证输出计划真实可达所报峰值。
- 复杂度：子集数 2ⁿ，每子集枚举划分与 Pareto 组合；n ≤ 10 时纯 Python
  亚秒级完成（实测随机网络约 0.02 s，小维度）。
- Pareto 前沿只保留精确整数点，不做近似或剪枝容差；最优性是**精确**的。
- 输出轴顺序采用确定性规范序（按最低输入索引首次出现的顺序），执行器每步
  做轴重排以对齐该顺序。

### 浮点误差口径

规划结果（成本、峰值、形状）全部是整数，无浮点误差。仅在使用 `execute_plan`
且数组数据为 `float` 时涉及浮点：每个输出元素是若干乘积按固定行主序**顺序
累加**的 IEEE 754 binary64 结果，舍入误差随求和项数线性增长，经验上界约为

```
|computed - exact| ≲ (n_terms - 1) · 2^-53 · Σ|a·b|
```

模块**不使用任何“宽容差”来掩盖边界错误**：整数演示数据按精确相等断言；
`DenseArray.allclose` 默认 `rtol=atol=1e-12`（仅供调用方自行比较时使用）。

## 测试与独立性

- `tests/reference.py`：**独立参考实现**，自行重述成本与内存数学并枚举
  n≤6 的全部二叉树，**从不导入被测核心**；规划器结果与其逐一对照
  （含对所有不同峰值阈值的上限扫描、无解边界）。
- `tests/test_planner.py`：参考对照（n=2..6 多网络）、共享多个轴、
  不连通拒绝、成本/内存权衡（770/203 vs 812/182）、预算耗尽与证明无解分离、
  封死 socket 后结果不变（离线保证）、树计数（链 Catalan、星 (n−1)!）。
- `tests/test_executor.py`：小张量真实执行，期望值由测试内独立多重循环
  暴力 einsum 给出（精确整数相等）；往返一致只作补充。
- `tests/test_validation.py`：bool、NaN/Infinity、越界、定位信息、纯度等。

## 已知限制

- 张量数硬上限 10（接口边界；枚举/DP 按此规模设计）。
- 仅支持标签出现 1 或 2 次的稠密收缩；无广播、复数以外的特殊语义、稀疏、
  自动微分步融合；不做布局/缓存行级真实字节建模，内存单位是标量数。
- 执行器面向**小张量核对**（平面 Python list），不是高性能运行时。
- 演示规模刻意很小；上限规模（10 张量）由测试覆盖而非 demo 穷举。

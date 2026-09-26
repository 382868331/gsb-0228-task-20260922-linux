# tensor_peak — 张量收缩的峰值内存规划

为离线科学计算编译工程提供的**可复用单文件模块**：给定若干稠密张量的维度
标签与每维大小，在“只允许两两收缩、禁止外积”的约束下，求出**满足给定内存
上限、标量乘法数最少**的收缩计划，并给出括号树、每步形状与峰值内存证据。

- **Python 版本**：Python 3.14.7（仅使用标准库；3.10+ 亦可运行）。
- **依赖**：无第三方依赖，安装 Python 后即可运行，**完全离线**（不访问网络、
  账号、数据库、中间件、云 API，不需要 Docker）。
- **数值口径**：规划器只处理形状与计数，全部结果为**精确整数**（标量乘法数、
  标量元素数），不存在浮点误差，因此输出值之间用 `==` 比较，无任何容差。
  演示中的逐元素核对同样使用确定性整数数据精确比对。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（含独立参考交叉验证）
python demo.py                            # 固定输入演示（约 0.1 秒）
python demo_iteration.py                  # 本轮：精度联合规划演示（<1 秒）
```

## 环境准备（Linux 原生 Python 3.14.7）

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python --version        # 应为 Python 3.14.7
```

仅使用标准库，`requirements.txt` 无第三方依赖，不需要 `pip install`，
全程离线（无 Docker、外部服务、网络或真实凭据）。

## 本轮迭代：括号顺序 × 浮点精度联合规划

在原“只按内存选括号顺序”之外，新增**每步两档模拟浮点精度**的联合规划：
同时选择括号顺序与每步精度，在峰值元素数上限与**最终绝对误差上限**内
最小化成本。规划只消费每个输入的**元素绝对值上界**与**已有绝对误差上界**，
不读取真实大张量数据。

```python
from tensor_peak import (
    TensorSpec, find_precision_plan, FP32, FP64, PRECISION_LEVELS,
    ErrorLimitInfeasibleError, PrecisionInfeasibleError,
    decimal_rounding_audit,
)

A = TensorSpec("A", ("i", "k"), (8, 10), abs_bound=1.5, abs_error=0)
plan = find_precision_plan([A, B, C], memory_limit=10**9,
                           error_limit=Fraction(1, 1000))
```

- 精度档（成本权重可通过自定义 `PrecisionLevel` 调整）：

| 档 | 有效位 p | 单位舍入 u | 每次乘法成本 |
|---|---|---|---|
| `FP32` | 24 | 2⁻²⁴ | 1 |
| `FP64` | 53 | 2⁻⁵³ | 4 |

- `TensorSpec` 新增 `abs_bound`（默认 1）与 `abs_error`（默认 0），
  接受 `int`/`Fraction`/`Decimal`/有限 `float`（负数与 NaN/Infinity 拒绝）。
- `find_precision_plan(tensors, memory_limit, error_limit, *,
  search_budget=None, levels=PRECISION_LEVELS) -> PrecisionContractionPlan`：
  返回每步精度、内存驻留证据与逐步误差账本；按 `total_cost` 最小优化，
  平局按峰值、误差。
- `PrecisionContractStep` 字段：`precision`、`dot_length`（点积项数 K）、
  `mult_count`、`step_cost`、`peak_at_step`、`lhs_bound`/`rhs_bound`、
  `term_mag_sum`、`input_error_contribution`、`rounding_error`、
  `total_abs_error`、`output_stored_bound`/`output_true_bound`。

### 误差口径（标准点积舍入界，保守先验传播）

一次收缩把每个输出元素视为 **K 项点积**（K = 收缩轴长度之积）。
设两侧元素的存储幅度上界为 Mₗ、Mᵣ，已有绝对误差上界为 eₗ、eᵣ：

- 项幅度和 S = K·Mₗ·Mᵣ；
- 输入已有误差传播（一阶 + 交叉项）：
  E_in = K·(eₗ·Mᵣ + eᵣ·Mₗ + eₗ·eᵣ)；
- 舍入界（每项一次乘法一次舍入的标准 γ_K 界）：
  γ_K = K·u/(1−K·u)，E_round = γ_K·S；
- 输出存储幅度上界 M′ = S + E_round；输出总绝对误差 e′ = E_in + E_round。

全部传播用 `fractions.Fraction` **精确有理运算**，不引入二进制浮点近似。
当 K·u ≥ 1 时 γ_K 无界，该档精度对此步不可行（搜索中排除；若所有档都
不可行，穷举后抛 `PrecisionInfeasibleError`）。界不满足的计划不会因为
“小样本碰巧算得准”而被放行：判定只依据先验界。

### Decimal 逐步舍入对照

`decimal_rounding_audit(plan, *, fill_scales=None, fill_values=None, signs=())`
在不读取真实张量的前提下，用不超过 `abs_bound` 的确定性常量填充，以
`Decimal` 高有效数字模拟 **p 位二进制就近偶舍入**逐步执行计划，返回每步
实测舍入误差与账本 `rounding_error` 的对照（实测恒 ≤ 账本界）。该函数
仅用于核对界的保守性，不参与规划。

### 无解与预算不足（仍然严格分开）

| 异常 | 含义 |
|---|---|
| `PrecisionInfeasibleError` | 某步 K·u≥1，对全部给定精度都不存在有界误差 |
| `MemoryLimitInfeasibleError` | 穷举证明峰值内存上限不可行 |
| `ErrorLimitInfeasibleError` | 内存可行，但穷举全部顺序×精度后最终误差界仍超上限 |
| `BudgetExhaustedError` | `search_budget` 耗尽，**未证明无解** |

原 `find_plan`/`pareto_front`（只按内存规划，精确整数）接口与语义保持不变。

## 问题模型与内存口径

- 每个维度标签在全部输入中**恰出现 1 或 2 次**：出现 1 次是输出轴（保留到
  最终结果），出现 2 次是收缩轴（在某次两两收缩中求和消去）。
- 收缩只能发生在**共享至少一个收缩轴**的两个中间结果之间；不共享轴即外积，
  **禁止**。若收缩轴构成的张量网络不连通，则任何合法计划都不存在，直接
  拒绝（`DisconnectedNetworkError`）。
- 一次收缩的标量乘法数 = 两侧所有轴（并集）维大小之积。
- **峰值驻留**以标量元素个数计：一次收缩执行的瞬间，同时驻留
  “两个操作数 + 新输出”，再加上为完成整体括号化而必须留在后台的其它中间
  结果；原始输入可按需载入（不占长期驻留），输出产生后被消费的临时量立即
  释放。每个节点可以选择**先求值左子树还是右子树**以压低峰值。
- `memory_limit` 是对该峰值（标量元素数）的硬上限。

## 公开接口

```python
from tensor_peak import (
    TensorSpec, find_plan, pareto_front, format_tree,
    # 数据结构
    OperandRef, ContractStep, ContractionPlan, FrontPoint, MAX_TENSORS,
    # 异常
    TensorPlanningError, InvalidProblemError, InfeasibleError,
    DisconnectedNetworkError, MemoryLimitInfeasibleError,
    BudgetExhaustedError,
)
```

- `TensorSpec(name: str, axes: tuple[str, ...], shape: tuple[int, ...])`
  不可变值对象；`name` 非空且全局唯一（保留名 `T0..T9` 禁止使用），同一张量
  内轴标签不重复，`shape` 与 `axes` 等长、各项为正整数。
- `find_plan(tensors, memory_limit, *, search_budget=None) -> ContractionPlan`
  - `tensors`：1..10 个 `TensorSpec`；
  - `memory_limit`：正整数峰值上限；
  - `search_budget`：可选正整数，限定候选收缩求值次数。默认 `None`，对
    n≤10 做完整穷举（亚秒级）。
- `pareto_front(tensors, *, search_budget=None) -> tuple[FrontPoint, ...]`
  返回无内存限制时 `(mult_count, peak_memory)` 的 Pareto 前沿（乘法升序、
  峰值严格下降），用于观察成本/内存权衡。
- `format_tree(tree)`：把括号树渲染为如 `(A x B)->[i,j]` 的文本。

`ContractionPlan` 字段：`total_mult_count`、`peak_memory`、`peak_step`、
`tree` / `tree_text`、`steps`、`output_axes` / `output_shape`。每个
`ContractStep` 含左右操作数（`OperandRef`：原始输入或前序中间量）、
`evaluated_first`、`contracted_axes`、输出形状、本步乘法数，以及
`residency`（收缩瞬间完整驻留清单）与 `peak_at_step`，可逐步复核峰值证据。

### 异常语义（搜索预算耗尽 ≠ 证明无解，严格分开）

| 异常 | 含义 |
|---|---|
| `InvalidProblemError`（`ValueError` 子类） | 输入违反契约，错误信息可定位到张量/维度/参数；不静默纠正 |
| `DisconnectedNetworkError`（`InfeasibleError` 子类） | 已证明网络不连通，无任何合法收缩树 |
| `MemoryLimitInfeasibleError`（`InfeasibleError` 子类） | 已完整穷举全部合法树，证明不存在满足上限的计划 |
| `BudgetExhaustedError`（`TensorPlanningError` 子类） | `search_budget` 耗尽、搜索停止；**未证明无解**，增大预算可重试 |

数值输入拒绝 `NaN`/`Infinity`；维大小、上限、预算只接受精确正整数，
**拒绝 `bool`**（尽管 `bool` 是 `int` 子类）与非整数 `float`。搜索是无状态
纯函数：失败只抛异常，不返回任何残缺或部分结果。

## 算法与设计取舍

- **子集 DP**：对每个张量子集保存其全部收缩方式的 Pareto 前沿
  `(累计乘法数, 子计划峰值)`。合并两个子集时枚举左右求值顺序：
  - 先求左：峰值 = `max(左内部峰值, |左结果| + 右内部峰值, |左|+|右|+|输出|)`；
  - 先求右对称，取较小者。
  乘法数为两侧累计加本节点乘法。前沿点在“乘法更少”与“峰值更低”上互不支配。
- 给定 `memory_limit` 时，构建中即剔除任何峰值超限的点。这是**精确剪枝**：
  父集峰值不可能低于已含子集的峰值，因此全集前沿为空等价于穷举性地证明无解；
  这与预算中断完全不同。
- 轴在中间结果中的排列按“标签在原始输入中的首次出现顺序”规范化，使等价计划
  的表示唯一、确定。
- 计划重建阶段按 DP 选择的求值顺序线性化，中间量名 `T0,T1,...` 即产生顺序；
  每步的 `residency` 显式列出该瞬间全部驻留对象，使峰值声明可逐步审计。
- **取舍**：内存单位是“标量元素个数”，不模拟分块、dtype 字节、对齐、
  非原地累加的写放大，也不模拟并行/多流执行；原始输入假定可按需载入、不计入
  长期驻留。这是题面契约下的数学子集，不宣称兼容 einsum/MLIR 等完整行业
  标准（不支持广播、对角、复数轴语义等）。

## 输入上限

- 张量数 1..**10**；每个标签出现 1 或 2 次；维大小为任意正整数（Python 大整数）。
- 子集 DP 复杂度约为 O(3^n · F²)（F 为前沿宽度，本类问题通常很小）；n=10
  稠密网络在普通机器上亚秒级完成。

## 测试与独立参考

- `tests/reference_enum.py`：**独立**全二叉树枚举参考，自行实现子集枚举与
  峰值口径，不导入被测核心。
- `tests/test_tensor_peak.py`：n=1..6 随机连通网络（确定性种子）的前沿、紧
  上限选择、不可行边界与参考逐一比对；含共享多轴、不连通拒绝、成本/内存
  权衡（四环实例：168 乘/峰值 66 与 174 乘/峰值 61 互为 Pareto）、预算耗尽
  与证明无解的区分、断网（封堵 `socket.socket`）下完成规划、非法输入、
  n=10 上限规模等。
- `tests/test_numeric.py`：独立的稠密逐元素执行器按计划逐步收缩，与独立直算
  枚举求和结果做精确 `==` 比对（往返一致只作补充，期望值不来自被测核心）。
- `tests/reference_precision.py`：独立枚举**全部括号顺序 × 每步精度组合**，
  自行书写 γ_K 误差传播与成本/峰值口径，不导入被测核心。
- `tests/test_precision.py`：n=2..5 多档误差上限下与独立全枚举逐一比对；
  混合精度选择、逐步账本恒等式、`Decimal` 对抗性舍入实测 ≤ 账本界、
  K·u≥1 边界、预算耗尽与证明无解分离、非法输入、原接口保持等用例。

## 已知限制

- 不做分块/布局/内核级内存建模，内存只计标量元素驻留。
- 不支持标签出现 ≥3 次、外积、广播、稀疏张量；非法情形直接报错。
- 误差界采用“每项一次乘法一次舍入、K 项求和”的 γ_K 标准先验模型；
  不模拟 FMA 融合、补偿求和、舍入模式切换或硬件特定行为。输入只提供
  统一的元素绝对值上界与已有误差上界，不做逐元素相关结构（相关性只会
  让真实误差更小，故统一界对任何相关结构都保守成立）。
- `decimal_rounding_audit` 的填充执行器面向小张量设计（演示/测试规模），
  用于核对界；规划本身不执行填充、不随张量数据量变慢。
- 前沿可能随极端维大小组合变宽；n≤10 下预算默认关闭即可穷举，必要时可用
  `search_budget` 显式管理（耗尽时得到的是 `BudgetExhaustedError` 而非无解
  结论）。

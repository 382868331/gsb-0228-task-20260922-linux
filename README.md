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
python -m venv .venv
. .venv/bin/activate
python -m unittest discover -s tests -v
python demo_iteration.py
```

本项目只使用 Python 标准库；`requirements.txt` 为空，创建 venv 后无需安装
第三方包。`demo_iteration.py` 使用确定性小整数现场计算，通常少于 1 秒，
约束为 90 秒内完成。

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
    TensorSpec, find_plan, pareto_front, format_tree, execute_plan,
    # 数据结构
    OperandRef, ContractStep, ContractionPlan, FrontPoint, MAX_TENSORS,
    # 异常
    TensorPlanningError, InvalidProblemError, InfeasibleError,
    DisconnectedNetworkError, MemoryLimitInfeasibleError,
    BudgetExhaustedError,
)
```

- `TensorSpec(name, axes, shape, parity=None)`
  不可变值对象；`name` 非空且全局唯一（保留名 `T0..T9` 禁止使用），同一张量
  内轴标签不重复，`shape` 与 `axes` 等长、各项为正整数。可选 `parity` 是与
  `axes` 等长的 `bool` 序列：`True` 表示奇指标，`False`/`None` 表示偶指标；
  同一标签在所有输入中的固定奇偶必须一致。
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
`evaluated_first`、`contracted_axes`、输出形状、本步乘法数、
`operation_axes`（本步规范操作轴排列，收缩轴成对出现）、
`lhs_permutation` / `rhs_permutation` / `output_permutation`、
`odd_swaps`（仅统计两个奇指标之间的交换次数）、`sign`，以及
`residency`（收缩瞬间完整驻留清单）与 `peak_at_step`，可逐步复核峰值和符号。
`ContractionPlan.axis_parity` 保存标签到固定奇偶的映射，`parity_sign` 是全计划
`step.sign` 的乘积。

`execute_plan(plan, inputs=None)` 使用标准嵌套 `list` 稠密执行计划；省略
`inputs` 时生成确定性小整数输入。执行器会显式应用每步记录的轴置换和符号，
返回嵌套 `list`，完全收缩为标量时返回数值。

### 奇指标符号口径

- 未提供 `parity` 时所有指标均为偶指标，`odd_swaps=0`、`sign=1`，规划成本、
  峰值和旧接口行为不变。
- 每步把左右操作数的轴置换为该步的规范操作排列：输出轴按整网首次出现顺序，
  收缩轴在目标排列中以配对形式出现。置换中的逆序只有两端都是奇指标时才计数；
  奇-奇交换奇数次乘 `-1`，偶数次为 `+1`；偶指标穿插不改变符号。
- 中间结果始终按整网首次出现顺序规范化，最终输出统一为
  `ContractionPlan.output_axes`，不把转置视为免费重排。不同括号顺序累计到同一
  整网规范排列时必须得到相同数值结果和符号。

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
- 轴在中间结果中的排列按“标签在原始输入中的首次出现顺序”规范化；奇指标符号
  通过当前步骤在整网规范轴令牌序列中的对应位置计算，避免只做局部转置而漏掉
  括号间的结合性符号。
- 计划重建阶段按 DP 选择的求值顺序线性化，中间量名 `T0,T1,...` 即产生顺序；
  每步的 `residency` 显式列出该瞬间全部驻留对象，使峰值声明可逐步审计。
- **取舍**：内存单位是“标量元素个数”，不模拟分块、dtype 字节、对齐、
  非原地累加的写放大，也不模拟并行/多流执行；原始输入假定可按需载入、不计入
  长期驻留。这是题面契约下的数学子集，不宣称兼容 einsum/MLIR 等完整行业
  标准（不支持广播、对角、复数轴语义等）。

## 输入上限

- 张量数 1..**10**；每个标签出现 1 或 2 次；维大小为任意正整数（Python 大整数）。
- 每个指标的奇偶性固定为偶或奇；不支持指标依赖位置/数值改变奇偶，不实现完整
  费米子物理库、Grassmann 代数对象或反交换数据结构，只在稠密数值收缩时计算
  必要置换符号。
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
- `tests/test_numeric.py`：独立的稠密逐元素参考按计划形状枚举求和，做精确
  `==` 比对（往返一致只作补充，期望值不来自被测核心）。
- `tests/test_parity.py`：奇指标规划的独立全指标求和和显式规范置换参考，覆盖
  两种括号顺序、两个奇指标交换、偶指标穿插、迹收缩、输出轴规范化和异常路径。

## 已知限制

- 不做分块/布局/内核级内存建模，内存只计标量元素驻留。
- 不支持标签出现 ≥3 次、外积、广播、稀疏张量；非法情形直接报错。
- 奇偶支持限定为每个指标固定偶/奇的普通有限维稠密小张量；执行器为教学/验证
  使用嵌套 list，不是优化 BLAS/费米子张量库。
- 不推导物理张量本身的阶奇偶、位置相关奇偶、算子共轭或完整 Grassmann 约定；
  调用者必须通过 `parity` 明确给出固定指标奇偶。
- 前沿可能随极端维大小组合变宽；n≤10 下预算默认关闭即可穷举，必要时可用
  `search_budget` 显式管理（耗尽时得到的是 `BudgetExhaustedError` 而非无解
  结论）。

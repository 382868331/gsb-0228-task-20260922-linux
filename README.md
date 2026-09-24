# tensor_peak — 张量收缩的峰值内存规划

为离线科学计算编译工程提供的**可复用单文件模块**：给定若干稠密张量的维度
标签与每维大小，在“只允许两两收缩、禁止外积”的约束下，求出**满足给定内存
上限、标量乘法数最少**的收缩计划，并给出括号树、每步形状与峰值内存证据。
并支持调用者指定**单个可切片维度**：沿该维切块后对每个切片沿用同一规划器，
累计逐切片计算量（含输出累加成本），返回逐切片计划与峰值中间存储。

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
```

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
    TensorSpec, SliceDim, find_plan, plan_slices, pareto_front, format_tree,
    # 数据结构
    OperandRef, ContractStep, ContractionPlan, SlicedPlan, FrontPoint,
    MAX_TENSORS,
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
- `SliceDim(axis: str, slices: tuple[int, ...])`：调用者指定的**单个可切片
  维度**。`axis` 是问题中存在的标签，`slices` 是各切片在该轴上的长度（正整数、
  允许不等长，**之和必须等于该维总大小**）。
- `plan_slices(tensors, memory_limit, slice_dim, *, search_budget=None)
  -> SlicedPlan`：沿 `slice_dim` 切块，对**每个切片独立调用旧规划器**
  `find_plan`（同一套子集 DP 与 `memory_limit` 口径），再汇总逐切片计划。
  - 切**收缩轴**（标签出现 2 次）：每片把该轴维长换成块长后照常收缩，各片
    得到形状相同的部分结果，需跨片相加。输出累加成本 =
    `(片数-1) × 输出元素数`，计入 `total_mult_count` 并单列在
    `accumulator_cost`。
  - 切**输出轴**（标签出现 1 次）：各片产出最终输出沿该轴的不相交分块，直接
    拼接，`accumulator_cost = 0`。
  - 逐切片顺序执行，`peak_intermediate_storage` = 各片计划峰值的最大值，
    `peak_slice` 给出达到它的切片下标。`memory_limit` 对每片单独生效；任一片
    不可行（或预算耗尽）即整体抛错，不返回任何部分结果。
  - 本次迭代**不做**共享子表达式跨片重计算，也不做多维/全局切片联合最优；
    每片就是独立的缩小版原问题。
- `pareto_front(tensors, *, search_budget=None) -> tuple[FrontPoint, ...]`
  返回无内存限制时 `(mult_count, peak_memory)` 的 Pareto 前沿（乘法升序、
  峰值严格下降），用于观察成本/内存权衡。
- `format_tree(tree)`：把括号树渲染为如 `(A x B)->[i,j]` 的文本。

`ContractionPlan` 字段：`total_mult_count`、`peak_memory`、`peak_step`、
`tree` / `tree_text`、`steps`、`output_axes` / `output_shape`，以及仅在切块
结果中非空的 `slicing`（普通 `find_plan` 结果恒为 `None`）。每个
`ContractStep` 含左右操作数（`OperandRef`：原始输入或前序中间量）、
`evaluated_first`、`contracted_axes`、输出形状、本步乘法数，以及
`residency`（收缩瞬间完整驻留清单）与 `peak_at_step`，可逐步复核峰值证据；
切块计划里的步骤另带 `sliced_axis` / `slice_index` 标注。

`SlicedPlan` 字段：`slices`（逐切片 `ContractionPlan`）、`total_mult_count`
（含累加成本）、`accumulator_cost`、`peak_intermediate_storage`、
`peak_slice`、`slice_axis`、`slice_sizes`。各切片计划的步骤名在各自命名空间
内独立（都是 `T0,T1,...`）。

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
- `tests/test_slicing.py`：切块功能。逐切片乘法/峰值与独立全树枚举参考逐一
  比对，累计量与峰值按解析关系（各片和、`(k-1)×输出大小` 累加、各片峰值取
  max）核对；覆盖收缩轴切块、输出轴切块（零累加）、单位块、单张量切块等
  正常与边界，以及轴不存在、块长和不符、某片在紧上限下被穷举证明无解、预算
  耗尽不返回部分结果等失败用例；并用独立稠密执行器逐切片执行、跨片累加/拼接，
  对独立直算结果逐元素 `==`。

## 已知限制

- 不做分块/布局/内核级内存建模，内存只计标量元素驻留。
- 不支持标签出现 ≥3 次、外积、广播、稀疏张量；非法情形直接报错。
- 前沿可能随极端维大小组合变宽；n≤10 下预算默认关闭即可穷举，必要时可用
  `search_budget` 显式管理（耗尽时得到的是 `BudgetExhaustedError` 而非无解
  结论）。
- 切块只支持**单个**维度，由调用者给定切块方案；规划器不自动选维、不自动
  分块，不做多维切块。各切片是相互独立的缩小版原问题：不做共享子表达式的
  跨片重计算，也不做跨切片的全局联合最优。峰值中间存储只统计各片收缩计划的
  中间驻留峰值；跨片输出累加器属于输出存储，沿用“原始/输出缓冲不计入中间
  驻留”的既有口径，不并入该峰值，其代价以 `accumulator_cost` 单列。

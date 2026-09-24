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
```

## 可切片收缩轴（`find_sliced_plan`）

允许调用者指定**一个出现两次的收缩轴**进行切块：把该轴坐标区间按固定
片宽划分为若干切片，对每个切片沿用原规划器（子集 DP + Pareto 前沿）
独立求最少乘法计划，各片的部分输出累加为最终结果。

- 每个切片都是一次完整的旧问题：被切轴在该片的两个输入上取相同坐标
  区间（宽度 `w`，最后一片允许更小），在片内即被收缩，因此**部分输出
  轴恒等于最终输出轴**（被切轴不会泄漏到部分输出中）。
- **累计计算量**：`total_mult_count` 为各片标量乘法之和；`total_add_count
  = (切片数 - 1) × 输出元素数`——首片初始化累加器，之后每片对每个输出
  元素做一次标量加法。注意**切块可能增加乘法数**：不含被切轴的子树收缩
  会随切片数重复（例如三环网络不切为 30 乘，沿某轴切 2 片为 32 乘）。
  切块用重复计算与累加加法换取更低的峰值驻留，是否值得由调用者按
  `(乘法数, 加法数, 峰值)` 自行权衡；本迭代不做跨切片的全局联合最优。
- **峰值中间存储**：首片无常驻数据；自第二片起，已产出的完整输出作为
  **常驻累加器**（`输出元素数` 个标量）在该片求值期间始终驻留。
  `peak_memory = max(首片片内峰值, 输出元素数 + 其余各片片内峰值)`。
  `memory_limit` 按此口径约束：规划第 k≥1 片时，片内可用上限自动扣除
  累加器大小（`limit - 输出元素数`）；任一片经穷举无可行计划即抛
  `MemoryLimitInfeasibleError`，消息标注切片编号与坐标区间。
- `slice_size >= 轴维` 时退化为单片（零累加加法）；`slice_size = 1`
  为逐坐标切片。切片数上限 `MAX_SLICES = 64`。

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
    TensorSpec, find_plan, find_sliced_plan, pareto_front, format_tree,
    # 数据结构
    OperandRef, ContractStep, ContractionPlan, SlicedPlan, FrontPoint,
    MAX_TENSORS, MAX_SLICES,
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
- `find_sliced_plan(tensors, memory_limit, sliced_axis, slice_size, *, search_budget=None) -> SlicedPlan`
  - `sliced_axis`：要切片的轴标签，必须是**恰出现 2 次的收缩轴**
    （输出轴不可切）；
  - `slice_size`：正整数片宽，切片数为 `ceil(轴维/片宽)`，最后一片允许
    更小；切片数超过 `MAX_SLICES` 按非法输入拒绝；
  - `search_budget`：每片独立一份预算（各片重新计数）。
- `format_tree(tree)`：把括号树渲染为如 `(A x B)->[i,j]` 的文本。

`ContractionPlan` 字段：`total_mult_count`、`peak_memory`、`peak_step`、
`tree` / `tree_text`、`steps`、`output_axes` / `output_shape`。每个
`ContractStep` 含左右操作数（`OperandRef`：原始输入或前序中间量）、
`evaluated_first`、`contracted_axes`、输出形状、本步乘法数，以及
`residency`（收缩瞬间完整驻留清单）与 `peak_at_step`，可逐步复核峰值证据。

`SlicedPlan` 字段：`sliced_axis`、`slice_size`、`num_slices`、
`slices`（每个元素是该切片的完整 `ContractionPlan`，其 `memory_limit`
已扣除常驻累加器）、`total_mult_count`、`total_add_count`、
`peak_memory`（含常驻累加器的整体峰值）、`output_size`、
`output_axes` / `output_shape`。

### 异常语义（搜索预算耗尽 ≠ 证明无解，严格分开）

| 异常 | 含义 |
|---|---|
| `InvalidProblemError`（`ValueError` 子类） | 输入违反契约，错误信息可定位到张量/维度/参数；不静默纠正 |
| `DisconnectedNetworkError`（`InfeasibleError` 子类） | 已证明网络不连通，无任何合法收缩树 |
| `MemoryLimitInfeasibleError`（`InfeasibleError` 子类） | 已完整穷举全部合法树，证明不存在满足上限的计划 |
| `BudgetExhaustedError`（`TensorPlanningError` 子类） | `search_budget` 耗尽、搜索停止；**未证明无解**，增大预算可重试 |

数值输入拒绝 `NaN`/`Infinity`；维大小、上限、预算、片宽只接受精确正整数，
**拒绝 `bool`**（尽管 `bool` 是 `int` 子类）与非整数 `float`。搜索是无状态
纯函数：失败只抛异常，不返回任何残缺或部分结果（切片规划在全部切片成功后
才一次性构造 `SlicedPlan`，中途失败不产生半更新状态）。

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
- **切片复用旧规划器**：`find_sliced_plan` 不引入新的搜索算法，而是为每个
  切片构造一个缩小后的旧问题（两个属主输入的被切轴维改为片宽并移到轴序
  末尾），逐片调用同一个子集 DP，再把逐片计划、累计乘法、累加加法、含
  常驻累加器的峰值汇总成 `SlicedPlan`。
- **取舍**：内存单位是“标量元素个数”，不模拟分块、dtype 字节、对齐、
  非原地累加的写放大，也不模拟并行/多流执行；原始输入假定可按需载入、不计入
  长期驻留。这是题面契约下的数学子集，不宣称兼容 einsum/MLIR 等完整行业
  标准（不支持广播、对角、复数轴语义等）。

## 输入上限

- 张量数 1..**10**；每个标签出现 1 或 2 次；维大小为任意正整数（Python 大整数）。
- 切片数 1..**64**（`MAX_SLICES`）；片宽为任意正整数，最后一片允许更小。
- 子集 DP 复杂度约为 O(3^n · F²)（F 为前沿宽度，本类问题通常很小）；n=10
  稠密网络在普通机器上亚秒级完成。切片规划对每片各跑一次同样规模的 DP
  （片数 ≤ 64），整体仍为亚秒至数秒量级。

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
- `tests/test_sliced.py`：`find_sliced_plan` 的正常、边界与失败用例。
  手算小例子核对逐片乘法/加法/峰值（等宽片、不等宽末片、片宽 1、单片
  退化、向量输出的常驻累加器峰值）；非法参数（输出轴不可切、轴不存在、
  片宽 0/bool/float/NaN、切片数超 `MAX_SLICES`、单张量无收缩轴）、过紧
  上限（首片可行而第二片扣除累加器后经穷举无解，异常定位到切片号）、
  预算耗尽；并对 n=2..5 确定性随机网络逐片调用 `reference_enum` 交叉
  核对乘法与峰值，再用独立逐片执行器（逐片真实收缩再累加）与独立直算
  逐元素 `==` 比对。

## 已知限制

- 不做分块/布局/内核级内存建模，内存只计标量元素驻留。
- 不支持标签出现 ≥3 次、外积、广播、稀疏张量；非法情形直接报错。
- 前沿可能随极端维大小组合变宽；n≤10 下预算默认关闭即可穷举，必要时可用
  `search_budget` 显式管理（耗尽时得到的是 `BudgetExhaustedError` 而非无解
  结论）。
- 切片规划一次只切**一个**收缩轴；逐片独立调用旧规划器，**不做**共享子
  表达式跨片复用/重计算优化，也不做多轴联合切块或跨切片的全局联合最优。
  累加器按完整输出常驻建模（整块驻留），不模拟累加器自身的分块写回。

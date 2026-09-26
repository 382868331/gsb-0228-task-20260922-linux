# tensor_peak — 张量收缩的峰值内存与模拟精度规划

给定稠密张量网络的维度标签、维大小和每个输入的元素绝对值上界，在“只允许
两两收缩、禁止外积”的约束下规划括号顺序。新版接口同时选择每个收缩步骤的
两档模拟浮点精度，在峰值元素数上限和最终绝对误差上限内最小化精度加权成本；
旧的只按内存和乘法数规划的接口继续保留。

规划器只读取形状、元素绝对值上界和输入误差上界，不读取真实大张量，也不会
因为某个小样本偶然准确而放行不满足保守界的计划。

## 环境准备

- 固定目标环境：Linux 原生 Python 3.14.7。
- 只使用 Python 标准库；核心、测试和演示均离线运行。
- 不使用 Docker、外部服务、网络 API 或真实凭据。

    python3.14 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip  # 可选；项目本身无第三方依赖
    python --version                     # 应显示 Python 3.14.7

## 运行命令

    python -m unittest discover -s tests -v
    python demo.py                  # 旧版：只按内存规划的演示
    python demo_iteration.py        # 新版：顺序+两档精度联合规划演示

两个演示都在普通 Linux 主机上远低于 90 秒完成；新版演示当前约 0.1 秒，
所有数字由算法当场计算，没有 sleep 或预录结果。

## 问题模型

- 每个维度标签在全部输入中恰出现 1 或 2 次：出现 1 次是输出轴，出现 2 次
  是收缩轴。
- 收缩双方必须至少共享一个收缩轴；不共享轴的外积禁止。
- 一次收缩的标量乘法数 = 两侧所有轴（并集）维大小之积。
- 峰值驻留以标量元素个数计：收缩瞬间同时驻留两个操作数、新输出以及后台
  暂存的其它中间结果。规划同时枚举左右子树求值顺序。
- 原始输入可按需载入，不假定整个网络的真实数据同时驻留。

## 旧版接口

旧版 find_plan(tensors, memory_limit, search_budget=None) 保持原有语义：
在 memory_limit 内最小化标量乘法数，不考虑精度。pareto_front 返回
(mult_count, peak_memory) Pareto 前沿。旧测试全部保留。

## 新版接口

    from decimal import Decimal
    from tensor_peak import TensorSpec, find_precision_plan

    plan = find_precision_plan(
        tensors,
        memory_limit=100,
        error_limit=Decimal("0.007"),
        value_bounds={"A": Decimal(1), "B": Decimal(1), "C": Decimal(1)},
        input_error_bounds={"A": Decimal("0.0001"), "B": Decimal(0), "C": Decimal(0)},
        precision_levels=DEFAULT_PRECISION_LEVELS,
        search_budget=None,
    )

- value_bounds：必选，字典必须恰好包含每个输入名；值为非负、有限的 Decimal、
  int 或十进制字符串，表示每个元素绝对值上界。
- input_error_bounds：可选，缺省为全 0；按每个输出元素的最坏绝对误差给上界。
- error_limit：必选正数，最终每个输出元素绝对误差的硬上限。
- precision_levels：必须恰好两档 PrecisionSpec；默认低档 unit_roundoff=10^-3、
  cost_per_mult=1，高档 unit_roundoff=10^-6、cost_per_mult=8。
- search_budget：None 表示完整搜索；正整数限制“顺序×精度候选求值次数”。

PrecisionPlan 在原有内存字段之外增加 total_cost、final_error_bound、
error_limit、precision_levels、value_bounds 和 input_error_bounds。每个
PrecisionContractStep 含 precision、contraction_length、gamma_n、左右操作数
幅度、名义中间量幅度、传入误差、舍入误差、输出误差、输出幅度界、本步成本、
形状、乘法数、驻留清单和峰值证据。

## 保守误差账本

一次收缩的共享轴长度记为 K，左右操作数每个输出元素的名义幅度和已有绝对
误差上界分别为 a、b 和 ea、eb，当前精度单位舍入为 u：

    gamma_K = K*u / (1 - K*u)
    nominal = K*a*b
    inherited = K*(a*eb + b*ea + ea*eb)
    rounding  = gamma_K * (a + ea) * (b + eb)
    new_error = inherited + rounding
    new_magnitude = nominal + new_error

若 K*u >= 1，该精度在这一步没有有效保守点积界，候选直接判为不可行。所有
传播用 Decimal 逐步计算，并对舍入项额外乘 1+10^-60 以覆盖 Decimal 有限精度
计算本身；误差界不使用真实张量采样、浮点统计或容差。

## 异常语义

- InvalidProblemError：输入契约错误。
- DisconnectedNetworkError：收缩网络不连通，禁止外积时无任何树。
- MemoryLimitInfeasibleError：旧接口完整穷举后证明内存上限无解。
- ConstraintsInfeasibleError：新接口完整穷举顺序和两档精度后证明内存或误差
  约束无解。
- BudgetExhaustedError：搜索预算不足而停止；没有证明无解。

## 算法与测试

旧接口用子集 DP 保存 (累计乘法数, 峰值) Pareto 前沿。新接口用子集 DP 保存
(累计成本, 峰值, 最终误差) 三维互不支配前沿；每次合并枚举左右求值顺序和两档
精度，用 Decimal 传播误差账本。输入数上限为 10。

- tests/test_tensor_peak.py：旧内存规划、Pareto、预算/无解区分、非法输入。
- tests/reference_enum.py：旧规划的独立全二叉树参考。
- tests/test_numeric.py：旧计划的稠密整数执行器和独立直算对照。
- tests/test_precision.py：小网络独立枚举所有括号顺序和两档精度组合，Decimal
  独立传播保守点积界，核对最优成本、峰值、最终误差、输入误差、内存无解、
  最终误差无解和预算不足。

## 限制

- 模拟精度是抽象十进制单位舍入模型，不声称对应某个具体硬件指令编码；
  需要时可传入两档自定义 PrecisionSpec。
- 内存只计稠密标量元素数，不模拟分块、布局、dtype 字节、对齐、并行流或
  内核写放大。
- 不支持标签出现 3 次及以上、外积、广播、对角、稀疏张量或复数轴语义。
- 误差界是逐元素最坏情况界；实际小样本误差更小不能使违反账本的计划变可行。

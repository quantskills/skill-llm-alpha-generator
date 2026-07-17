# 算子白名单（LLM prompt 硬约束）

> 本文件列出表达式引擎允许的**全部 22 个算子**。这是 LLM 生成候选公式、GP 演化、三层校验共用的**唯一算子集**。
> LLM prompt 里逐字给出这份清单（`_gen_system_prompt` 用 `sorted(OPERATOR_NAMES)` 拼接），越界算子会在**语法层**被 `validate_expression` 拦下记入 rejected。
> 权威定义在 `scripts/operators.py` 的 `OPERATORS` 字典，新增算子请改那里并同步本文件。

## 公式语法

- **前缀函数式**（Python 调用形式）：`sub(ts_mean(close,5), ts_mean(close,20))`
- **叶子**：已注册特征名（裸标识符，如 `close`）或数字常数（如 `5`、`-2.0`）
- **时序算子的最后一个参数必须是整数窗口常数**（如 5 / 10 / 20），以 const 叶子出现；窗口 `n >= 0`，负窗口在前视层被拒
- 只能用已注册特征（见 data_guide 的特征表），不能臆造
- 量纲不一致不能加减（`add`/`sub` 两侧同量纲，见量纲规则）

## 一阶时序（10 个，category=`一阶时序`）

窗口参数 `n` 为整数常数。

| 算子 | arity | 语义 |
|---|---|---|
| `ts_mean(x, n)` | 2 | 滚动均值 |
| `ts_std(x, n)` | 2 | 滚动标准差（ddof=1，单点返回 0） |
| `ts_max(x, n)` | 2 | 滚动最大值 |
| `ts_min(x, n)` | 2 | 滚动最小值 |
| `ts_rank(x, n)` | 2 | 当前值在最近 n 期内的时序百分位（0~1） |
| `ts_zscore(x, n)` | 2 | 滚动 z-score = (x - 滚动均值) / 滚动标准差 |
| `delay(x, n)` | 2 | 滞后 n 期（shift） |
| `diff(x, n)` | 2 | n 期差分 x - delay(x, n) |
| `ts_decay_linear(x, n)` | 2 | 线性衰减加权移动平均（越近权重越大，归一化） |
| `ts_corr(a, b, n)` | 3 | a 与 b 在滚动窗口 n 内的皮尔逊相关系数 |

## 二阶组合（11 个，category=`二阶组合`）

| 算子 | arity | 语义 |
|---|---|---|
| `add(a, b)` | 2 | 加 |
| `sub(a, b)` | 2 | 减 |
| `mul(a, b)` | 2 | 乘 |
| `protected_div(a, b)` | 2 | 除零保护除法（\|b\| 很小时返回 1.0） |
| `signed_power(a, b)` | 2 | 带符号幂 sign(a)·\|a\|^b |
| `max2(a, b)` | 2 | 逐元素取较大值 |
| `min2(a, b)` | 2 | 逐元素取较小值 |
| `log(x)` | 1 | 保护对数 log(\|x\| + eps) |
| `abs(x)` | 1 | 绝对值 |
| `sign(x)` | 1 | 符号函数（-1/0/1） |
| `rank(x)` | 1 | 百分位排序（0~1；单标的时序退化为整段 pct rank） |

## 条件逻辑（1 个，category=`条件逻辑`）

| 算子 | arity | 语义 |
|---|---|---|
| `if_then_else(cond, a, b)` | 3 | cond>0 取 a，否则取 b（cond 视作布尔，>0 为真） |

## 量纲约束（校验器第二层）

量纲标签：`price` / `return` / `ratio` / `volume` / `oi` / `bool` / `unknown`。校验规则（`validator.infer_node_dimension`）：

- `add` / `sub`：左右子树**必须同量纲**，否则非法（如 price + volume）
- `mul` / `protected_div`：放宽，结果量纲统一记 `ratio`
- `log` / `signed_power`（及未来的 `sqrt`）：输入须为连续量（`price`/`return`/`ratio`/`volume`/`oi`/`unknown`），`bool` 进入即非法
- `if_then_else(cond, a, b)`：a / b 两分支要求同量纲，cond 不参与量纲约束
- 常数（const）视作无量纲 `unknown`，可与任意量纲兼容

## 前视约束（校验器第三层）

- 时序算子（`ts_*` / `delay` / `diff` / `ts_corr`）的窗口 `n` 必须 `>= 0`；`n < 0` 会读到未来数据，判非法
- 本算子集所有时序算子都是「回看」的，无内置未来函数

## 给 LLM 的硬约束小结（写进 system prompt）

1. 只能用上面 22 个算子，严格拼写
2. 时序算子最后一个参数是整数窗口常数（5 / 10 / 20 …）
3. 只能用已注册特征名，不臆造
4. 不同量纲不能 add / sub（如 price 与 volume）
5. 必须调用 `emit_alpha_candidates` 工具返回公式列表

# 适应度与 AlphaEval 五维设计（G00 LLM-Alpha 因子挖掘）

本 skill 有**两套评分**，职责不同：

- **GP 适应度（fitness.py）**：单标量，指导遗传编程演化搜索，越大越好。追求「快而稳」的排序信号。
- **AlphaEval 五维（alpha_eval.py）**：一张「雷达图」，对入选 top_k 因子做多角度体检，用于筛选 / 复盘 / 报告。

> **严格样本外裁决归回测 skill。** 本 skill 只做**样本内**的 RankIC / 五维打分，是「粗筛 + 可解释性」，不是终裁。费率 / 滑点 / 资金费率 / walk-forward 严格样本外都归回测 skill。

## 一、GP 适应度 evaluate_fitness

```python
evaluate_fitness(signal, future_return,
                 elite_signals=None, node_count=None,
                 *, lam=0.15, alpha=0.2, complexity_threshold=30)
    -> (fitness: float, details: dict)
```

```
fitness = base_score - diversity_penalty - complexity_penalty

base_score          = |Spearman rankIC(signal, future_return)| × (1 - λ·turnover)
diversity_penalty   = α · max(|corr(signal, 每个精英)|)
complexity_penalty  = min(0.005 · max(0, node_count - 30), 0.1)
```

### 三个组成

1. **主指标：RankIC × 换手惩罚**
   - `rankIC`：signal 与 future_return 的 Spearman 秩相关，衡量单调预测力（不假设线性）。
   - `turnover`：把 signal 转百分位秩，相邻期秩变化的平均绝对值（0~1），衡量排序抖动 / 交易成本代理。
   - 折扣因子 `(1 - λ·turnover)` 夹到 [0,1]（λ=0.15），换手越高扣得越多。

2. **多样性惩罚**：给了 `elite_signals` 时，减去与各精英信号的**最大 |秩相关|** × α（α=0.2），逼迫种群产出与已有精英不相关的新信号，避免同质化。

3. **复杂度惩罚**：`node_count > 30` 后每超 1 个节点扣 0.005，**封顶 0.1**，抑制表达式膨胀（bloat），但不让复杂度项压过主指标。

### 退化保护

- 有效样本 `< 30`（`_MIN_VALID`）→ fitness 判 **0.0**，此类信号被自然淘汰。
- 常数序列（nunique < 2）→ rankIC = 0。
- NaN/Inf 位置对齐后剔除再算。

### details 键

`rankic` / `abs_rankic` / `turnover` / `base_score` / `diversity_penalty` / `complexity_penalty` / `max_corr_elite` / `node_count` / `n_valid` / `valid`。

## 二、AlphaEval 五维 alpha_eval

```python
alpha_eval(signal, future_return, all_signals=None,
           formula_str=None, llm_client=None, config=None)
    -> {pps, pfs, rre, logic, diversity, weighted_score, detail}
```

每维归一化到 [0,1]，加权汇总：

```
weighted_score = 0.40·pps + 0.20·pfs + 0.20·rre + 0.15·logic + 0.05·diversity
```

| 维 | 全称 | 度量 | 降级 |
|---|---|---|---|
| **pps** | Predictive Power Score | rankIC 幅度 + 分块 IC 稳定性(IR)，映射到 [0,1]（IC_SCALE=0.05，IR_SCALE=1.0） | 样本不足→0 |
| **pfs** | Predictive Fitness Stability | 相邻期信号秩一致性（1 - 平均秩变化），换手越低越稳 | 样本不足→0 |
| **rre** | Robustness to Random perturbation | 叠加多档高斯噪声（0.1/0.25/0.5 倍标准差，各重复 5 次）后 rankIC 保持率均值 | base_ic<0.02→0 |
| **logic** | 金融逻辑 | 给了 formula_str + llm_client 则 LLM 打 1-10 分归一化；否则中性 | 无 LLM→0.5 |
| **diversity** | 多样性 | 给了 all_signals 则算信号相关矩阵特征值分布的**谱熵**，越不相关谱熵越高 | 无 others→0.5 |

### 设计原则

- **任意维度在数据退化 / 依赖缺失时都稳健返回中性或 0，绝不抛异常中断 pipeline。**
- LLM 完全可选：无 llm_client 时 logic 降级 0.5，其它四维全是确定性计算。
- rre 的门槛设计：base_ic < 0.02（明显高于大样本随机信号的 IC 噪声地板 ~1/√n）时判信号本无预测力，谈「加噪保持率」无意义，直接 rre=0。

### detail 结构

`detail.weights` + 逐维子诊断 `detail.{pps, pfs, rre, logic, diversity}`（各含该维内部中间量，如 `pps.rankic` / `pfs.turnover`）。

## 三、轻量防过拟合（本 skill 内）

样本内层面已做的**轻量**防过拟合，不替代回测 skill 的严格样本外裁决：

- **换手惩罚**：主指标直接对高换手打折，抑制「靠频繁翻转刷 IC」。
- **复杂度惩罚**：抑制表达式膨胀，偏好简洁公式。
- **多样性惩罚 / 谱熵**：避免 top_k 因子高度同质、互为镜像。
- **rre 扰动鲁棒性**：加噪后 IC 崩掉的脆弱信号被压分。
- **最小样本门槛**：样本 < 30 直接判 0。

## 四、明确不做的事（归回测 skill）

- walk-forward / 严格样本外 / 滚动重训
- 费率 / 滑点 / 资金费率 / 保证金
- 组合层面的持仓、换手成本、回撤、夏普等绩效
- 多重比较校正 / deflated Sharpe 等严格统计裁决

本 skill 的产物是**候选因子 + 样本内证据 + 经济解释**，交给回测 skill 做终裁。

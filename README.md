# llm-alpha-generator

公式化 Alpha 因子挖掘 skill，面向股票、期货以及其他按时间对齐的结构化数值数据。OHLCV 是内置数据适配器的默认入口，不是因子表达式的限制；只要数据经过清洗、时间对齐并注册为可用特征，价格、成交、基本面、宏观、盘口、另类数据和模型输出都可以参与因子构造。它负责生成、校验、精修、评价和解释因子，不负责正式回测、交易执行或资金曲线分析。

## 功能

完整流程：

1. 读取并标准化时间序列数据；内置适配器默认支持股票和期货行情。
2. 计算内置价格、收益、成交量、持仓量、波动率和技术指标，也可以接入用户注册的其他数值特征。
3. 使用 LLM 生成多样化候选公式。
4. 执行算子白名单、表达式解析、量纲约束和前视检查。
5. 使用遗传编程对合法候选进行 warm-start 精修。
6. 使用 AlphaEval 五维指标评价因子：预测有效性、稳健性、可解释性、多样性和简洁性。
7. 使用 LLM 生成结构化经济解释。
8. 生成 HTML 报告和结构化运行结果。

## 强制 LLM 规则

候选生成、特征量纲判断、AlphaEval 逻辑评分和经济解释永久强制使用 LLM。调用失败、返回格式错误或字段缺失时立即报错，不会纯 GP、使用 0.5 中性分、生成模板解释或发布不完整报告。

唯一外部模型配置入口是 `config["llm"]`，所有阶段共享同一运行时客户端：

```python
config = {
    "llm": {
        "base_url": "https://your-llm-gateway.example/v1",
        "api_key": "<YOUR_API_KEY>",
        "model": "<YOUR_MODEL>",
        "protocol": "anthropic-tools",
    }
}
```

四个字段必须同时提供。未配置外部模型时，使用当前 Codex、Claude Code 等工具宿主提供的大模型。禁止使用 `enabled`、`provider`、环境变量开关或第二套配置入口。

## 使用方法

### Python 调用

```python
from scripts.generate import run

result = run(
    universe="RB.SHF",
    start_date="20230101",
    end_date="20241231",
    config={
        "data": {"use_synthetic": False, "asset_type": "auto"},
        "n_candidates": 50,
        "top_k": 10,
        "gp": {"pop_size": 80, "n_gen": 15, "seed": 42},
        "output_dir": "output",
    },
)
```

### 命令行调用

```bash
python scripts/generate.py --universe RB.SHF --start 20230101 --end 20241231 --out output
```

### 合成数据验收

```python
result = run(
    "SYNTH", "20230101", "20241231",
    config={"data": {"use_synthetic": True}, "n_candidates": 20, "top_k": 5},
)
```

合成数据只用于测试流程，不应作为正式研究结论。

## 数据接入

内置行情由 `scripts/data_loader.py` 统一接入，详细口径见 `references/data_guide.md`。如果使用基本面、宏观、盘口、情绪或其他另类数据，需要先通过数据适配器载入，并保证每个特征具有明确的时间戳、数值类型和可追溯的数据来源。数据账号只能在运行环境中提供，仓库不保存账号、密码、API key 或本机路径：

```powershell
$env:PANDA_DATA_USERNAME = "<YOUR_USERNAME>"
$env:PANDA_DATA_PASSWORD = "<YOUR_PASSWORD>"
```

没有可用数据或特征无法完成时间对齐时，流程不会伪造结果，也不会发布正式报告。

## 输出

`run()` 返回 `factors`、`trajectory`、`rejected`、`report_path` 和 `meta`。因子结果包含公式、五维评分、RankIC、样本内外指标、信号统计、经济解释和 LLM 元数据。

`meta["llm_runtime"]` 记录运行模式、模型、`run_id`、每个阶段状态、`request_id` 和 `payload_hash`。报告包含入选因子、五维评分、经济解释、拒绝候选、GP 迭代图和每轮迭代明细，并在完整成功后原子发布。

## 目录结构

```text
skill-llm-alpha-generator/
├── agents/          # Agent 加载器和元数据
├── references/      # 按需读取的规范和方法文档
├── scripts/         # 主流程、运行时、报告和测试
├── .gitignore       # 忽略缓存、凭证和运行结果
├── LICENSE
├── README.md
├── README.en.md
├── requirements.txt
└── SKILL.md
```

主要模块：`generate.py` 主流程，`llm_runtime.py` 唯一 LLM 边界，`dimension_inferrer.py` 量纲判断，`alpha_eval.py` 五维评分，`llm_explainer.py` 经济解释，`report_builder.py` HTML 渲染，`data_loader.py` 行情接入。

## 测试与安全发布

```bash
python scripts/test.py -q
python -X utf8 <CODEX_HOME>/skills/.system/skill-creator/scripts/quick_validate.py .
```

提交 Git 前确认没有 `.env`、凭证文件、运行报告、结果 JSON、真实 API key、账号、密码、邮箱或个人路径。`.gitignore` 已覆盖本地缓存、报告和凭证文件。详细规则见 `SKILL.md`、`references/method_guide.md`、`references/llm_policy.md` 和 `references/output_contract.md`。

Formulaic alpha-factor mining skill for stocks and futures.

The workflow uses the current tool-host LLM by default, or one complete external configuration under `config.llm`. Candidate generation, dimension inference, AlphaEval logic scoring, and economic explanation are mandatory LLM stages. Any failure stops the run and prevents report publication.

## Layout

- `SKILL.md`: agent workflow and contracts
- `agents/`: portable loader and agent metadata
- `references/`: data, method, LLM, operator, fitness, and output contracts
- `scripts/`: deterministic factor mining implementation and tests

Run the test suite with `python scripts/test.py`.

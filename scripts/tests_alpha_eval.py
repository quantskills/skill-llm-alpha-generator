# -*- coding: utf-8 -*-
"""
tests_alpha_eval.py — alpha_eval.py 单元测试

用合成数据验证 AlphaEval 五维打分的核心行为：
    1. 强信号（与未来收益强相关）→ pps 高；随机信号 → pps 低
    2. 平滑信号 pfs 高于高换手信号
    3. 强信号加噪后 rre 合理（在 (0,1] 之间且有一定保持率）；随机信号 rre≈0
    4. 显式关闭 LLM 时 logic 为 0.5；启用时注入 mock client 走 LLM，调用失败硬报错
    5. 多样性：不相关信号集 diversity 高于高度相关信号集；无 all_signals → 0.5
    6. 所有维度都落在 [0,1]，weighted_score 落在 [0,1] 且等于加权和
    7. 退化输入（样本不足 / 常数 / 全 NaN）稳健不抛异常

运行：pytest tests_alpha_eval.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# 保证能以裸模块名 import（运行时需要 scripts/ 在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_eval import (  # noqa: E402
    alpha_eval, compute_pps, compute_pfs, compute_rre,
    compute_logic, compute_diversity,
)
from llm_explainer import LLMError  # noqa: E402
from fitness import compute_future_return  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助：造随机游走价格 + 对齐的未来收益
# ---------------------------------------------------------------------------
def _make_price(n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.02, size=n)
    price = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(price)


def _strong_signal(future_return: pd.Series, seed: int = 7) -> pd.Series:
    """与未来收益强相关的信号 = 未来收益 + 少量噪声。"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.003, size=len(future_return))
    return future_return + pd.Series(noise, index=future_return.index)


# ---------------------------------------------------------------------------
# 1) PPS：强信号高、随机信号低
# ---------------------------------------------------------------------------
def test_pps_strong_vs_random():
    n = 3000  # 大样本让随机信号的 rankIC 收敛到噪声地板（避免小样本 IC 假阳）
    close = _make_price(n, seed=42)
    fut = compute_future_return(close, horizon=1)

    strong = _strong_signal(fut)
    rng = np.random.default_rng(99)
    rand = pd.Series(rng.normal(0.0, 1.0, size=n))

    d_strong = compute_pps(strong, fut)
    d_rand = compute_pps(rand, fut)

    assert d_strong['pps'] > 0.6, f"强信号 pps={d_strong['pps']}"
    assert d_rand['pps'] < 0.3, f"随机信号 pps={d_rand['pps']}"
    assert d_strong['pps'] > d_rand['pps']
    # rankIC 幅度关系
    assert abs(d_strong['rankic']) > abs(d_rand['rankic'])


# ---------------------------------------------------------------------------
# 2) PFS：平滑信号比高换手信号更稳
# ---------------------------------------------------------------------------
def test_pfs_smooth_vs_choppy():
    n = 500
    rng = np.random.default_rng(3)
    # 平滑信号：缓变趋势（相邻期排名几乎不变）
    smooth = pd.Series(np.cumsum(rng.normal(0.0, 0.1, size=n)))
    # 高换手信号：每期独立随机（相邻期排名频繁翻转）
    choppy = pd.Series(rng.normal(0.0, 1.0, size=n))

    d_smooth = compute_pfs(smooth)
    d_choppy = compute_pfs(choppy)

    assert d_smooth['pfs'] > d_choppy['pfs'], (
        f"平滑 pfs={d_smooth['pfs']} 应 > 高换手 pfs={d_choppy['pfs']}")
    assert 0.0 <= d_smooth['pfs'] <= 1.0
    assert 0.0 <= d_choppy['pfs'] <= 1.0


# ---------------------------------------------------------------------------
# 3) RRE：强信号加噪后保持率合理；随机信号 rre≈0
# ---------------------------------------------------------------------------
def test_rre_reasonable():
    n = 3000
    close = _make_price(n, seed=42)
    fut = compute_future_return(close, horizon=1)
    strong = _strong_signal(fut)

    d = compute_rre(strong, fut, rng=np.random.default_rng(1))
    # 有预测力的信号：加噪后 rre 应在 (0,1]，且小噪声保持率高
    assert 0.0 < d['rre'] <= 1.0, f"rre={d['rre']}"
    assert d['base_ic'] > 0.5
    # 小噪声档保持率应明显高于大噪声档
    per = d['per_level']
    assert per['0.1'] >= per['0.5'] - 1e-9, f"per_level={per}"

    # 随机信号：base_ic≈0 → rre=0
    rng = np.random.default_rng(99)
    rand = pd.Series(rng.normal(0.0, 1.0, size=n))
    d_rand = compute_rre(rand, fut, rng=np.random.default_rng(2))
    assert d_rand['rre'] == 0.0, f"随机信号 rre 应为 0，得 {d_rand['rre']}"


# ---------------------------------------------------------------------------
# 4) logic：不带 llm 降级 0.5；mock client 走 LLM
# ---------------------------------------------------------------------------
def test_logic_requires_llm():
    with pytest.raises(LLMError):
        compute_logic('sub(close, sma(close, 20))', llm_client=None)
    with pytest.raises(LLMError):
        compute_logic(None, llm_client=None)


class _MockBlock:
    """模拟 anthropic 响应内容块（tool_use）。"""
    def __init__(self, score, reason):
        self.type = 'tool_use'
        self.input = {'score': score, 'reason': reason}


class _MockResp:
    def __init__(self, blocks):
        self.content = blocks


class _MockMessages:
    def __init__(self, score, reason):
        self._score = score
        self._reason = reason
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _MockResp([_MockBlock(self._score, self._reason)])


class _MockClient:
    """模拟注入的 anthropic 客户端。"""
    def __init__(self, score=8, reason='动量逻辑清晰'):
        self.messages = _MockMessages(score, reason)


def test_logic_with_mock_llm():
    client = _MockClient(score=8, reason='价格减均线=动量，逻辑清晰')
    d = compute_logic('sub(close, sma(close, 20))', llm_client=client)
    assert d['source'] == 'llm'
    assert d['raw_score'] == 8.0
    # 8 → (8-1)/9 ≈ 0.777
    assert abs(d['logic'] - (8 - 1) / 9) < 1e-9
    assert 0.0 <= d['logic'] <= 1.0

    # 强制 tool use 已下发
    assert client.messages.last_kwargs['tool_choice']['name'] == 'report_logic_score'


def test_logic_llm_failure_fails_loudly():
    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError('network down')

    class _BoomClient:
        messages = _BoomMessages()

    with pytest.raises(LLMError):
        compute_logic('sub(close, sma(close, 20))', llm_client=_BoomClient())


# ---------------------------------------------------------------------------
# 5) diversity：不相关信号集高于高相关信号集；无 all_signals → 0.5
# ---------------------------------------------------------------------------
def test_diversity_uncorrelated_higher():
    n = 400
    rng = np.random.default_rng(5)
    target = pd.Series(rng.normal(size=n))

    # 高相关集合：都是 target 的微扰
    corr_set = [target * 1.01 + rng.normal(0, 1e-3, size=n),
                target * 0.99 + rng.normal(0, 1e-3, size=n)]
    # 不相关集合：各自独立随机
    uncorr_set = [pd.Series(rng.normal(size=n)),
                  pd.Series(rng.normal(size=n))]

    d_corr = compute_diversity(target, corr_set)
    d_uncorr = compute_diversity(target, uncorr_set)

    assert d_uncorr['diversity'] > d_corr['diversity'], (
        f"不相关 div={d_uncorr['diversity']} 应 > 高相关 div={d_corr['diversity']}")
    assert 0.0 <= d_corr['diversity'] <= 1.0
    assert 0.0 <= d_uncorr['diversity'] <= 1.0


def test_diversity_neutral_without_signals():
    n = 200
    target = pd.Series(np.random.default_rng(1).normal(size=n))
    d = compute_diversity(target, None)
    assert d['diversity'] == 0.5
    d2 = compute_diversity(target, [])
    assert d2['diversity'] == 0.5


# ---------------------------------------------------------------------------
# 6) 主入口：五维都在 [0,1]，weighted_score 合法且等于加权和
# ---------------------------------------------------------------------------
def test_alpha_eval_full_ranges():
    n = 600
    close = _make_price(n, seed=42)
    fut = compute_future_return(close, horizon=1)
    strong = _strong_signal(fut)
    others = [pd.Series(np.random.default_rng(i).normal(size=n)) for i in range(3)]

    res = alpha_eval(strong, fut, all_signals=others,
                     formula_str='sub(close, sma(close, 20))',
                     llm_client=_MockClient(score=8))

    # 五维 + 综合分都在 [0,1]
    for k in ('pps', 'pfs', 'rre', 'logic', 'diversity', 'weighted_score'):
        assert 0.0 <= res[k] <= 1.0, f"{k}={res[k]} 越界"

    assert res['logic'] == pytest.approx((8 - 1) / 9)
    assert res['logic_source'] == 'llm'
    assert res['logic_degraded'] is False

    # weighted_score == 加权和
    w = res['detail']['weights']
    expected = (w['pps'] * res['pps'] + w['pfs'] * res['pfs']
                + w['rre'] * res['rre'] + w['logic'] * res['logic']
                + w['diversity'] * res['diversity'])
    assert abs(res['weighted_score'] - expected) < 1e-9

    # 权重和为 1
    assert abs(sum(w.values()) - 1.0) < 1e-12

    # 强信号：pps 应偏高
    assert res['pps'] > 0.5


def test_alpha_eval_with_mock_llm():
    n = 400
    close = _make_price(n, seed=1)
    fut = compute_future_return(close, horizon=1)
    strong = _strong_signal(fut)
    client = _MockClient(score=9, reason='测试')

    res = alpha_eval(strong, fut, formula_str='sub(close, sma(close, 20))',
                     llm_client=client)
    assert res['detail']['logic']['source'] == 'llm'
    assert abs(res['logic'] - (9 - 1) / 9) < 1e-9


# ---------------------------------------------------------------------------
# 7) 退化输入稳健
# ---------------------------------------------------------------------------
def test_degenerate_inputs():
    # 样本不足
    sig = pd.Series(np.arange(10, dtype=float))
    fut = pd.Series(np.arange(10, dtype=float))
    res = alpha_eval(sig, fut, formula_str='sub(close, sma(close, 20))',
                     llm_client=_MockClient())
    for k in ('pps', 'pfs', 'rre', 'logic', 'diversity', 'weighted_score'):
        assert 0.0 <= res[k] <= 1.0
    assert res['pps'] == 0.0  # 样本不足

    # 常数信号
    n = 100
    const = pd.Series(np.full(n, 3.0))
    fut2 = pd.Series(np.random.default_rng(1).normal(size=n))
    res_c = alpha_eval(const, fut2, formula_str='sub(close, sma(close, 20))',
                       llm_client=_MockClient())
    for k in ('pps', 'pfs', 'rre', 'logic', 'diversity', 'weighted_score'):
        assert 0.0 <= res_c[k] <= 1.0

    # 全 NaN 信号
    nan_sig = pd.Series(np.full(n, np.nan))
    res_n = alpha_eval(nan_sig, fut2, formula_str='sub(close, sma(close, 20))',
                       llm_client=_MockClient())
    for k in ('pps', 'pfs', 'rre', 'logic', 'diversity', 'weighted_score'):
        assert 0.0 <= res_n[k] <= 1.0


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))

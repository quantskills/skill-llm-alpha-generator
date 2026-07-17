# -*- coding: utf-8 -*-
"""
tests_fitness.py — fitness.py 单元测试

用合成数据验证适应度函数的四个核心行为：
    1. 与未来收益强相关的信号 → rankIC 高、fitness 高
    2. 随机信号 → rankIC ≈ 0、fitness ≈ 0
    3. 高换手信号相对低换手信号被惩罚（base_score 更低）
    4. 与精英高相关的信号被扣分（diversity_penalty > 0，fitness 下降）

运行：pytest tests_fitness.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# 保证能以裸模块名 import（运行时需要 scripts/ 在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitness import compute_future_return, evaluate_fitness  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助：造一段随机游走价格 + 对齐的未来收益
# ---------------------------------------------------------------------------
def _make_price(n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    # 随机游走：对数收益累加成价格
    log_ret = rng.normal(0.0, 0.02, size=n)
    price = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(price)


# ---------------------------------------------------------------------------
# 1) compute_future_return 基本正确性
# ---------------------------------------------------------------------------
def test_compute_future_return_basic():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    fr = compute_future_return(close, horizon=1)
    # 每期涨 10%
    assert abs(fr.iloc[0] - 0.10) < 1e-9
    assert abs(fr.iloc[1] - 0.10) < 1e-9
    assert abs(fr.iloc[2] - 0.10) < 1e-9
    # 末尾无未来价 → NaN
    assert np.isnan(fr.iloc[-1])


def test_compute_future_return_horizon():
    close = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    fr = compute_future_return(close, horizon=2)
    # 两期后：100→121 = +21%
    assert abs(fr.iloc[0] - 0.21) < 1e-9
    # 末尾两个为 NaN
    assert np.isnan(fr.iloc[-1])
    assert np.isnan(fr.iloc[-2])


# ---------------------------------------------------------------------------
# 2) 强相关信号 → IC 高
# ---------------------------------------------------------------------------
def test_strong_signal_high_ic():
    rng = np.random.default_rng(42)
    n = 500
    close = _make_price(n, seed=42)
    future_return = compute_future_return(close, horizon=1)

    # 构造一个和未来收益强相关的信号：= 未来收益 + 少量噪声
    noise = rng.normal(0.0, 0.005, size=n)
    signal = future_return + pd.Series(noise)

    fit, details = evaluate_fitness(signal, future_return)
    # rankIC 应该很高
    assert details['abs_rankic'] > 0.8, f"abs_rankic={details['abs_rankic']}"
    assert fit > 0.5, f"fitness={fit}"
    assert details['valid'] is True


# ---------------------------------------------------------------------------
# 3) 随机信号 → IC ≈ 0
# ---------------------------------------------------------------------------
def test_random_signal_low_ic():
    rng = np.random.default_rng(42)
    n = 500
    close = _make_price(n, seed=42)
    future_return = compute_future_return(close, horizon=1)

    # 与未来收益无关的独立随机信号
    signal = pd.Series(rng.normal(0.0, 1.0, size=n))

    fit, details = evaluate_fitness(signal, future_return)
    # rankIC 应接近 0
    assert details['abs_rankic'] < 0.15, f"abs_rankic={details['abs_rankic']}"
    assert abs(fit) < 0.15, f"fitness={fit}"


# ---------------------------------------------------------------------------
# 4) 高换手信号被惩罚
# ---------------------------------------------------------------------------
def test_high_turnover_penalized():
    rng = np.random.default_rng(42)
    n = 500
    close = _make_price(n, seed=42)
    future_return = compute_future_return(close, horizon=1)

    # 基础：一个与未来收益中等相关的平滑信号（低换手）
    base = future_return.rolling(5, min_periods=1).mean()

    # 低换手版本：平滑信号本身（相邻期排序变化小）
    low_turn_signal = base.copy()

    # 高换手版本：在平滑信号上叠加大幅抖动，使相邻期排序频繁翻转，
    # 但保持与未来收益的整体相关性尽量接近（用同一 base + 交替符号扰动）
    flip = pd.Series(rng.normal(0.0, base.std() * 3 + 1e-6, size=n))
    high_turn_signal = base + flip

    _, det_low = evaluate_fitness(low_turn_signal, future_return)
    _, det_high = evaluate_fitness(high_turn_signal, future_return)

    # 高换手信号的 turnover 应明显更大
    assert det_high['turnover'] > det_low['turnover'], (
        f"high turnover={det_high['turnover']} 应 > low={det_low['turnover']}")

    # 单独验证换手折扣机制：同一 rankIC 下，turnover 越大 base_score 越小
    # 用一个固定信号，人为对比不同 turnover 的折扣效果
    fit_smooth, _ = evaluate_fitness(low_turn_signal, future_return, lam=0.15)
    # turnover=0 时折扣为 1，base_score == abs_rankic
    ic = det_low['abs_rankic']
    expected_no_discount = ic * (1 - 0.15 * det_low['turnover'])
    assert abs(det_low['base_score'] - expected_no_discount) < 1e-9

    # λ 越大，惩罚越重
    fit_lam0, d0 = evaluate_fitness(high_turn_signal, future_return, lam=0.0)
    fit_lam_big, dbig = evaluate_fitness(high_turn_signal, future_return, lam=0.5)
    assert dbig['base_score'] <= d0['base_score']


# ---------------------------------------------------------------------------
# 5) 与精英高相关被扣分
# ---------------------------------------------------------------------------
def test_diversity_penalty():
    rng = np.random.default_rng(42)
    n = 500
    close = _make_price(n, seed=42)
    future_return = compute_future_return(close, horizon=1)

    # 一个有预测力的信号
    noise = rng.normal(0.0, 0.005, size=n)
    signal = future_return + pd.Series(noise)

    # 精英池里放一个与 signal 几乎相同的信号（高相关）
    elite_similar = signal * 1.01 + pd.Series(rng.normal(0.0, 1e-4, size=n))

    # 无精英
    fit_no_elite, det_no = evaluate_fitness(signal, future_return)
    # 有高相关精英
    fit_with_elite, det_with = evaluate_fitness(
        signal, future_return, elite_signals=[elite_similar])

    # 多样性惩罚应 > 0，且 fitness 下降
    assert det_with['diversity_penalty'] > 0.0
    assert det_with['max_corr_elite'] > 0.8
    assert fit_with_elite < fit_no_elite, (
        f"with_elite={fit_with_elite} 应 < no_elite={fit_no_elite}")

    # 与不相关精英对比：惩罚应更小
    elite_random = pd.Series(rng.normal(0.0, 1.0, size=n))
    _, det_rand = evaluate_fitness(
        signal, future_return, elite_signals=[elite_random])
    assert det_rand['diversity_penalty'] < det_with['diversity_penalty']


# ---------------------------------------------------------------------------
# 6) 复杂度惩罚
# ---------------------------------------------------------------------------
def test_complexity_penalty():
    rng = np.random.default_rng(42)
    n = 500
    close = _make_price(n, seed=42)
    future_return = compute_future_return(close, horizon=1)
    signal = future_return + pd.Series(rng.normal(0.0, 0.005, size=n))

    # 节点数在阈值内 → 无惩罚
    fit_small, det_small = evaluate_fitness(signal, future_return, node_count=10)
    assert det_small['complexity_penalty'] == 0.0

    # 节点数超阈值 → 有惩罚，fitness 下降
    fit_big, det_big = evaluate_fitness(signal, future_return, node_count=60)
    assert det_big['complexity_penalty'] > 0.0
    assert fit_big < fit_small


# ---------------------------------------------------------------------------
# 7) 有效样本不足 → 返回 0
# ---------------------------------------------------------------------------
def test_insufficient_samples():
    signal = pd.Series(np.arange(10, dtype=float))
    future_return = pd.Series(np.arange(10, dtype=float))
    fit, details = evaluate_fitness(signal, future_return)
    assert fit == 0.0
    assert details['valid'] is False
    assert details['n_valid'] == 10


# ---------------------------------------------------------------------------
# 8) 全 NaN / 常数信号 → 稳健返回 0
# ---------------------------------------------------------------------------
def test_degenerate_signals():
    n = 100
    future_return = pd.Series(np.random.default_rng(1).normal(size=n))
    # 常数信号
    const_signal = pd.Series(np.full(n, 3.0))
    fit_c, det_c = evaluate_fitness(const_signal, future_return)
    assert fit_c == 0.0
    assert det_c['rankic'] == 0.0

    # 全 NaN 信号
    nan_signal = pd.Series(np.full(n, np.nan))
    fit_n, det_n = evaluate_fitness(nan_signal, future_return)
    assert fit_n == 0.0
    assert det_n['valid'] is False


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))

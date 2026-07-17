# -*- coding: utf-8 -*-
"""
tests_gp.py — GPEngine 单元测试

思路：
    合成一段数据，人为埋入一个已知信号——
    future_ret 与 ts_mean(close, 5) 强相关。
    fitness_fn 用「因子与 future_ret 的真实 IC（斯皮尔曼/皮尔逊）」。
    验证：
      1) GP 跑若干代后 best_fitness 明显 > 0（能挖到与信号相关的公式）；
      2) trajectory 的 best_fitness 单调不降（精英保留保证）。

为跑得快，pop_size/n_gen 用小值（50/10）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# 让裸模块名 import 生效
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expression import Node, evaluate, to_formula_string  # noqa: E402
from gp_engine import GPEngine  # noqa: E402


# ---------------------------------------------------------------------------
# 合成数据：埋入 future_ret ~ ts_mean(close, 5) 的强相关信号
# ---------------------------------------------------------------------------
def _make_data(n: int = 600, seed: int = 7):
    rng = np.random.default_rng(seed)
    # close 走一段带噪声的随机游走
    close = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0)
    volume = pd.Series(np.abs(rng.normal(1000, 100, n)))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close.shift(1).fillna(close.iloc[0])

    # 已知信号：ts_mean(close, 5) 的 z 化值 → 决定未来收益 future_ret
    ma5 = close.rolling(5, min_periods=1).mean()
    signal = (ma5 - ma5.mean()) / (ma5.std() + 1e-9)
    noise = rng.normal(0, 0.3, n)
    future_ret = 0.9 * signal.values + noise  # 强相关（信噪比高）
    future_ret = pd.Series(future_ret)

    data = {
        'open': open_.reset_index(drop=True),
        'high': high.reset_index(drop=True),
        'low': low.reset_index(drop=True),
        'close': close.reset_index(drop=True),
        'volume': volume.reset_index(drop=True),
    }
    return data, future_ret


def _make_fitness_fn(data, target: pd.Series):
    """构造只吃 node 的 fitness 闭包：因子与 target 的皮尔逊 IC（绝对值）。

    用绝对值：GP 可能挖到与信号负相关的等价公式，方向不重要。
    """
    tgt = target.reset_index(drop=True)

    def fitness_fn(node: Node) -> float:
        factor = evaluate(node, data)
        if not isinstance(factor, pd.Series):
            factor = pd.Series(np.full(len(tgt), float(factor)))
        factor = factor.reset_index(drop=True)
        # 对齐 + 清洗
        df = pd.DataFrame({'f': factor, 't': tgt}).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(df) < 20 or df['f'].std() < 1e-12:
            return 0.0
        ic = df['f'].corr(df['t'])
        if not np.isfinite(ic):
            return 0.0
        return abs(float(ic))

    return fitness_fn


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
def test_gp_finds_signal_and_monotone_trajectory():
    """GP 应能挖到与埋入信号相关的公式，且轨迹 best_fitness 单调不降。"""
    data, future_ret = _make_data()
    fitness_fn = _make_fitness_fn(data, future_ret)

    engine = GPEngine(
        feature_names=list(data.keys()),
        fitness_fn=fitness_fn,
        pop_size=50,
        n_gen=10,
        max_depth=6,
        seed=42,
    )
    result = engine.run()

    # 1) best_fitness 明显 > 0
    assert result['best_fitness'] > 0.3, (
        f"best_fitness 太低: {result['best_fitness']}, "
        f"best={to_formula_string(result['best'])}")

    # 2) trajectory 的 best_fitness 单调不降（精英保留）
    bests = [t['best_fitness'] for t in result['trajectory']]
    for prev, cur in zip(bests, bests[1:]):
        assert cur >= prev - 1e-12, f"best_fitness 出现下降: {bests}"

    # 3) trajectory 结构完整
    for t in result['trajectory']:
        assert {'gen', 'best_fitness', 'mean_fitness', 'diversity',
                'best_formula', 'current_best_fitness',
                'current_best_formula'} <= set(t.keys())
        assert isinstance(t['best_formula'], str)
        assert isinstance(t['current_best_formula'], str)
        assert 0.0 <= t['diversity'] <= 1.0

    # 4) best 是一棵可求值的 Node，top_k 非空且降序
    factor = evaluate(result['best'], data)
    assert isinstance(factor, pd.Series)
    assert len(result['top_k']) >= 1
    tk_fits = [f for _, f in result['top_k']]
    for prev, cur in zip(tk_fits, tk_fits[1:]):
        assert prev >= cur - 1e-12


def test_warm_start_seeds_used():
    """warm-start：注入的种子（含已知最优公式）应被采纳，best_fitness 更高更快。"""
    data, future_ret = _make_data()
    fitness_fn = _make_fitness_fn(data, future_ret)

    # 直接把已知最优公式 ts_mean(close, 5) 作为种子之一注入
    seed_node = Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)])
    warm = [seed_node]

    engine = GPEngine(
        feature_names=list(data.keys()),
        fitness_fn=fitness_fn,
        pop_size=50,
        n_gen=10,
        max_depth=6,
        seed=123,
    )
    result = engine.run(warm_start_pop=warm)

    # 已知最优公式的 IC
    seed_fit = fitness_fn(seed_node)
    # warm-start 后全局最优不应低于种子公式本身
    assert result['best_fitness'] >= seed_fit - 1e-9, (
        f"warm-start 后 best_fitness({result['best_fitness']}) "
        f"低于种子({seed_fit})")
    assert result['best_fitness'] > 0.5


def test_validator_filters_illegal():
    """validator_fn 拒绝所有含 'volume' 的个体：最终 best 不应包含 volume。"""
    data, future_ret = _make_data()
    fitness_fn = _make_fitness_fn(data, future_ret)

    from expression import collect_features

    def validator(node: Node) -> bool:
        return 'volume' not in collect_features(node)

    engine = GPEngine(
        feature_names=list(data.keys()),
        fitness_fn=fitness_fn,
        validator_fn=validator,
        pop_size=40,
        n_gen=8,
        max_depth=5,
        seed=2024,
    )
    result = engine.run()
    assert 'volume' not in collect_features(result['best'])
    # top_k 里也不应出现 volume
    for node, _ in result['top_k']:
        assert 'volume' not in collect_features(node)


if __name__ == '__main__':
    test_gp_finds_signal_and_monotone_trajectory()
    test_warm_start_seeds_used()
    test_validator_filters_illegal()
    print("all gp tests passed")

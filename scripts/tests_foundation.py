# -*- coding: utf-8 -*-
"""
tests_foundation.py — 地基模块单测

覆盖：
    - 表达式树构造 + 求值正确性（手工核对）
    - count_nodes / depth 正确
    - 每个算子都能跑不报错
    - protected_div 除零返回合理值
    - FeatureRegistry register/get/has/list 正确
    - random_tree 生成的树能求值

运行： pytest scripts/tests_foundation.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from operators import (OPERATORS, OPERATOR_NAMES, get_operator, list_operators,
                       op_protected_div, op_signed_power, winsorize)
from expression import (Node, evaluate, count_nodes, depth, to_formula_string,
                        collect_features, collect_operators, random_tree)
from feature_registry import FeatureRegistry


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
@pytest.fixture
def synth_data():
    """构造固定 seed 的合成量价数据。"""
    rng = np.random.default_rng(42)
    n = 100
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    volume = pd.Series(rng.uniform(1000, 5000, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    return {'close': close, 'volume': volume, 'high': high, 'low': low}


# ---------------------------------------------------------------------------
# 表达式树构造 + 求值正确性
# ---------------------------------------------------------------------------
def test_build_and_evaluate_add_ts_mean(synth_data):
    """构造 add(close, ts_mean(close, 5)) 并手工核对求值结果。"""
    tree = Node.make_op('add', [
        Node.feature('close'),
        Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)]),
    ])
    result = evaluate(tree, synth_data)

    # 手工核对：close + close 的 5 期滚动均值（min_periods=1）
    close = synth_data['close']
    expected = close + close.rolling(5, min_periods=1).mean()
    pd.testing.assert_series_equal(result, expected)


def test_formula_string(synth_data):
    """to_formula_string 输出可读公式。"""
    tree = Node.make_op('add', [
        Node.feature('close'),
        Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)]),
    ])
    assert to_formula_string(tree) == "add(close, ts_mean(close, 5))"


def test_collect_features_and_operators():
    """collect_features / collect_operators 正确收集。"""
    tree = Node.make_op('sub', [
        Node.make_op('ts_std', [Node.feature('high'), Node.const(10)]),
        Node.feature('low'),
    ])
    assert collect_features(tree) == {'high', 'low'}
    assert collect_operators(tree) == {'sub', 'ts_std'}


# ---------------------------------------------------------------------------
# count_nodes / depth
# ---------------------------------------------------------------------------
def test_count_nodes():
    """count_nodes 统计正确。"""
    # add(close, ts_mean(close, 5)) = add + close + ts_mean + close + 5 = 5 个节点
    tree = Node.make_op('add', [
        Node.feature('close'),
        Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)]),
    ])
    assert count_nodes(tree) == 5


def test_depth():
    """depth 计算正确。"""
    leaf = Node.feature('close')
    assert depth(leaf) == 1

    # add(close, ts_mean(close, 5)) 深度 = 3
    tree = Node.make_op('add', [
        Node.feature('close'),
        Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)]),
    ])
    assert depth(tree) == 3


def test_arity_check():
    """make_op 对错误的子节点数量应报错。"""
    with pytest.raises(ValueError):
        Node.make_op('add', [Node.feature('close')])  # add 需要 2 个
    with pytest.raises(KeyError):
        Node.make_op('not_an_op', [Node.feature('close')])


# ---------------------------------------------------------------------------
# 每个算子都能跑
# ---------------------------------------------------------------------------
def test_all_operators_run(synth_data):
    """遍历全部算子，构造合法参数，确认都能求值不报错。"""
    close = synth_data['close']
    high = synth_data['high']

    for name, meta in OPERATORS.items():
        arity = meta['arity']
        # 时序 / 窗口类算子最后一参数用窗口常数
        window_ops = {'ts_mean', 'ts_std', 'ts_max', 'ts_min', 'ts_rank',
                      'ts_zscore', 'delay', 'diff', 'ts_decay_linear', 'ts_corr'}
        children: list[Node] = []
        for i in range(arity):
            is_last = (i == arity - 1)
            if name in window_ops and is_last:
                children.append(Node.const(5))
            else:
                # 交替用 close / high 做序列参数
                children.append(Node.feature('close' if i % 2 == 0 else 'high'))
        tree = Node.make_op(name, children)
        result = evaluate(tree, synth_data)
        assert isinstance(result, pd.Series), f"{name} 未返回 Series"
        assert len(result) == len(close), f"{name} 长度不符"


def test_protected_div_zero():
    """protected_div 除零应返回合理值（约定为 1.0），不产生 inf/nan。"""
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([0.0, 0.0, 2.0])
    result = op_protected_div(a, b)
    # 前两个除零位置应为 1.0
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 1.0
    # 第三个 3/2 = 1.5
    assert result.iloc[2] == pytest.approx(1.5)
    assert not np.isinf(result).any()
    assert not result.isna().any()


# ---------------------------------------------------------------------------
# m8 数值卫生：signed_power 指数 clip + winsorize 缩尾
# ---------------------------------------------------------------------------
def test_signed_power_no_overflow_extreme_scale():
    """极端量级 + 大指数不再溢出成 inf（m8：指数先 clip 到 [-3,3]）。"""
    # a 量级 1e10（如成交额），b 给到 20（未 clip 时 (1e10)^20 = 1e200 早已爆表）
    a = pd.Series([1e10, -1e10, 1e8])
    b = pd.Series([20.0, 20.0, 20.0])
    result = op_signed_power(a, b)
    assert not np.isinf(result).any(), "指数被 clip 后不应再溢出 inf"
    assert not result.isna().any()
    # 符号必须保留：sign(a) * |a|^b
    assert result.iloc[0] > 0 and result.iloc[1] < 0


def test_signed_power_exponent_clipped():
    """指数超过 3 与恰好等于 3 结果相同（证明 clip 生效，不是碰巧没溢出）。"""
    a = pd.Series([2.0, 2.0])
    r_big = op_signed_power(a, pd.Series([100.0, 100.0]))
    r_cap = op_signed_power(a, pd.Series([3.0, 3.0]))
    # clip 到 3 后二者一致：2^3 = 8
    assert r_big.iloc[0] == pytest.approx(r_cap.iloc[0])
    assert r_cap.iloc[0] == pytest.approx(2.0 ** 3, rel=1e-6)


def test_winsorize_clips_extremes_and_kills_inf():
    """winsorize 把 inf 转 NaN、把极端分位截到边界，且不改中间秩序。"""
    s = pd.Series([np.inf, -np.inf] + list(range(1, 101)))  # 2 个 inf + 1..100
    w = winsorize(s, 0.05, 0.05)
    assert not np.isinf(w).any(), "inf 应被清成 NaN"
    valid = w.dropna()
    # 5% 缩尾后极值被压到分位边界，最大不应还是 100
    assert valid.max() < 100
    assert valid.min() > 1


def test_winsorize_constant_series_passthrough():
    """常数 / 全 NaN 序列无分位可算，原样返回不报错（inf 仍清成 NaN）。"""
    const = pd.Series([5.0, 5.0, 5.0])
    assert winsorize(const).tolist() == [5.0, 5.0, 5.0]
    allnan = pd.Series([np.inf, -np.inf])
    assert winsorize(allnan).isna().all()
    """get_operator / list_operators / OPERATOR_NAMES 一致。"""
    assert set(list_operators()) == OPERATOR_NAMES
    assert get_operator('add')['arity'] == 2
    with pytest.raises(KeyError):
        get_operator('does_not_exist')


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------
def test_registry_basic():
    """register / get / has / list 基本正确。"""
    reg = FeatureRegistry()
    reg.register('my_feat', unit='dimensionless', semantic='volatility',
                 description='测试特征')
    assert reg.has('my_feat')
    assert not reg.has('missing')
    rec = reg.get('my_feat')
    assert rec['unit'] == 'dimensionless'
    assert rec['semantic'] == 'volatility'
    assert rec['description'] == '测试特征'
    assert 'my_feat' in reg.list_features()
    assert reg.feature_names() == {'my_feat'}


def test_registry_default_unit():
    """内置标准特征应命中默认单位/语义。"""
    reg = FeatureRegistry()
    reg.register('close')
    reg.register('volume')
    reg.register('amount')
    reg.register('open_interest')
    reg.register('log_ret')
    # 单位（硬约束）：金额和成交量不再混为一谈
    assert reg.get_unit('close') == 'price'
    assert reg.get_unit('volume') == 'count'
    assert reg.get_unit('amount') == 'money'          # 修复 m13：金额独立于成交量
    assert reg.get_unit('open_interest') == 'count'
    assert reg.get_unit('log_ret') == 'dimensionless'
    # 语义（软标注）
    assert reg.get_semantic('open_interest') == 'open_interest'
    assert reg.get_semantic('log_ret') == 'return'


def test_registry_unknown_unit_fallback():
    """未知特征无默认单位时回退 unknown。"""
    reg = FeatureRegistry()
    reg.register('weird_feature_xyz')
    assert reg.get_unit('weird_feature_xyz') == 'unknown'
    assert reg.get_semantic('weird_feature_xyz') == 'unknown'


def test_registry_set_unit():
    """set_unit / set_semantic 回填（供推断器）。"""
    reg = FeatureRegistry()
    reg.register('weird_feature_xyz')
    reg.set_unit('weird_feature_xyz', 'dimensionless')
    reg.set_semantic('weird_feature_xyz', 'volatility')
    assert reg.get_unit('weird_feature_xyz') == 'dimensionless'
    assert reg.get_semantic('weird_feature_xyz') == 'volatility'
    # 对未注册的特征 set_unit 应顺带注册
    reg.set_unit('brand_new', 'count')
    assert reg.has('brand_new')
    assert reg.get_unit('brand_new') == 'count'


def test_registry_register_dataframe():
    """register_dataframe 批量注册并识别单位。"""
    df = pd.DataFrame({
        'close': np.arange(5, dtype=float),
        'volume': np.arange(5, dtype=float),
        'my_ratio': np.arange(5, dtype=float),
    })
    reg = FeatureRegistry()
    reg.register_dataframe(df, unit_map={'my_ratio': 'dimensionless'})
    assert reg.has('close') and reg.has('volume') and reg.has('my_ratio')
    assert reg.get_unit('close') == 'price'              # 默认映射
    assert reg.get_unit('volume') == 'count'             # 默认映射
    assert reg.get_unit('my_ratio') == 'dimensionless'   # 显式 unit_map


def test_registry_invalid_unit():
    """非法单位应报错。"""
    reg = FeatureRegistry()
    with pytest.raises(ValueError):
        reg.register('x', unit='not_a_unit')
    with pytest.raises(ValueError):
        reg.register('y', semantic='not_a_semantic')


def test_registry_to_prompt_list():
    """to_prompt_list 生成含名字+单位/语义+说明的文本。"""
    reg = FeatureRegistry()
    reg.register('close', description='收盘价')
    text = reg.to_prompt_list()
    assert 'close' in text
    assert 'price' in text
    assert '收盘价' in text


# ---------------------------------------------------------------------------
# random_tree
# ---------------------------------------------------------------------------
def test_random_tree_evaluable(synth_data):
    """random_tree 生成的多棵树都能求值成 Series。"""
    rng = np.random.default_rng(42)
    features = ['close', 'volume', 'high', 'low']
    for _ in range(30):
        tree = random_tree(features, max_depth=4, rng=rng)
        result = evaluate(tree, synth_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(synth_data['close'])
        # 用到的特征必须都在可用特征集合内
        assert collect_features(tree).issubset(set(features))
        # 深度不超过设定上限
        assert depth(tree) <= 4


def test_random_tree_empty_features():
    """空特征列表应报错。"""
    with pytest.raises(ValueError):
        random_tree([], max_depth=3)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))

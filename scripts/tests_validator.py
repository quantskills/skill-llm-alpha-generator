# -*- coding: utf-8 -*-
"""
tests_validator.py — validator.py 三层校验器单测

覆盖：
    - 合法表达式通过：add(close, ts_mean(close, 5))
    - 量纲非法被拦：add(close, turnover_ratio)（price + ratio）
    - 特征未注册被拦：用未注册变量
    - 前视被拦：delay(close, -1)
    - 连续量算子对 bool 输入被拦：log(bool 特征)

运行：
    在 scripts/ 目录内执行 `pytest tests_validator.py -v`
    （三模块用裸模块名互相 import，需 scripts/ 在 sys.path）
"""
from __future__ import annotations

import os
import sys

# 保证 scripts/ 在 sys.path，允许 `from operators import ...` 等裸模块名导入
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pytest

from expression import Node
from feature_registry import FeatureRegistry
from validator import validate_expression, validate_batch, infer_node_unit


# ---------------------------------------------------------------------------
# fixtures / 工具
# ---------------------------------------------------------------------------
def _make_registry() -> FeatureRegistry:
    """构造一个注册表：close=price、turnover_ratio=dimensionless、is_up=bool、
    volume=count、amount=money（后两者用于验证乘除单位推导 + 修复 m13）。"""
    reg = FeatureRegistry(seed_defaults=True)
    reg.register('close', unit='price', semantic='price', description='收盘价')
    reg.register('turnover_ratio', unit='dimensionless', semantic='liquidity',
                 description='换手率')
    reg.register('is_up', unit='bool', dtype='bool', description='是否上涨')
    reg.register('volume', unit='count', semantic='volume', description='成交量')
    reg.register('amount', unit='money', semantic='turnover', description='成交额')
    return reg


# ---------------------------------------------------------------------------
# ① 合法表达式通过
# ---------------------------------------------------------------------------
def test_valid_expression_passes():
    """add(close, ts_mean(close, 5)) —— price + price，应通过三层。"""
    reg = _make_registry()
    expr = Node.make_op('add', [
        Node.feature('close'),
        Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)]),
    ])
    res = validate_expression(expr, reg)
    assert res['valid'] is True, res
    assert res['layer'] is None
    assert res['reason'] == ''


# ---------------------------------------------------------------------------
# ② 量纲非法被拦
# ---------------------------------------------------------------------------
def test_dimension_mismatch_blocked():
    """add(close, turnover_ratio) —— price + dimensionless，单位层应拦下。"""
    reg = _make_registry()
    expr = Node.make_op('add', [
        Node.feature('close'),
        Node.feature('turnover_ratio'),
    ])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '量纲'
    assert 'price' in res['reason'] and 'dimensionless' in res['reason']


def test_add_cross_entity_blocked():
    """add(close, volume) —— price + count，实体跨单位应拦（修复缺陷1：量级差不再混过）。"""
    reg = _make_registry()
    expr = Node.make_op('add', [Node.feature('close'), Node.feature('volume')])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '量纲'
    assert 'price' in res['reason'] and 'count' in res['reason']


def test_add_same_money_ok():
    """add(amount, amount) —— money + money，同单位应通过。"""
    reg = _make_registry()
    expr = Node.make_op('add', [Node.feature('amount'), Node.feature('amount')])
    res = validate_expression(expr, reg)
    assert res['valid'] is True, res


def test_mul_price_count_is_money():
    """mul(close, volume) —— 价格×计数 → 金额，合法且顶层单位=money。"""
    reg = _make_registry()
    expr = Node.make_op('mul', [Node.feature('close'), Node.feature('volume')])
    res = validate_expression(expr, reg)
    assert res['valid'] is True, res
    assert infer_node_unit(expr, reg) == 'money'


def test_div_money_count_is_price():
    """protected_div(amount, volume) —— 金额÷计数 → 价格，合法且顶层单位=price。"""
    reg = _make_registry()
    expr = Node.make_op('protected_div', [Node.feature('amount'),
                                          Node.feature('volume')])
    res = validate_expression(expr, reg)
    assert res['valid'] is True, res
    assert infer_node_unit(expr, reg) == 'price'


def test_mul_relaxes_dimension():
    """mul(close, turnover_ratio) —— 价格×无量纲 → 价格（乘法保留实体单位）。"""
    reg = _make_registry()
    expr = Node.make_op('mul', [
        Node.feature('close'),
        Node.feature('turnover_ratio'),
    ])
    res = validate_expression(expr, reg)
    assert res['valid'] is True, res
    # 价格 × 无量纲 = 价格（不再一律拍平成 ratio）
    assert infer_node_unit(expr, reg) == 'price'


# ---------------------------------------------------------------------------
# ③ 特征未注册被拦
# ---------------------------------------------------------------------------
def test_unregistered_feature_blocked():
    """使用未注册变量 mystery_feat，注册表层应拦下。"""
    reg = _make_registry()
    expr = Node.make_op('add', [
        Node.feature('close'),
        Node.feature('mystery_feat'),  # 未注册
    ])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '注册表'
    assert 'mystery_feat' in res['reason']


# ---------------------------------------------------------------------------
# ④ 前视被拦
# ---------------------------------------------------------------------------
def test_lookahead_negative_window_blocked():
    """delay(close, -1) —— 负窗口读未来，前视层应拦下。"""
    reg = _make_registry()
    expr = Node.make_op('delay', [Node.feature('close'), Node.const(-1)])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '前视'
    assert '-1' in res['reason']


def test_diff_negative_window_blocked():
    """diff(close, -3) —— 负窗口同样被前视层拦下。"""
    reg = _make_registry()
    expr = Node.make_op('diff', [Node.feature('close'), Node.const(-3)])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '前视'


# ---------------------------------------------------------------------------
# ④b 单标的场景禁用横截面 rank（m9）
# ---------------------------------------------------------------------------
def test_single_asset_rank_blocked():
    """rank(close) —— 单标的时序禁用横截面 rank（前视/退化），前视层应拦下。"""
    reg = _make_registry()
    expr = Node.make_op('rank', [Node.feature('close')])
    res = validate_expression(expr, reg, single_asset=True)
    assert res['valid'] is False
    assert res['layer'] == '前视'
    assert 'ts_rank' in res['reason']  # 提示替代算子


def test_single_asset_nested_rank_blocked():
    """rank 嵌在子树里也应被单标的场景拦下（含 rank 即拦）。
    用 ts_mean(rank(close),5)：量纲合法（rank→dimensionless，ts_mean 保留），
    确保是在 rank 检查处被拦，而非先被量纲层拦下。"""
    reg = _make_registry()
    expr = Node.make_op('ts_mean', [
        Node.make_op('rank', [Node.feature('close')]),
        Node.const(5),
    ])
    # 多标的（默认）下量纲/前视都过 → 合法，先确认基线
    assert validate_expression(expr, reg)['valid'] is True
    # 单标的下 rank 被前视层拦下
    res = validate_expression(expr, reg, single_asset=True)
    assert res['valid'] is False
    assert res['layer'] == '前视'


def test_single_asset_ts_rank_allowed():
    """ts_rank(close, 10) —— 滚动、不前视，单标的场景仍合法（rank 的正确替代）。"""
    reg = _make_registry()
    expr = Node.make_op('ts_rank', [Node.feature('close'), Node.const(10)])
    res = validate_expression(expr, reg, single_asset=True)
    assert res['valid'] is True


def test_cross_section_rank_allowed_by_default():
    """默认 single_asset=False（多标的横截面），rank 合法——不改老行为。"""
    reg = _make_registry()
    expr = Node.make_op('rank', [Node.feature('close')])
    assert validate_expression(expr, reg)['valid'] is True
    assert validate_expression(expr, reg, single_asset=False)['valid'] is True


def test_batch_passes_single_asset_flag():
    """validate_batch 应把 single_asset 透传给每棵树。"""
    reg = _make_registry()
    nodes = [
        Node.make_op('rank', [Node.feature('close')]),
        Node.make_op('ts_rank', [Node.feature('close'), Node.const(5)]),
    ]
    res = validate_batch(nodes, reg, single_asset=True)
    assert res[0]['valid'] is False and res[0]['layer'] == '前视'  # rank 被拦
    assert res[1]['valid'] is True                                # ts_rank 放行
def test_log_of_bool_blocked():
    """log(is_up) —— bool 进连续量算子，量纲层应拦下。"""
    reg = _make_registry()
    expr = Node.make_op('log', [Node.feature('is_up')])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '量纲'
    assert 'bool' in res['reason']


def test_signed_power_of_bool_blocked():
    """signed_power(is_up, 2) —— bool 进 signed_power，量纲层应拦下。"""
    reg = _make_registry()
    expr = Node.make_op('signed_power', [Node.feature('is_up'), Node.const(2)])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '量纲'


# ---------------------------------------------------------------------------
# 补充：if_then_else 分支量纲
# ---------------------------------------------------------------------------
def test_if_then_else_branch_mismatch_blocked():
    """if_then_else(is_up, close, turnover_ratio) —— 两分支 price/ratio 不同量纲，被拦。"""
    reg = _make_registry()
    expr = Node.make_op('if_then_else', [
        Node.feature('is_up'),
        Node.feature('close'),
        Node.feature('turnover_ratio'),
    ])
    res = validate_expression(expr, reg)
    assert res['valid'] is False
    assert res['layer'] == '量纲'


def test_if_then_else_same_branch_passes():
    """if_then_else(is_up, close, ts_mean(close,5)) —— 两分支同为 price，应通过。"""
    reg = _make_registry()
    expr = Node.make_op('if_then_else', [
        Node.feature('is_up'),
        Node.feature('close'),
        Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)]),
    ])
    res = validate_expression(expr, reg)
    assert res['valid'] is True, res


# ---------------------------------------------------------------------------
# 补充：批量校验
# ---------------------------------------------------------------------------
def test_validate_batch():
    """validate_batch 逐个返回结果，顺序与输入一致。"""
    reg = _make_registry()
    good = Node.make_op('add', [
        Node.feature('close'),
        Node.make_op('ts_mean', [Node.feature('close'), Node.const(5)]),
    ])
    bad_dim = Node.make_op('add', [Node.feature('close'), Node.feature('turnover_ratio')])
    bad_look = Node.make_op('delay', [Node.feature('close'), Node.const(-1)])

    results = validate_batch([good, bad_dim, bad_look], reg)
    assert len(results) == 3
    assert results[0]['valid'] is True
    assert results[1]['valid'] is False and results[1]['layer'] == '量纲'
    assert results[2]['valid'] is False and results[2]['layer'] == '前视'


# ---------------------------------------------------------------------------
# m11：validator 真实过滤防回归——从「公式字符串」解析出的非法表达式，
#      三层各自能拦下。这更贴近运行时真实路径（LLM 产出的是字符串）。
#
# 说明（parse_formula 的边界，见 generate.parse_formula）：
#   - 未知算子 / 未注册特征：parse_formula 在「解析阶段」就直接抛 ValueError，
#     不会返回可交给 validator 的 Node。故：
#       · 语法层：无法走字符串入口喂给 validator，改用「绕过 make_op 直接构造
#         raw Node」把未知算子塞进 validator 的语法白名单检查（make_op 也会挡）。
#       · 注册表层：模拟真实场景——用「含该特征的注册表」解析出字符串，再拿
#         「不含该特征的注册表」去 validate（解析期与校验期注册表不一致），
#         从而让 validator 的注册表层真正拦下。
#   - 量纲不兼容相加 / 负窗口：parse_formula 不做这两类检查，能正常解析出 Node，
#     直接走字符串入口喂给 validator 即可被对应层拦下。
# ---------------------------------------------------------------------------
from expression import collect_operators  # noqa: E402
from generate import parse_formula        # noqa: E402


def test_m11_syntax_unknown_operator_blocked():
    """语法层：未知算子应被 validator 拦下（layer='语法'）。

    parse_formula 与 Node.make_op 都会在构造阶段就挡下未知算子，故这里绕过二者、
    直接构造 raw Node，验证 validator 的语法白名单本身能独立拦下越界算子。
    """
    reg = _make_registry()
    # 绕过 make_op（它会 KeyError），直接构造一个持有未知算子的节点
    bogus = Node(node_type='op', op='bogus_op', children=[Node.feature('close')])
    # 同时确认走 parse_formula 字符串入口会在解析阶段抛错（不会漏到 validator）
    with pytest.raises(ValueError):
        parse_formula('bogus_op(close)', reg)
    # validator 语法层独立拦截
    res = validate_expression(bogus, reg)
    assert res['valid'] is False
    assert res['layer'] == '语法'
    assert 'bogus_op' in res['reason']


def test_m11_registry_unregistered_feature_via_string_blocked():
    """注册表层：字符串里含某特征，但校验期注册表未登记 → validator 拦下。

    模拟真实错配：用「含 close 的注册表」解析公式字符串成功，再换「缺 close 的
    注册表」去 validate，注册表层应判非法（layer='注册表'）。
    """
    reg_full = _make_registry()
    node = parse_formula('add(close, turnover_ratio)', reg_full)  # 解析期 OK
    # 换一个不含 close 的注册表（只留 turnover_ratio）
    reg_missing = FeatureRegistry(seed_defaults=False)
    reg_missing.register('turnover_ratio', unit='dimensionless',
                         semantic='liquidity', description='换手率')
    res = validate_expression(node, reg_missing)
    assert res['valid'] is False
    assert res['layer'] == '注册表'
    assert 'close' in res['reason']


def test_m11_dimension_incompatible_add_via_string_blocked():
    """量纲层：字符串 add(close, turnover_ratio)（price + dimensionless）
    经 parse_formula 解析后应被量纲层拦下（layer='量纲'）。"""
    reg = _make_registry()
    node = parse_formula('add(close, turnover_ratio)', reg)
    res = validate_expression(node, reg)
    assert res['valid'] is False
    assert res['layer'] == '量纲'
    assert 'price' in res['reason'] and 'dimensionless' in res['reason']


def test_m11_lookahead_negative_window_via_string_blocked():
    """前视层：字符串 delay(close, -1)（一元负号折叠成负窗口）经 parse_formula
    解析后应被前视层拦下（layer='前视'）。"""
    reg = _make_registry()
    node = parse_formula('delay(close, -1)', reg)
    # 确认解析确实把 -1 折叠成常数窗口 -1
    assert node.op == 'delay'
    assert node.children[-1].node_type == 'const' and node.children[-1].value == -1
    res = validate_expression(node, reg)
    assert res['valid'] is False
    assert res['layer'] == '前视'
    assert '-1' in res['reason']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

# -*- coding: utf-8 -*-
"""
tests_dimensions.py — dimensions.py 组合表 / 兼容规则单测

覆盖双维度量纲体系的地基规则（纯函数，无数据依赖，全可复现）：
    - 乘法组合表：价格×计数→金额，无量纲传导，兜底 dimensionless，unknown 传播
    - 除法组合表：金额÷计数→价格，金额÷价格→计数，同实体相除→无量纲
    - 加减兼容：实体严格同单位、无量纲之间放宽、unknown 通配
    - merge_units：取更具体单位

运行：pytest tests_dimensions.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dimensions import (  # noqa: E402
    UNITS, SEMANTICS,
    combine_mul, combine_div, add_sub_compatible, merge_units,
)


# ---------------------------------------------------------------------------
# 枚举完整性
# ---------------------------------------------------------------------------
def test_units_enum():
    assert UNITS == {'price', 'money', 'count', 'dimensionless', 'bool', 'unknown'}


def test_semantics_has_core():
    for s in ('price', 'return', 'momentum', 'volatility', 'volume',
              'turnover', 'open_interest', 'unknown'):
        assert s in SEMANTICS


# ---------------------------------------------------------------------------
# 乘法组合表
# ---------------------------------------------------------------------------
def test_mul_price_count_is_money():
    # 价格 × 计数 = 金额（成交价 × 成交手数 = 成交额），可交换
    assert combine_mul('price', 'count') == 'money'
    assert combine_mul('count', 'price') == 'money'


def test_mul_dimensionless_transmits():
    # X × 无量纲 = X
    assert combine_mul('price', 'dimensionless') == 'price'
    assert combine_mul('dimensionless', 'money') == 'money'
    assert combine_mul('count', 'bool') == 'count'
    # 无量纲 × 无量纲 = 无量纲
    assert combine_mul('dimensionless', 'bool') == 'dimensionless'


def test_mul_unknown_propagates():
    assert combine_mul('unknown', 'price') == 'unknown'
    assert combine_mul('money', 'unknown') == 'unknown'


def test_mul_fallback_dimensionless():
    # 无明确金融意义的实体组合 → 兜底无量纲（乘法放宽，不报错）
    assert combine_mul('price', 'price') == 'dimensionless'
    assert combine_mul('price', 'money') == 'dimensionless'
    assert combine_mul('money', 'money') == 'dimensionless'
    assert combine_mul('count', 'count') == 'dimensionless'


# ---------------------------------------------------------------------------
# 除法组合表
# ---------------------------------------------------------------------------
def test_div_money_over_count_is_price():
    # 金额 ÷ 计数 = 价格（成交额 / 成交量 = 均价）
    assert combine_div('money', 'count') == 'price'


def test_div_money_over_price_is_count():
    # 金额 ÷ 价格 = 计数
    assert combine_div('money', 'price') == 'count'


def test_div_same_entity_is_dimensionless():
    assert combine_div('price', 'price') == 'dimensionless'
    assert combine_div('money', 'money') == 'dimensionless'
    assert combine_div('count', 'count') == 'dimensionless'


def test_div_by_dimensionless_transmits():
    # X ÷ 无量纲 = X
    assert combine_div('price', 'dimensionless') == 'price'
    assert combine_div('money', 'bool') == 'money'


def test_div_dimensionless_over_entity():
    # 无量纲 ÷ 实体 → 无量纲
    assert combine_div('dimensionless', 'price') == 'dimensionless'


def test_div_unknown_propagates():
    assert combine_div('unknown', 'count') == 'unknown'
    assert combine_div('price', 'unknown') == 'unknown'


def test_div_fallback_dimensionless():
    # 无表项的实体组合（如 price÷count、price÷money）→ 兜底无量纲
    assert combine_div('price', 'count') == 'dimensionless'
    assert combine_div('count', 'money') == 'dimensionless'


# ---------------------------------------------------------------------------
# 加减兼容：实体严、无量纲宽
# ---------------------------------------------------------------------------
def test_add_sub_same_entity_ok():
    assert add_sub_compatible('price', 'price') is True
    assert add_sub_compatible('money', 'money') is True
    assert add_sub_compatible('count', 'count') is True


def test_add_sub_cross_entity_blocked():
    # 修复 m13 核心：金额+成交量、价格+金额 都必须拦
    assert add_sub_compatible('money', 'count') is False
    assert add_sub_compatible('price', 'money') is False
    assert add_sub_compatible('price', 'count') is False


def test_add_sub_dimensionless_relaxed():
    # 无量纲之间放宽（比率、标准化后、bool 都是纯数）
    assert add_sub_compatible('dimensionless', 'dimensionless') is True
    assert add_sub_compatible('dimensionless', 'bool') is True
    assert add_sub_compatible('bool', 'bool') is True


def test_add_sub_unknown_wildcard():
    # unknown（常数 / 未标注）通配兼容任意单位
    assert add_sub_compatible('unknown', 'price') is True
    assert add_sub_compatible('money', 'unknown') is True


def test_add_sub_entity_vs_dimensionless_blocked():
    # 实体 与 无量纲 不兼容（价格 + 比率 应拦）
    assert add_sub_compatible('price', 'dimensionless') is False
    assert add_sub_compatible('count', 'bool') is False


# ---------------------------------------------------------------------------
# merge_units
# ---------------------------------------------------------------------------
def test_merge_units():
    assert merge_units('price', 'price') == 'price'
    assert merge_units('unknown', 'money') == 'money'
    assert merge_units('count', 'unknown') == 'count'
    # 两个不同无量纲 → 归一 dimensionless
    assert merge_units('dimensionless', 'bool') == 'dimensionless'

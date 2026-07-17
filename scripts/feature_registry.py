# -*- coding: utf-8 -*-
"""
feature_registry.py — 动态特征注册表（双维度量纲：物理单位 × 金融语义）

不写死白名单：特征在运行时动态注册（可由 DataFrame 批量注册，
也可由量纲推断器回填 unit/semantic）。供 LLM prompt 生成特征清单、
供表达式引擎校验特征可用性。

每条特征记录结构：
    {
        'name':        str,   # 特征名
        'unit':        str,   # 物理单位（硬约束）: price/money/count/dimensionless/bool/unknown
        'semantic':    str,   # 金融语义（软标注）: price/return/momentum/volatility/...
        'dtype':       str,   # 数据类型标签（'float'/'int'/'bool' 等）
        'description': str,   # 人类可读说明
    }

量纲拆成两个正交维度（详见 dimensions.py）：
    - unit    参与合法性校验（加减同单位、乘除靠组合表推导）。
    - semantic 只用于喂 LLM 打分/解释/报告，不参与校验、不进 GP。
"""
from __future__ import annotations

import pandas as pd

from dimensions import UNITS, SEMANTICS

# 合法单位 / 语义标签（从 dimensions 单一源引入，避免多处漂移）
VALID_UNITS = UNITS
VALID_SEMANTICS = SEMANTICS

# 内置标准量价特征的默认单位
_DEFAULT_UNITS: dict[str, str] = {
    # 价格类
    'open': 'price', 'high': 'price', 'low': 'price', 'close': 'price',
    'vwap': 'price', 'twap': 'price', 'mid': 'price', 'settle': 'price',
    # 金额（成交额，单位元）
    'amount': 'money', 'turnover_amt': 'money',
    # 计数（成交量/持仓量，单位手/张）
    'volume': 'count', 'turnover': 'count', 'vol': 'count',
    'open_interest': 'count', 'oi': 'count',
    # 收益率类（无量纲）
    'ret': 'dimensionless', 'return': 'dimensionless', 'ret_1': 'dimensionless',
    'ret_5': 'dimensionless', 'log_ret': 'dimensionless',
    'log_return': 'dimensionless', 'pct_change': 'dimensionless',
}

# 内置标准量价特征的默认语义
_DEFAULT_SEMANTICS: dict[str, str] = {
    'open': 'price', 'high': 'price', 'low': 'price', 'close': 'price',
    'vwap': 'price', 'twap': 'price', 'mid': 'price', 'settle': 'price',
    'amount': 'turnover', 'turnover_amt': 'turnover',
    'volume': 'volume', 'turnover': 'volume', 'vol': 'volume',
    'open_interest': 'open_interest', 'oi': 'open_interest',
    'ret': 'return', 'return': 'return', 'ret_1': 'return', 'ret_5': 'return',
    'log_ret': 'return', 'log_return': 'return', 'pct_change': 'return',
}


class FeatureRegistry:
    """管理特征的动态注册表（双维度量纲）。"""

    def __init__(self, seed_defaults: bool = True):
        """初始化。

        参数：
            seed_defaults: 是否预置内置标准量价特征的默认单位/语义（默认 True）。
                           注意：只预置「默认映射」，特征本身仍需 register 后才算已注册。
        """
        self._features: dict[str, dict] = {}
        # 预置默认映射（作为 register 时 unit/semantic 的默认回退）
        if seed_defaults:
            self._default_units = dict(_DEFAULT_UNITS)
            self._default_semantics = dict(_DEFAULT_SEMANTICS)
        else:
            self._default_units = {}
            self._default_semantics = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(self, name: str, unit: str | None = None,
                 semantic: str | None = None,
                 dtype: str = 'float', description: str = '') -> dict:
        """注册一个特征。

        参数：
            name:        特征名
            unit:        物理单位；None 时按内置默认映射推断，仍无则 'unknown'
            semantic:    金融语义；None 时按内置默认映射推断，仍无则 'unknown'
            dtype:       数据类型标签
            description: 说明文本

        返回：注册后的特征记录 dict。
        """
        if unit is None:
            unit = self._default_units.get(name, 'unknown')
        if semantic is None:
            semantic = self._default_semantics.get(name, 'unknown')
        if unit not in VALID_UNITS:
            raise ValueError(
                f"非法单位 {unit!r}，合法取值: {sorted(VALID_UNITS)}")
        if semantic not in VALID_SEMANTICS:
            raise ValueError(
                f"非法语义 {semantic!r}，合法取值: {sorted(VALID_SEMANTICS)}")
        record = {
            'name': name,
            'unit': unit,
            'semantic': semantic,
            'dtype': dtype,
            'description': description,
        }
        self._features[name] = record
        return record

    def register_dataframe(self, df: pd.DataFrame,
                           unit_map: dict[str, str] | None = None,
                           semantic_map: dict[str, str] | None = None) -> None:
        """批量把 DataFrame 的列注册进来。

        参数：
            df:           数据表，逐列注册
            unit_map:     {列名: 单位} 显式覆盖；未提供的列按内置默认映射 / unknown
            semantic_map: {列名: 语义} 显式覆盖；未提供的列按内置默认映射 / unknown
        """
        unit_map = unit_map or {}
        semantic_map = semantic_map or {}
        for col in df.columns:
            col = str(col)
            # 用列的 dtype 生成一个粗粒度 dtype 标签
            pd_dtype = str(df[col].dtype)
            if pd_dtype.startswith('bool'):
                dtype_tag = 'bool'
            elif pd_dtype.startswith('int'):
                dtype_tag = 'int'
            else:
                dtype_tag = 'float'
            self.register(col, unit=unit_map.get(col),
                          semantic=semantic_map.get(col), dtype=dtype_tag)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, name: str) -> dict:
        """取特征记录；不存在抛 KeyError。"""
        if name not in self._features:
            raise KeyError(f"特征 {name!r} 未注册")
        return self._features[name]

    def has(self, name: str) -> bool:
        """是否已注册该特征。"""
        return name in self._features

    def list_features(self) -> list[str]:
        """已注册特征名列表（按注册顺序）。"""
        return list(self._features.keys())

    def feature_names(self) -> set[str]:
        """已注册特征名集合。"""
        return set(self._features.keys())

    # ------------------------------------------------------------------
    # 单位（硬约束）
    # ------------------------------------------------------------------
    def get_unit(self, name: str) -> str:
        """取特征物理单位；未注册抛 KeyError。"""
        return self.get(name)['unit']

    def set_unit(self, name: str, unit: str) -> None:
        """回填 / 修改特征单位（供量纲推断器使用）。未注册则顺带注册。"""
        if unit not in VALID_UNITS:
            raise ValueError(
                f"非法单位 {unit!r}，合法取值: {sorted(VALID_UNITS)}")
        if name not in self._features:
            self.register(name, unit=unit)
        else:
            self._features[name]['unit'] = unit

    # ------------------------------------------------------------------
    # 语义（软标注）
    # ------------------------------------------------------------------
    def get_semantic(self, name: str) -> str:
        """取特征金融语义；未注册抛 KeyError。"""
        return self.get(name)['semantic']

    def set_semantic(self, name: str, semantic: str) -> None:
        """回填 / 修改特征语义（供量纲推断器使用）。未注册则顺带注册。"""
        if semantic not in VALID_SEMANTICS:
            raise ValueError(
                f"非法语义 {semantic!r}，合法取值: {sorted(VALID_SEMANTICS)}")
        if name not in self._features:
            self.register(name, semantic=semantic)
        else:
            self._features[name]['semantic'] = semantic

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def to_prompt_list(self) -> str:
        """生成给 LLM prompt 用的特征清单文本。

        格式（每行一个特征）：
            - <name> [<unit>/<semantic>]: <description>
        """
        lines = []
        for name, rec in self._features.items():
            desc = rec['description'] or '(无说明)'
            lines.append(
                f"- {name} [{rec['unit']}/{rec['semantic']}]: {desc}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._features)} features)"

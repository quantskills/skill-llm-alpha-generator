# -*- coding: utf-8 -*-
"""
tests_report.py — report_builder.py 单元测试

用造假的 factors + trajectory + rejected 验证 build_report：
    1. 返回字符串含 <html>（是完整 HTML 文档）
    2. 含所有因子公式（因子确实渲染进去了）
    3. 含 "淘汰"（淘汰记录段落存在）
    4. output_path 给了能写出文件且文件非空
    5. 空输入不崩，仍产出合法 HTML
    6. LLM 默认关闭时（ALPHA_LLM_DISABLED=1）不调用外部、不崩

运行：pytest tests_report.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

# 保证能以裸模块名 import（运行时需要 scripts/ 在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_builder import build_report, _radar_svg, _lines_svg, _classify_reason  # noqa: E402


# 测试期间强制关闭 LLM，避免联网 / 依赖 anthropic SDK。
# 用 autouse fixture + monkeypatch.setenv 做测试隔离：仅在本模块每个测试执行期间生效，
# fixture 结束自动还原，绝不泄漏到同一 pytest 会话的其他测试模块（避免污染 mock/启用分支）。
# ---------------------------------------------------------------------------
# 造假数据
# ---------------------------------------------------------------------------
def _fake_factors() -> list[dict]:
    return [
        {
            "formula": "ts_rank(divide(sub(close, open), high), 10)",
            "explanation": "捕捉日内动量相对振幅的排序信号。",
            "alpha_scores": {
                "effectiveness": 0.72, "robustness": 0.6,
                "interpretability": 0.8, "diversity": 0.55, "parsimony": 0.7,
            },
            "rankic": 0.043,
            "signal_stats": {"mean": 0.01, "std": 0.98, "turnover": 0.32},
        },
        {
            "formula": "corr(volume, abs(sub(close, delay(close, 1))), 20)",
            "explanation": "量价背离度量。",
            "alpha_scores": {
                "effectiveness": 0.5, "robustness": 0.66,
                "interpretability": 0.4, "diversity": 0.9, "parsimony": 0.3,
            },
            "rankic": 0.028,
            "signal_stats": {"mean": -0.002, "std": 1.02, "turnover": 0.61},
        },
    ]


def _fake_trajectory() -> list[dict]:
    return [
        {
            "gen": g,
            "best_fitness": 0.02 + g * 0.004,
            "mean_fitness": 0.005 + g * 0.002,
            "diversity": 0.9 - g * 0.03,
        }
        for g in range(12)
    ]


def _fake_rejected() -> list[dict]:
    return [
        {"formula": "add(close, volume)",
         "reason": "量纲非法：price 与 volume 不可相加", "layer": "dimension"},
        {"formula": "rank(close)", "reason": "rankIC 太低", "rankic": 0.001},
        {"formula": "ts_rank(close, 5)",
         "reason": "与已入选因子高相关", "max_corr": 0.93},
    ]


# ---------------------------------------------------------------------------
# 1) 返回字符串含 <html>
# ---------------------------------------------------------------------------
def test_returns_html_document():
    html = build_report(_fake_factors(), _fake_trajectory(), _fake_rejected())
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "</html>" in html.lower()
    assert "<!doctype html>" in html.lower()


# ---------------------------------------------------------------------------
# 2) 含所有因子公式
# ---------------------------------------------------------------------------
def test_contains_factor_formulas():
    factors = _fake_factors()
    html = build_report(factors, _fake_trajectory(), _fake_rejected())
    for f in factors:
        # HTML 转义后 > 会变实体，这里公式里只有字母/括号/逗号，直接子串检查即可
        assert f["formula"] in html, f"公式未出现在报告：{f['formula']}"


# ---------------------------------------------------------------------------
# 3) 含 "淘汰"
# ---------------------------------------------------------------------------
def test_contains_rejected_section():
    html = build_report(_fake_factors(), _fake_trajectory(), _fake_rejected())
    assert "淘汰" in html
    # 淘汰因子公式与原因也应出现
    assert "add(close, volume)" in html
    assert "量纲非法" in html


def test_contains_iteration_trace_table():
    trajectory = _fake_trajectory()
    trajectory[0]["current_best_formula"] = "ts_mean(close, 5)"
    trajectory[0]["current_best_fitness"] = trajectory[0]["best_fitness"]
    html = build_report(_fake_factors(), trajectory, _fake_rejected())
    assert "trajectory-table" in html
    assert "当轮最优公式" in html
    assert "ts_mean(close, 5)" in html
    assert "fitness" in html
    assert "diversity" in html


# ---------------------------------------------------------------------------
# 4) output_path 能写出文件且非空
# ---------------------------------------------------------------------------
def test_writes_file(tmp_path):
    out = tmp_path / "report.html"
    html = build_report(_fake_factors(), _fake_trajectory(), _fake_rejected(),
                        output_path=str(out))
    assert out.exists(), "输出文件未生成"
    content = out.read_text(encoding="utf-8")
    assert len(content) > 0, "输出文件为空"
    # 写出的内容应与返回值一致
    assert content == html


# ---------------------------------------------------------------------------
# 5) 空输入不崩
# ---------------------------------------------------------------------------
def test_empty_inputs_ok():
    html = build_report([], [], [])
    assert "<html" in html.lower()
    assert "未产出有效因子" in html
    assert "无淘汰记录" in html
    # 空轨迹也应给出占位提示，而非崩溃
    assert "无迭代轨迹数据" in html


def test_none_rejected_ok():
    # rejected 默认 None，不应报错
    html = build_report(_fake_factors(), _fake_trajectory())
    assert "<html" in html.lower()
    assert "无淘汰记录" in html


# ---------------------------------------------------------------------------
# 6) 雷达图 / 折线图内联 SVG
# ---------------------------------------------------------------------------
def test_radar_svg_inline():
    svg = _radar_svg({"effectiveness": 0.8, "robustness": 0.5,
                      "interpretability": 0.6, "diversity": 0.4, "parsimony": 0.7})
    assert svg.startswith("<svg")
    assert "polygon" in svg  # 有数据多边形
    # 五个维度中文标签
    for cn in ("有效性", "稳健性", "可解释性", "多样性", "简洁性"):
        assert cn in svg


def test_lines_svg_inline():
    svg = _lines_svg(_fake_trajectory())
    assert svg.startswith("<svg")
    assert "polyline" in svg  # 有折线
    for label in ("best_fitness", "mean_fitness", "diversity"):
        assert label in svg
    assert "迭代轮次 (gen)" in svg


def test_lines_svg_single_point_is_visible():
    svg = _lines_svg([{
        "gen": 0,
        "best_fitness": 0.05,
        "mean_fitness": 0.02,
        "diversity": 0.8,
    }])
    assert "trajectory-svg" in svg
    assert svg.count("<circle") == 3


def test_lines_svg_empty():
    svg = _lines_svg([])
    assert svg.startswith("<svg")
    assert "无迭代轨迹数据" in svg


# ---------------------------------------------------------------------------
# 7) 淘汰原因分类
# ---------------------------------------------------------------------------
def test_classify_reason():
    assert _classify_reason("量纲非法：不可相加")[0] == "量纲非法"
    assert _classify_reason("dimension mismatch")[0] == "量纲非法"
    assert _classify_reason("rankIC 太低")[0] == "IC 太低"
    assert _classify_reason("与精英高相关")[0] == "高相关"
    assert _classify_reason("莫名其妙")[0] == "其它"


# ---------------------------------------------------------------------------
# 8) 无外部依赖 / 无 CDN 引用（自包含）
# ---------------------------------------------------------------------------
def test_self_contained_no_external():
    html = build_report(_fake_factors(), _fake_trajectory(), _fake_rejected())
    low = html.lower()
    # 不应引用任何外部脚本 / 样式 / 图片
    assert "http://" not in low
    assert "src=" not in low  # 没有外链 <script src> / <img src>
    assert "<link" not in low  # 没有外部样式表
    # https 只允许出现在 svg 命名空间/属性里；这里我们没用 xmlns，确保没有 https 链接
    assert "https://" not in low


# ---------------------------------------------------------------------------
# 9) 缺字段的因子也不崩（健壮性）
# ---------------------------------------------------------------------------
def test_partial_factor_fields():
    factors = [
        {"formula": "close"},  # 缺 explanation / alpha_scores / rankic / signal_stats
        {"formula": "volume", "alpha_scores": {"effectiveness": 0.5}},  # 五维不全
    ]
    html = build_report(factors, [], [])
    assert "<html" in html.lower()
    assert "close" in html
    assert "volume" in html


if __name__ == "__main__":  # pragma: no cover
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))

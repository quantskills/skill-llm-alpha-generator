# -*- coding: utf-8 -*-
"""
report_builder.py — 自包含 HTML 可视化报告生成器

把一次 alpha 挖掘的结果（因子集合 + GP 迭代轨迹 + 淘汰记录）渲染成
**单文件 HTML**：所有 CSS/JS/图表全部内联，不依赖任何外部 CDN、
不依赖 matplotlib/plotly，双击即可在浏览器打开。

三部分结构：
    ① 顶部因子卡片：公式 / 经济解释 / AlphaEval 五维雷达图(内联 SVG) / rankIC / 信号统计
    ② 中部 GP 迭代追溯：best_fitness / mean_fitness / diversity 三条折线(内联 SVG)
    ③ 底部淘汰记录：每个被淘汰因子 + 淘汰原因(量纲非法/IC 太低/高相关) 表格

对外主入口：
    build_report(factors, trajectory, rejected=None, output_path=None) -> str
        返回完整 HTML 字符串；output_path 非空时同时写文件。

LLM is forbidden here. This module is a render-only consumer of already
validated factor records; all LLM calls happen before build_report().
"""
from __future__ import annotations

import html
import json
import math
import os
from typing import Any

# AlphaEval 五维（雷达图的五个轴，顺序固定）
_ALPHA_DIMS: list[tuple[str, str]] = [
    ("effectiveness", "有效性"),
    ("robustness", "稳健性"),
    ("interpretability", "可解释性"),
    ("diversity", "多样性"),
    ("parsimony", "简洁性"),
]

# 一套协调的深色配色（暗色主题，护眼且对比清晰）
_PALETTE = {
    "bg": "#0f1420",          # 页面底
    "card": "#1a2130",        # 卡片底
    "card_alt": "#141a26",    # 次级底（表格斑马纹）
    "border": "#2a3448",      # 边框
    "text": "#e6ebf5",        # 主文字
    "muted": "#8a97b0",       # 次要文字
    "accent": "#5b9dff",      # 主强调（best_fitness / 雷达描边）
    "accent2": "#3ddc97",     # 次强调（mean_fitness）
    "accent3": "#ffb454",     # 三强调（diversity）
    "danger": "#ff6b7d",      # 淘汰/危险
    "radar_fill": "rgba(91,157,255,0.28)",
}


# ===========================================================================
# 小工具：安全取值 / 数字格式化 / HTML 转义
# ===========================================================================
def _esc(text: Any) -> str:
    """HTML 转义，None → 空串。"""
    if text is None:
        return ""
    return html.escape(str(text))


def _fmt(value: Any, digits: int = 4) -> str:
    """把数字格式化为字符串；非数字/缺失 → '—'。"""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if math.isnan(f) or math.isinf(f):
        return "—"
    # 整数就不带小数
    if abs(f - round(f)) < 1e-12 and abs(f) < 1e15:
        return str(int(round(f)))
    return f"{f:.{digits}f}"


def _clip01(value: Any) -> float:
    """把任意值夹到 [0,1]，非数字→0。雷达五维分数用。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return min(1.0, max(0.0, f))


# ===========================================================================
# 内联 SVG：AlphaEval 五维雷达图
# ===========================================================================
def _radar_svg(scores: dict, size: int = 260) -> str:
    """画一张五维雷达图（纯内联 SVG，无外部依赖）。

    参数：
        scores: 五维分数 dict，键为 _ALPHA_DIMS 里的英文名，值域 [0,1]
        size:   SVG 边长（正方形）

    返回：<svg>...</svg> 字符串。
    """
    cx = cy = size / 2.0
    radius = size / 2.0 - 70  # Keep axis labels inside the SVG viewBox.
    n = len(_ALPHA_DIMS)
    scores = scores or {}

    def _point(idx: int, r: float) -> tuple[float, float]:
        # 从正上方(12 点钟)开始，顺时针分布
        ang = -math.pi / 2 + 2 * math.pi * idx / n
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    parts: list[str] = []

    # 背景网格：4 层同心多边形
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                       (_point(i, radius * ring) for i in range(n)))
        parts.append(
            f'<polygon points="{pts}" fill="none" '
            f'stroke="{_PALETTE["border"]}" stroke-width="1"/>'
        )

    # 从中心到各轴顶点的射线 + 轴标签
    for i, (key, label_cn) in enumerate(_ALPHA_DIMS):
        ax, ay = _point(i, radius)
        parts.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
            f'stroke="{_PALETTE["border"]}" stroke-width="1"/>'
        )
        # Labels are placed near the axis tip, then clamped inside the
        # viewBox so long Chinese labels and scores are never clipped.
        lx, ly = _point(i, radius + 36)
        val = _clip01(scores.get(key))
        anchor = "middle"
        if lx < cx - 4:
            anchor = "end"
            lx = max(lx, 76)
        elif lx > cx + 4:
            anchor = "start"
            lx = min(lx, size - 76)
        ly = min(max(ly, 20), size - 22)
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{_PALETTE["muted"]}" '
            f'font-size="11" text-anchor="{anchor}" dominant-baseline="middle">'
            f'{_esc(label_cn)} {val:.2f}</text>'
        )

    # 数据多边形
    data_pts = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in
        (_point(i, radius * _clip01(scores.get(key)))
         for i, (key, _) in enumerate(_ALPHA_DIMS))
    )
    parts.append(
        f'<polygon points="{data_pts}" fill="{_PALETTE["radar_fill"]}" '
        f'stroke="{_PALETTE["accent"]}" stroke-width="2"/>'
    )
    # 数据顶点小圆
    for i, (key, _) in enumerate(_ALPHA_DIMS):
        px, py = _point(i, radius * _clip01(scores.get(key)))
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" '
            f'fill="{_PALETTE["accent"]}"/>'
        )

    return (f'<svg class="radar-svg" viewBox="0 0 {size} {size}" '
            f'width="100%" height="{size}" role="img" '
            f'aria-label="AlphaEval 五维雷达图">{"".join(parts)}</svg>')


# ===========================================================================
# 内联 SVG：GP 迭代折线图
# ===========================================================================
def _legacy_lines_svg(trajectory: list[dict], width: int = 720, height: int = 320) -> str:
    """画 best_fitness / mean_fitness / diversity 三条折线（内联 SVG）。

    参数：
        trajectory: 每代一个 dict，含 gen / best_fitness / mean_fitness / diversity
        width/height: SVG 尺寸

    返回：<svg>...</svg> 字符串；轨迹为空时返回一段占位提示。
    """
    if not trajectory:
        return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
                f'height="{height}"><text x="{width/2}" y="{height/2}" '
                f'fill="{_PALETTE["muted"]}" font-size="14" '
                f'text-anchor="middle">（无迭代轨迹数据）</text></svg>')

    pad_l, pad_r, pad_t, pad_b = 52, 16, 20, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # x 轴：代数（用 gen 字段，缺失则用序号）
    gens = [t.get("gen", i) for i, t in enumerate(trajectory)]
    gmin, gmax = min(gens), max(gens)
    gspan = (gmax - gmin) or 1

    series = {
        "best_fitness": (_PALETTE["accent"], "best_fitness"),
        "mean_fitness": (_PALETTE["accent2"], "mean_fitness"),
        "diversity": (_PALETTE["accent3"], "diversity"),
    }

    # y 轴范围：把三条线的所有有效值放一起取 min/max
    all_vals: list[float] = []
    for t in trajectory:
        for key in series:
            v = t.get(key)
            try:
                fv = float(v)
                if not (math.isnan(fv) or math.isinf(fv)):
                    all_vals.append(fv)
            except (TypeError, ValueError):
                continue
    if not all_vals:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(all_vals), max(all_vals)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        else:
            margin = (vmax - vmin) * 0.08
            vmin -= margin
            vmax += margin
    vspan = (vmax - vmin) or 1

    def _sx(g: float) -> float:
        return pad_l + plot_w * (g - gmin) / gspan

    def _sy(v: float) -> float:
        return pad_t + plot_h * (1 - (v - vmin) / vspan)

    parts: list[str] = []

    # y 轴网格 + 刻度（5 条水平线）
    for i in range(5):
        frac = i / 4.0
        y = pad_t + plot_h * frac
        val = vmax - (vmax - vmin) * frac
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="{_PALETTE["border"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y:.1f}" fill="{_PALETTE["muted"]}" '
            f'font-size="10" text-anchor="end" dominant-baseline="middle">'
            f'{val:.3f}</text>'
        )

    # x 轴刻度（最多 8 个代数标签，避免拥挤）
    step = max(1, len(trajectory) // 8)
    for i in range(0, len(trajectory), step):
        g = gens[i]
        x = _sx(g)
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 16}" '
            f'fill="{_PALETTE["muted"]}" font-size="10" text-anchor="middle">'
            f'{_esc(g)}</text>'
        )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 4}" '
        f'fill="{_PALETTE["muted"]}" font-size="11" text-anchor="middle">迭代代数 (gen)</text>'
    )

    # 三条折线
    for key, (color, _label) in series.items():
        pts: list[str] = []
        for i, t in enumerate(trajectory):
            v = t.get(key)
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    continue
            except (TypeError, ValueError):
                continue
            pts.append(f"{_sx(gens[i]):.1f},{_sy(fv):.1f}")
        if not pts:
            continue
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{color}" stroke-width="2.2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        # 端点小圆
        for p in pts:
            px, py = p.split(",")
            parts.append(f'<circle cx="{px}" cy="{py}" r="2.6" fill="{color}"/>')

    # 图例
    lx = pad_l + 6
    ly = pad_t + 6
    for key, (color, label) in series.items():
        parts.append(
            f'<rect x="{lx}" y="{ly - 8}" width="14" height="4" rx="2" fill="{color}"/>'
            f'<text x="{lx + 20}" y="{ly - 6}" fill="{_PALETTE["text"]}" '
            f'font-size="11" dominant-baseline="middle">{_esc(label)}</text>'
        )
        lx += 20 + 10 * len(label) + 24

    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'role="img" aria-label="GP 迭代折线图">{"".join(parts)}</svg>')


def _lines_svg(trajectory: list[dict], width: int = 760, height: int = 360) -> str:
    """Render fitness and diversity on separate axes with stable x positions."""
    if not trajectory:
        return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
                f'height="{height}"><text x="{width/2}" y="{height/2}" '
                f'fill="{_PALETTE["muted"]}" font-size="14" '
                f'text-anchor="middle">（无迭代轨迹数据）</text></svg>')

    pad_l, pad_r, pad_t, pad_b = 72, 72, 34, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n_points = len(trajectory)

    def _number(value):
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    gens = [t.get("gen", i) for i, t in enumerate(trajectory)]
    numeric_gens = [_number(g) for g in gens]
    valid_gens = [g for g in numeric_gens if g is not None]
    gmin = min(valid_gens, default=0.0)
    gmax = max(valid_gens, default=float(max(1, n_points - 1)))
    gspan = gmax - gmin

    def _sx(index: int) -> float:
        if n_points == 1:
            return pad_l + plot_w / 2.0
        if gspan > 0 and numeric_gens[index] is not None:
            return pad_l + plot_w * (numeric_gens[index] - gmin) / gspan
        return pad_l + plot_w * index / (n_points - 1)

    def _axis_range(keys: tuple[str, ...], default: tuple[float, float]) -> tuple[float, float]:
        values = [_number(item.get(key)) for item in trajectory for key in keys]
        values = [value for value in values if value is not None]
        if not values:
            return default
        lo, hi = min(values), max(values)
        margin = max(abs(lo) * 0.12, 0.01) if lo == hi else (hi - lo) * 0.12
        return lo - margin, hi + margin

    fitness_min, fitness_max = _axis_range(("best_fitness", "mean_fitness"), (-0.1, 0.1))
    diversity_min, diversity_max = _axis_range(("diversity",), (0.0, 1.0))

    def _sy(value: float, lo: float, hi: float) -> float:
        span = hi - lo or 1.0
        return pad_t + plot_h * (1.0 - (value - lo) / span)

    parts: list[str] = []
    for i in range(5):
        frac = i / 4.0
        y = pad_t + plot_h * frac
        left_val = fitness_max - (fitness_max - fitness_min) * frac
        right_val = diversity_max - (diversity_max - diversity_min) * frac
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="{_PALETTE["border"]}" stroke-width="1"/>'
            f'<text x="{pad_l - 9}" y="{y:.1f}" fill="{_PALETTE["muted"]}" '
            f'font-size="10" text-anchor="end" dominant-baseline="middle">{left_val:.3f}</text>'
            f'<text x="{pad_l + plot_w + 9}" y="{y:.1f}" fill="{_PALETTE["muted"]}" '
            f'font-size="10" text-anchor="start" dominant-baseline="middle">{right_val:.2f}</text>'
        )

    parts.append(
        f'<text x="{pad_l}" y="16" fill="{_PALETTE["muted"]}" font-size="10">fitness</text>'
        f'<text x="{pad_l + plot_w}" y="16" fill="{_PALETTE["muted"]}" '
        f'font-size="10" text-anchor="end">diversity</text>'
    )

    tick_indices = list(range(0, n_points, max(1, math.ceil(n_points / 8))))
    if n_points > 1 and tick_indices[-1] != n_points - 1:
        tick_indices.append(n_points - 1)
    for index in tick_indices:
        parts.append(
            f'<text x="{_sx(index):.1f}" y="{pad_t + plot_h + 18}" '
            f'fill="{_PALETTE["muted"]}" font-size="10" text-anchor="middle">'
            f'{_esc(gens[index])}</text>'
        )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 5}" '
        f'fill="{_PALETTE["muted"]}" font-size="11" text-anchor="middle">迭代轮次 (gen)</text>'
    )

    series = (
        ("best_fitness", _PALETTE["accent"], fitness_min, fitness_max),
        ("mean_fitness", _PALETTE["accent2"], fitness_min, fitness_max),
        ("diversity", _PALETTE["accent3"], diversity_min, diversity_max),
    )
    for key, color, lo, hi in series:
        points: list[str] = []
        point_indices: list[int] = []
        for index, item in enumerate(trajectory):
            value = _number(item.get(key))
            if value is None:
                continue
            points.append(f"{_sx(index):.1f},{_sy(value, lo, hi):.1f}")
            point_indices.append(index)
        if not points:
            continue
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for point, index in zip(points, point_indices):
            px, py = point.split(",")
            formula = trajectory[index].get("current_best_formula") or trajectory[index].get("best_formula") or ""
            title = f'gen {gens[index]}: {key}={_fmt(trajectory[index].get(key), 5)}'
            if formula and key == "best_fitness":
                title += f' | {formula}'
            parts.append(
                f'<circle cx="{px}" cy="{py}" r="3" fill="{color}">'
                f'<title>{_esc(title)}</title></circle>'
            )

    labels = (
        ("best_fitness", _PALETTE["accent"]),
        ("mean_fitness", _PALETTE["accent2"]),
        ("diversity", _PALETTE["accent3"]),
    )
    lx, ly = pad_l + 8, pad_t + 8
    for label, color in labels:
        parts.append(
            f'<rect x="{lx}" y="{ly - 8}" width="14" height="4" rx="2" fill="{color}"/>'
            f'<text x="{lx + 20}" y="{ly - 6}" fill="{_PALETTE["text"]}" '
            f'font-size="11" dominant-baseline="middle">{label}</text>'
        )
        lx += 20 + 10 * len(label) + 24

    return (f'<svg class="trajectory-svg" viewBox="0 0 {width} {height}" '
            f'width="100%" height="{height}" role="img" aria-label="GP 迭代折线图">'
            f'{"".join(parts)}</svg>')


def _trajectory_table_html(trajectory: list[dict]) -> str:
    """Render one auditable row for every recorded GP generation."""
    if not trajectory:
        return '<p class="empty">无逐轮迭代记录。</p>'

    rows = []
    for item in trajectory:
        formula = item.get("current_best_formula") or item.get("best_formula") or "—"
        rows.append(
            f'<tr><td>{_esc(item.get("gen", "—"))}</td>'
            f'<td class="trajectory-formula"><code>{_esc(formula)}</code></td>'
            f'<td>{_fmt(item.get("current_best_fitness", item.get("best_fitness")), 5)}</td>'
            f'<td>{_fmt(item.get("best_fitness"), 5)}</td>'
            f'<td>{_fmt(item.get("mean_fitness"), 5)}</td>'
            f'<td>{_fmt(item.get("diversity"), 4)}</td></tr>'
        )
    return (
        '<div class="trajectory-table-wrap"><table class="trajectory-table">'
        '<thead><tr><th>轮次</th><th>当轮最优公式</th><th>当轮最优 fitness</th>'
        '<th>截至当前 best</th><th>群体均值</th><th>多样性</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


# ===========================================================================
# HTML 片段：因子卡片 / 淘汰表格
# ===========================================================================
def _signal_stats_html(stats: dict | None) -> str:
    """把 signal_stats dict 渲染成小标签列表。"""
    if not stats:
        return '<div class="stat-empty">无信号统计</div>'
    chips = []
    for k, v in stats.items():
        chips.append(
            f'<span class="chip"><span class="chip-k">{_esc(k)}</span>'
            f'<span class="chip-v">{_fmt(v)}</span></span>'
        )
    return f'<div class="chips">{"".join(chips)}</div>'


def _factor_card_html(idx: int, factor: dict) -> str:
    """渲染单个因子卡片。"""
    formula = factor.get("formula", "")
    raw_expl = factor.get("explanation", "")
    scores = factor.get("alpha_scores", {}) or {}
    rankic = factor.get("rankic")
    signal_stats = factor.get("signal_stats")

    # m10 样本外验证字段（旧结构无此字段时优雅降级为「—」）
    train_ic = factor.get("train_ic", rankic)
    oos_ic = factor.get("oos_ic")
    has_oos = "oos_ic" in factor and oos_ic is not None
    oos_failed = bool(factor.get("oos_failed"))
    oos_note = factor.get("oos_note") or ""

    # M2④：LLM 不可用时显式标注。占位解释（含“未启用”）→ 标注；
    # logic 维降级中性（logic_degraded 或 logic_source 以 neutral 打头）→ 标注。
    expl_degraded = (not raw_expl) or ("未启用" in str(raw_expl))
    explanation = raw_expl or "（无经济解释）"
    if expl_degraded:
        explanation = f"{explanation}（LLM 不可用/未生效）"

    logic_source = str(factor.get("logic_source") or "")
    logic_degraded = bool(factor.get("logic_degraded")) or logic_source.startswith("neutral")
    interp_note = "（LLM 不可用/未生效）" if logic_degraded else ""

    # 综合分（五维简单平均，仅展示用，不改任何字段）
    vals = [_clip01(scores.get(k)) for k, _ in _ALPHA_DIMS]
    overall = sum(vals) / len(vals) if vals else 0.0

    # metrics 区：样本内 IC + 样本外 IC 并列。样本外失效则红色高亮 + 角标。
    metrics = [
        f'<div class="metric"><div class="metric-v">{_fmt(train_ic)}</div>'
        f'<div class="metric-k">样本内 IC</div></div>'
    ]
    if has_oos:
        oos_color = _PALETTE["danger"] if oos_failed else _PALETTE["accent2"]
        badge = (f'<div class="oos-badge">{_esc(oos_note or "样本外失效")}</div>'
                 if oos_failed else "")
        metrics.append(
            f'<div class="metric"><div class="metric-v" style="color:{oos_color}">'
            f'{_fmt(oos_ic)}</div><div class="metric-k">样本外 IC</div>{badge}</div>'
        )
    else:
        metrics.append(
            '<div class="metric"><div class="metric-v">—</div>'
            '<div class="metric-k">样本外 IC</div></div>'
        )
    metrics_html = "".join(metrics)
    score_items = []
    for key, label_cn in _ALPHA_DIMS:
        score_items.append(
            f'<div class="score-pill"><span>{_esc(label_cn)}</span>'
            f'<strong>{_clip01(scores.get(key)):.2f}</strong></div>'
        )
    score_grid_html = "".join(score_items)

    return f"""
    <div class="factor-card">
      <div class="factor-head">
        <span class="factor-no">#{idx + 1}</span>
        <span class="factor-overall">综合 {overall:.2f}</span>
      </div>
      <div class="factor-body">
        <div class="factor-left">
          <div class="label">因子公式</div>
          <pre class="formula">{_esc(formula)}</pre>
          <div class="label">经济解释</div>
          <p class="explanation">{_esc(explanation)}</p>
          <div class="metrics">
            {metrics_html}
          </div>
          <div class="label">信号统计</div>
          {_signal_stats_html(signal_stats)}
        </div>
        <div class="factor-right">
          <div class="label">AlphaEval 五维{_esc(interp_note)}</div>
          {_radar_svg(scores)}
          <div class="score-grid">{score_grid_html}</div>
        </div>
      </div>
    </div>"""


# 淘汰原因归类：把自由文本原因映射到三大类，用于上色/标签
def _classify_reason(reason: str) -> tuple[str, str]:
    """返回 (类别标签, css 类名)。"""
    r = str(reason or "").lower()
    if any(kw in r for kw in ("量纲", "dimension", "dim", "非法", "invalid", "illegal")):
        return "量纲非法", "tag-dim"
    if any(kw in r for kw in ("相关", "corr", "冗余", "redundan")):
        return "高相关", "tag-corr"
    if any(kw in r for kw in ("ic", "fitness", "适应度", "太低", "low")):
        return "IC 太低", "tag-ic"
    return "其它", "tag-other"


def _rejected_table_html(rejected: list[dict] | None) -> str:
    """渲染底部淘汰记录表格。"""
    if not rejected:
        return '<p class="empty">本轮无淘汰记录。</p>'

    rows = []
    for i, rec in enumerate(rejected):
        formula = rec.get("formula", "")
        reason = rec.get("reason", "") or rec.get("explanation", "")
        cat, css = _classify_reason(reason)
        # 额外可选字段：rankic / correlation，有就展示
        extra_bits = []
        for key, label in (("rankic", "rankIC"), ("correlation", "相关"),
                           ("max_corr", "最大相关"), ("layer", "层")):
            if key in rec and rec[key] is not None:
                extra_bits.append(f"{label}={_fmt(rec[key])}")
        extra = "；".join(extra_bits)
        rows.append(f"""
        <tr>
          <td class="col-no">{i + 1}</td>
          <td class="col-formula"><code>{_esc(formula)}</code></td>
          <td class="col-cat"><span class="tag {css}">{cat}</span></td>
          <td class="col-reason">{_esc(reason)}{(' <span class="muted">(' + _esc(extra) + ')</span>') if extra else ''}</td>
        </tr>""")

    return f"""
    <table class="reject-table">
      <thead>
        <tr><th>#</th><th>被淘汰因子公式</th><th>淘汰类别</th><th>淘汰原因</th></tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ===========================================================================
# CSS（全部内联）
# ===========================================================================
def _css() -> str:
    p = _PALETTE
    return f"""
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0;
      background: {p['bg']}; color: {p['text']};
      font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
      line-height: 1.6;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }}
    header.top {{
      border-bottom: 1px solid {p['border']};
      padding-bottom: 20px; margin-bottom: 28px;
    }}
    header.top h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0.5px; }}
    header.top .sub {{ color: {p['muted']}; font-size: 13px; }}
    section {{ margin-bottom: 40px; }}
    section > h2 {{
      font-size: 18px; margin: 0 0 16px;
      padding-left: 10px; border-left: 3px solid {p['accent']};
    }}
    .label {{ color: {p['muted']}; font-size: 12px; margin: 12px 0 4px;
             text-transform: uppercase; letter-spacing: 0.6px; }}
    .factor-card {{
      background: {p['card']}; border: 1px solid {p['border']};
      border-radius: 12px; padding: 18px 20px; margin-bottom: 18px;
    }}
    .factor-head {{ display: flex; justify-content: space-between; align-items: center; }}
    .factor-no {{ font-size: 16px; font-weight: 700; color: {p['accent']}; }}
    .factor-overall {{ font-size: 13px; color: {p['muted']};
                      background: {p['card_alt']}; padding: 3px 10px; border-radius: 999px; }}
    .factor-body {{ display: flex; gap: 24px; margin-top: 8px; flex-wrap: wrap; }}
    .factor-left {{ flex: 1 1 340px; min-width: 300px; }}
    .factor-right {{ flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; }}
    pre.formula {{
      background: {p['card_alt']}; border: 1px solid {p['border']};
      border-radius: 8px; padding: 12px 14px; margin: 0;
      font-family: "Cascadia Code", "Consolas", monospace; font-size: 13px;
      color: {p['accent2']}; white-space: pre-wrap; word-break: break-word;
    }}
    p.explanation {{ margin: 0; font-size: 14px; color: {p['text']}; }}
    .metrics {{ display: flex; gap: 14px; margin-top: 12px; }}
    .metric {{ background: {p['card_alt']}; border: 1px solid {p['border']};
              border-radius: 8px; padding: 8px 16px; text-align: center; }}
    .metric-v {{ font-size: 20px; font-weight: 700; color: {p['accent']}; }}
    .metric-k {{ font-size: 11px; color: {p['muted']}; }}
    .radar-svg {{ display: block; max-width: 280px; margin: 0 auto; overflow: visible; }}
    .score-grid {{
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px; margin-top: 10px;
    }}
    .score-pill {{
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px; min-width: 0; background: {p['card_alt']};
      border: 1px solid {p['border']}; border-radius: 6px;
      padding: 5px 8px; font-size: 12px;
    }}
    .score-pill span {{ color: {p['muted']}; white-space: nowrap; }}
    .score-pill strong {{ color: {p['text']}; font-weight: 700; }}
    .oos-badge {{ margin-top: 4px; font-size: 10px; font-weight: 600;
                 color: {p['danger']}; background: rgba(255,107,125,0.15);
                 border-radius: 4px; padding: 1px 6px; white-space: nowrap; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{ background: {p['card_alt']}; border: 1px solid {p['border']};
            border-radius: 6px; padding: 3px 8px; font-size: 12px; }}
    .chip-k {{ color: {p['muted']}; margin-right: 6px; }}
    .chip-v {{ color: {p['text']}; font-weight: 600; }}
    .stat-empty, .empty {{ color: {p['muted']}; font-size: 13px; font-style: italic; }}
    .chart-box {{
      background: {p['card']}; border: 1px solid {p['border']};
      border-radius: 12px; padding: 18px 20px;
    }}
    .trajectory-svg {{ display: block; width: 100%; height: auto; min-height: 300px; }}
    .trajectory-table-wrap {{ margin-top: 16px; overflow-x: auto; }}
    table.trajectory-table {{
      width: 100%; min-width: 760px; border-collapse: collapse; font-size: 12px;
      background: {p['card']}; border: 1px solid {p['border']};
    }}
    .trajectory-table th {{
      text-align: left; padding: 9px 10px; color: {p['muted']};
      background: {p['card_alt']}; border-bottom: 1px solid {p['border']};
      font-weight: 600; white-space: nowrap;
    }}
    .trajectory-table td {{
      padding: 8px 10px; border-bottom: 1px solid {p['border']};
      vertical-align: top; white-space: nowrap;
    }}
    .trajectory-table tr:nth-child(even) td {{ background: {p['card_alt']}; }}
    .trajectory-table tr:last-child td {{ border-bottom: none; }}
    .trajectory-formula {{ min-width: 360px; white-space: normal !important; }}
    .trajectory-formula code {{ color: {p['accent2']}; word-break: break-word; }}
    table.reject-table {{
      width: 100%; border-collapse: collapse; font-size: 13px;
      background: {p['card']}; border: 1px solid {p['border']}; border-radius: 12px;
      overflow: hidden;
    }}
    .reject-table th {{
      text-align: left; padding: 10px 14px; color: {p['muted']};
      background: {p['card_alt']}; border-bottom: 1px solid {p['border']};
      font-weight: 600;
    }}
    .reject-table td {{ padding: 10px 14px; border-bottom: 1px solid {p['border']};
                       vertical-align: top; }}
    .reject-table tr:nth-child(even) td {{ background: {p['card_alt']}; }}
    .reject-table tr:last-child td {{ border-bottom: none; }}
    .col-no {{ width: 36px; color: {p['muted']}; }}
    .col-formula code {{ font-family: "Cascadia Code", "Consolas", monospace;
                        color: {p['accent2']}; word-break: break-word; }}
    .tag {{ display: inline-block; padding: 2px 9px; border-radius: 999px;
           font-size: 12px; font-weight: 600; white-space: nowrap; }}
    .tag-dim {{ background: rgba(255,180,84,0.15); color: {p['accent3']}; }}
    .tag-corr {{ background: rgba(255,107,125,0.15); color: {p['danger']}; }}
    .tag-ic {{ background: rgba(91,157,255,0.15); color: {p['accent']}; }}
    .tag-other {{ background: {p['card_alt']}; color: {p['muted']}; }}
    .muted {{ color: {p['muted']}; }}
    footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid {p['border']};
             color: {p['muted']}; font-size: 12px; text-align: center; }}
    """


# ===========================================================================
# LLM 经济解释润色（可选；照 llm_policy，只产文字绝不改数值）
# ===========================================================================
_LLM_TOOL = {
    "name": "emit_explanations",
    "description": "为每个因子给出简短的经济解释（自然语言），不涉及任何数值。",
    "input_schema": {
        "type": "object",
        "properties": {
            "explanations": {
                "type": "array",
                "description": "与输入因子顺序一一对应的经济解释",
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {"type": "integer", "description": "因子序号，从 0 开始"},
                        "explanation": {"type": "string", "description": "≤80 字中文经济解释"},
                    },
                    "required": ["idx", "explanation"],
                },
            },
        },
        "required": ["explanations"],
    },
}

_LLM_SYSTEM = (
    "你是量化因子经济含义解读助手。给定若干因子的公式，"
    "为每个因子写一句简短（≤80 字）的中文经济解释，说明它可能捕捉的市场行为/风险溢价。"
    "严格约束：只输出自然语言解释，绝不输出、评价或修改任何数值指标（如 rankIC、分数）。"
    "必须调用 emit_explanations 工具返回。"
)


def _llm_enabled(cfg: dict) -> bool:
    raise RuntimeError("report_builder is render-only; LLM calls are forbidden here")


def _build_llm_client(cfg: dict):
    """构造 anthropic 客户端；SDK 缺失/无 KEY → 返回 (None, model)。失败绝不抛。"""
    raise RuntimeError("report_builder is render-only; LLM clients are forbidden here")
    # 模型解析统一走 llm_explainer._resolve_model（裸别名 opus→具体名 + 运行时读环境），
    # 绝不兜底裸别名 "opus"（网关不认，会 model_not_found）。
    model = "render-only"
    if False:
        return None, model
    try:
        client = None
    except Exception:  # noqa: BLE001 —— 构造失败也降级
        return None, model
    return client, model


def _llm_explain(factors: list[dict], config: dict | None) -> dict[int, str]:
    raise RuntimeError("report_builder is render-only; use llm_explainer before rendering")

    """用 LLM 为因子润色经济解释，返回 {idx: explanation}。

    失败/关闭 → 返回空 dict（pipeline 继续，卡片用原 explanation）。
    LLM 只产文字，绝不改任何数值字段。
    """
    cfg = (config or {}).get("llm", {}) if config else {}
    if not _llm_enabled(cfg):
        return {}

    client, model = _build_llm_client(cfg)
    if client is None:
        return {}

    # 只把公式喂给 LLM（脱敏；不给数值指标）
    payload = [{"idx": i, "formula": f.get("formula", "")}
               for i, f in enumerate(factors)]
    user = ("以下是本轮挖掘出的因子公式，请为每个因子写一句经济解释：\n"
            f"{json.dumps(payload, ensure_ascii=False)}")

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=_LLM_SYSTEM,
            tools=[_LLM_TOOL],
            tool_choice={"type": "tool", "name": "emit_explanations"},
            messages=[{"role": "user", "content": user}],
            timeout=60,
        )
    except Exception:  # noqa: BLE001 —— 任何异常立刻降级不重试
        return {}

    # 提取 tool_use；非 tool_use 回复降级 None（空 dict）
    tool_use = next(
        (b for b in getattr(resp, "content", [])
         if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_use is None:
        return {}

    data = tool_use.input or {}
    out: dict[int, str] = {}
    for item in data.get("explanations", []) or []:
        try:
            idx = int(item.get("idx"))
        except (TypeError, ValueError):
            continue
        expl = item.get("explanation")
        if isinstance(expl, str) and expl.strip():
            out[idx] = expl.strip()
    return out


# ===========================================================================
# 主入口
# ===========================================================================
def build_report(factors: list[dict],
                 trajectory: list[dict],
                 rejected: list[dict] | None = None,
                 output_path: str | None = None,
                 config: dict | None = None) -> str:
    """生成自包含 HTML 可视化报告。

    参数：
        factors:    因子列表，每个含
                    {formula, explanation, alpha_scores(五维dict), rankic, signal_stats}
        trajectory: GP 每代轨迹，每个含 {gen, best_fitness, mean_fitness, diversity}
        rejected:   淘汰记录列表，每个含 {formula, reason, ...}（可选）
        output_path: 非空则把 HTML 写到该路径（UTF-8）
        config:     可选配置，config.llm.{enabled,model,base_url,api_key} 控制 LLM 润色

    返回：
        完整 HTML 字符串（单文件，所有 CSS/JS/SVG 内联，双击可看）。
    """
    factors = list(factors or [])
    trajectory = list(trajectory or [])
    rejected = list(rejected or [])

    # 各部分 HTML
    cards_html = "".join(_factor_card_html(i, f) for i, f in enumerate(factors)) \
        if factors else '<p class="empty">本轮未产出有效因子。</p>'
    lines_html = _lines_svg(trajectory)
    trajectory_table_html = _trajectory_table_html(trajectory)
    reject_html = _rejected_table_html(rejected)

    n_factor = len(factors)
    n_gen = len(trajectory)
    n_reject = len(rejected)
    # 样本外失效计数（m10）：有 oos 字段且失效的因子数
    n_oos_failed = sum(1 for f in factors if f.get("oos_failed"))
    oos_summary = (f" · <span style=\"color:{_PALETTE['danger']}\">样本外失效 "
                   f"{n_oos_failed} 个</span>") if n_oos_failed else ""

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Alpha 因子挖掘报告</title>
<style>{_css()}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>Alpha 因子挖掘报告</h1>
    <div class="sub">共 {n_factor} 个入选因子 · {n_gen} 个迭代轨迹点 · {n_reject} 条淘汰记录{oos_summary}</div>
  </header>

  <section id="factors">
    <h2>① 入选因子</h2>
    {cards_html}
  </section>

  <section id="trajectory">
    <h2>② GP 迭代追溯</h2>
    <div class="chart-box">
      {lines_html}
      {trajectory_table_html}
    </div>
  </section>

  <section id="rejected">
    <h2>③ 淘汰记录</h2>
    {reject_html}
  </section>

  <footer>由 skill-llm-alpha-generator / report_builder.py 生成 · 单文件离线报告</footer>
</div>
</body>
</html>"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)

    return html_doc


# 命令行自测：造点假数据写出 demo 报告
if __name__ == "__main__":  # pragma: no cover
    demo_factors = [
        {
            "formula": "ts_rank(divide(sub(close, open), high), 10)",
            "explanation": "捕捉日内动量相对振幅的排序信号。",
            "alpha_scores": {"effectiveness": 0.72, "robustness": 0.6,
                             "interpretability": 0.8, "diversity": 0.55, "parsimony": 0.7},
            "rankic": 0.043,
            "signal_stats": {"mean": 0.01, "std": 0.98, "turnover": 0.32, "coverage": 0.95},
        },
    ]
    demo_traj = [
        {"gen": g, "best_fitness": 0.02 + g * 0.004,
         "mean_fitness": 0.005 + g * 0.002, "diversity": 0.9 - g * 0.03}
        for g in range(15)
    ]
    demo_rej = [
        {"formula": "add(close, volume)", "reason": "量纲非法：price 与 volume 不可相加"},
        {"formula": "rank(close)", "reason": "rankIC 太低 (0.001)", "rankic": 0.001},
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_report.html")
    build_report(demo_factors, demo_traj, demo_rej, output_path=out)
    print(f"demo 报告已写出：{out}")

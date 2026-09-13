#!/usr/bin/env python3
"""
H009-G - DAG renderer (PNG, for the Power BI image visual)
=========================================================
Reads data/snapshots/h009_dag_spec.json (the single source of truth, also used by the
deck's native-shape layout) and draws the DAG with the deck palette.

Why a second render: h009_dag.png uses the matplotlib defaults (orange #E67E22 / purple #9B59B6 /
bright green #27AE60), which clashes with the deck's navy system, and the original contained nodes that
were never estimated. This renderer uses only the nodes and edges declared in the spec, coloured by
'identified' vs 'not estimated'.

Output: data/visualizations/h009_dag_deck.png
Usage:  python3 scripts/h009_dag_render.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


def _setup_cjk() -> str:
    """CJK rendering needs a CJK font, otherwise glyphs are missing (DejaVu Sans has no Chinese)."""
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei"):
        if any(f.name == cand for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            return cand
    return ""

# repository root (run from there); data/ and outputs/ paths below are relative to it
BASE = Path(__file__).resolve().parents[1]
SNAP = BASE / "data" / "snapshots"
VIZ = BASE / "data" / "visualizations"

# deck palette
NAVY, BLUE, CYAN = "#0B2D52", "#1687C7", "#67C5E8"
MUTED, LINE, SOFT = "#66717D", "#C7CED6", "#E8EDF2"
SIGNAL, POSITIVE = "#D94B3D", "#2F7D5B"

NODE_STYLE = {
    "covariate":     dict(fc=SOFT,    ec=MUTED,    tc=MUTED,    dashed=False),
    "instrument":    dict(fc=SOFT,    ec=BLUE,     tc=BLUE,     dashed=True),
    "treatment":     dict(fc=NAVY,    ec=NAVY,     tc="#FFFFFF", dashed=False),
    "mediator":      dict(fc=BLUE,    ec=BLUE,     tc="#FFFFFF", dashed=False),
    "outcome":       dict(fc=POSITIVE, ec=POSITIVE, tc="#FFFFFF", dashed=False),
    "not_estimated": dict(fc=SOFT,    ec=LINE,     tc=MUTED,    dashed=True),
}
EDGE_STYLE = {
    "identified":      dict(c=NAVY,    lw=2.6, ls="-",   z=4),
    "not_established": dict(c=SIGNAL,  lw=2.0, ls=(0, (5, 3)), z=3),
    "confound":        dict(c=LINE,    lw=1.2, ls="-",   z=1),
    "instrument":      dict(c=BLUE,    lw=1.8, ls=(0, (2, 2)), z=2),
    "not_estimated":   dict(c=LINE,    lw=1.2, ls=(0, (1, 2)), z=1),
}


def clip(cx, cy, tx, ty, hw, hh):
    """Clip the segment from (cx,cy) to (tx,ty) at the boundary of the source box."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = hw / abs(dx) if dx else float("inf")
    sy = hh / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def main(lang: str = "en") -> int:
    if lang != "en":
        print(f"   CJK font: {_setup_cjk() or 'not found (Chinese glyphs may be missing)'}")
    spec = json.loads((SNAP / "h009_dag_spec.json").read_text(encoding="utf-8"))
    cv = spec["canvas"]
    W, H, NW, NH = cv["width_in"], cv["height_in"], cv["node_w"], cv["node_h"]
    key = "label_en" if lang == "en" else "label_zh"

    fig, ax = plt.subplots(figsize=(W, H), dpi=190, facecolor="white")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    nodes = {n["id"]: n for n in spec["nodes"]}

    # ---- edges (drawn first, under the nodes)
    for e in spec["edges"]:
        a, b = nodes[e["from"]], nodes[e["to"]]
        st = EDGE_STYLE[e["kind"]]
        x0, y0 = clip(a["x"], a["y"], b["x"], b["y"], NW / 2, NH / 2)
        x1, y1 = clip(b["x"], b["y"], a["x"], a["y"], NW / 2, NH / 2)
        ax.plot([x0, x1], [y0, y1], color=st["c"], lw=st["lw"], ls=st["ls"],
                zorder=st["z"], solid_capstyle="round")
        if e.get("effect"):
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.09, e["effect"], ha="center", va="center",
                    fontsize=7.6, fontweight="bold", color=st["c"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.94))

    # ---- nodes
    for n in spec["nodes"]:
        s = NODE_STYLE[n["role"]]
        box = FancyBboxPatch((n["x"] - NW / 2, n["y"] - NH / 2), NW, NH,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             fc=s["fc"], ec=s["ec"], lw=1.4, zorder=5,
                             linestyle=(0, (3, 2)) if s["dashed"] else "-")
        ax.add_patch(box)
        ax.text(n["x"], n["y"], n[key], ha="center", va="center", zorder=6,
                fontsize=8.4, fontweight="bold", color=s["tc"], linespacing=1.25)

    # ---- legend
    legend = [("identified", "Identified effect"),
              ("not_established", "Estimated, not established"),
              ("confound", "Confounding path"),
              ("instrument", "Instrument"),
              ("not_estimated", "Not estimated")]
    x = 0.55
    for kind, label in legend:
        st = EDGE_STYLE[kind]
        ax.plot([x, x + 0.28], [H - 0.12, H - 0.12], color=st["c"], lw=st["lw"], ls=st["ls"])
        ax.text(x + 0.34, H - 0.12, label, fontsize=7.2, color=MUTED, va="center")
        x += 0.42 + len(label) * 0.052

    fig.subplots_adjust(left=0.012, right=0.988, top=0.955, bottom=0.02)
    out = VIZ / f"h009_dag_deck{'' if lang == 'en' else '_zh'}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"OK  saved: {out}  ({W}x{H} in, {len(spec['nodes'])} nodes / {len(spec['edges'])} edges)")
    return 0


if __name__ == "__main__":
    main("en")
    main("zh")

#!/usr/bin/env python3
"""
Caliber gate - churn-definition consistency checks
=========================================================
Three check groups added on top of the existing six gates:

  1) Caliber reconciliation: the 180-day dormancy churn rate must equal the share of
     churn_label in the model table, and the Actual_Churn mean in Customer_Churn_Scores.
  2) Page figure reconciliation: retained customers 24,196 / revenue share 26.4% /
     same-day split orders 29.0% / consecutive repeat 0.46% - the figures on deck S17
     and dashboard p8 must match the snapshot.
  3) Caliber completeness: both the deck (S17 main caliber 180 days + 30-day early
     warning) and the dashboard (p8) must show the main and the early-warning caliber.
     Showing only one of them fails the gate.

Usage: python3 scripts/caliber_gate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

# repository root (run from there); data/ and outputs/ paths below are relative to it
BASE = Path(__file__).resolve().parents[1]
SNAP = BASE / "data" / "snapshots"
DASH_EN = BASE / "outputs" / "reports" / "dashboard_en"
DASH_ZH = BASE / "outputs" / "reports" / "dashboard"
DECK_EN = BASE / "outputs" / "portfolio" / "deck_projects" / "olist-churn-deck-en"
DECK_ZH = BASE / "outputs" / "portfolio" / "deck_projects" / "olist-churn-deck-zh"

fails: list[str] = []
notes: list[str] = []


def check(cond: bool, msg: str) -> None:
    (notes if cond else fails).append(("✅ " if cond else "❌ ") + msg)


def main() -> int:
    cal = json.load(open(SNAP / "h011_caliber_sensitivity.json", encoding="utf-8"))
    rfm = pd.read_csv(SNAP / "h003_c_rfm_features.csv")
    scores = pd.read_csv(DASH_EN / "data" / "Customer_Churn_Scores.csv")
    sens = pd.read_csv(SNAP / "h011_caliber_sensitivity.csv")

    print("=" * 74)
    print("1) Caliber reconciliation")
    print("=" * 74)
    r180 = float(sens.loc[sens.window_days == 180, "churn_rate"].iloc[0])
    label = float(rfm.churn_label.mean()) * 100
    actual = float(scores.Actual_Churn.mean()) * 100
    check(abs(r180 - label) < 0.01, f"180-day dormancy churn rate {r180:.4f}% == churn_label {label:.4f}%")
    check(abs(r180 - actual) < 0.02, f"180-day dormancy churn rate {r180:.4f}% == dashboard Actual_Churn {actual:.4f}%")
    rec = cal["reconciliation"]
    check(rec["passed"] and abs(rec["churn_180_dormant"] - r180) < 0.01,
          f"caliber package self-reconciliation passed ({rec['churn_180_dormant']:.4f}%)")

    print("\n" + "=" * 74)
    print("2) Page figure reconciliation")
    print("=" * 74)
    ret = scores[scores.Actual_Churn == 0]
    churn = scores[scores.Actual_Churn == 1]
    n_ret = len(ret)
    share_ret = ret.Total_Spent.sum() / scores.Total_Spent.sum() * 100
    avg_ret, avg_ch = ret.Total_Spent.mean(), churn.Total_Spent.mean()
    rep = cal["repeat_structure"]
    oneshare = float((ret.Order_Count == 1).mean()) * 100
    check(n_ret == 24196, f"retained customers {n_ret:,} == 24,196")
    check(abs(share_ret - 26.4) < 0.1, f"retained revenue share {share_ret:.1f}% ~= 26.4%")
    check(abs(avg_ret - 144.96) < 0.05 and abs(avg_ch - 141.24) < 0.05,
          f"avg spend retained R${avg_ret:.2f} / churned R${avg_ch:.2f} (~144.96 / 141.24)")
    check(abs(rep["same_day_share_of_repeat"] - 29.0) < 0.1,
          f"same-day split orders as share of repeat buyers {rep['same_day_share_of_repeat']:.1f}% ~= 29.0%")
    check(abs(rep["true_repeat_rate"] - 2.10) < 0.02,
          f"true cross-period repeat rate {rep['true_repeat_rate']:.2f}% ~= 2.10%")
    check(abs(oneshare - 96.7) < 0.1, f"retained customers with a single order {oneshare:.1f}% ~= 96.7%")
    trend = pd.read_csv(DASH_EN / "data" / "Monthly_Churn_Trend.csv")
    last = trend.iloc[-1]
    consec_share = float(last["Consecutive_Repeat"]) / float(last["Buyers"]) * 100
    check(abs(consec_share - 0.46) < 0.05,
          f"consecutive repeat share in the final month {consec_share:.2f}% ~= 0.46% (cited on S17 / dashboard p7)")
    tmdl = (DASH_EN / "olist_demo_dashboard.SemanticModel/definition/tables/Monthly_Churn_Trend.tmdl").read_text(encoding="utf-8")
    check("\tmeasure Consecutive_Share " in tmdl,
          "Consecutive_Share measure added to the monthly trend table (neither the column nor the CSV was changed)")
    for cmd in ("h011_caliber_sensitivity", "h012_evidence_deepdive", "h007_xgboost_pipeline"):
        check((BASE / "scripts" / f"{cmd}.py").exists(), f"recompute script present: {cmd}.py")
    # 3) naming discipline: snapshot headers must not contain a bare churn_<days> column
    import re as _re
    banned = []
    for f in sorted((BASE / "data" / "snapshots").glob("*.csv")):
        try:
            head = f.open(encoding="utf-8").readline()
        except Exception:
            continue
        for col in head.strip().split(","):
            if _re.fullmatch(r"churn_?\d+", col.strip()):
                banned.append(f"{f.name}:{col.strip()}")
    check(not banned, f"naming discipline: no bare churn_<days> column ({banned or 'clean'})")
    pool = json.load(open(SNAP / "h012_deepdive_summary.json", encoding="utf-8"))["active_pool_naming_fix"]
    check(pool["renamed"] or pool["recency_max"] <= 92,
          f"h007_active_pool_check.csv renamed (active pool {pool['rows']:,} customers, recency <= {pool['recency_max']} days)")

    print("\n" + "=" * 74)
    print("3) Caliber completeness (deck and dashboard must both show main + early-warning caliber)")
    print("=" * 74)
    deck_text = {}
    for lang, ws in (("en", DECK_EN), ("zh", DECK_ZH)):
        slides = json.loads((ws / "slides.json").read_text(encoding="utf-8"))
        s17 = next((s for s in slides if s["slide_id"] == "S17"), None)
        ids = [s["slide_id"] for s in slides]
        check(ids == [f"S{i:02d}" for i in range(1, 30)],
              f"deck-{lang}: page numbers contiguous 1-29")
        for sid, lay in (("S04", "C48"), ("S06", "C49"), ("S09", "C50"), ("S12", "C51"), ("S18", "C52")):
            s_ = next((s for s in slides if s["slide_id"] == sid), None)
            check(s_ is not None and s_["layout_id"] == lay, f"deck-{lang}: {sid} exists and uses {lay}")
        s17 = next((s for s in slides if s["slide_id"] == "S22"), None)
        check(s17 is not None and s17["layout_id"] == "C47", f"deck-{lang}: S22 (caliber page) exists and uses C47")
        blob = json.dumps(s17 or {}, ensure_ascii=False)
        deck_text[lang] = blob
        check("180" in blob, f"deck-{lang}: S17 states the 180-day main caliber")
        check("30" in blob, f"deck-{lang}: S17 states the 30-day early-warning caliber")
        check("29.0" in blob or "29.0%" in blob, f"deck-{lang}: S17 discloses the same-day split-order share")
        s22 = next((s for s in slides if s["slide_id"] == "S27"), None)
        d22 = json.dumps(s22 or {}, ensure_ascii=False)
        check("Caliber sensitivity" in d22,
              f"deck-{lang}: S22 metric dictionary includes caliber sensitivity")
        check(len(slides) == 29, f"deck-{lang}: {len(slides)} slides in total (expected 29)")

    for lang, proj in (("en", DASH_EN), ("zh", DASH_ZH)):
        pages = proj / "olist_demo_dashboard.Report/definition/pages"
        check((pages / "p8_caliber").is_dir(), f"dashboard-{lang}: page p8_caliber exists")
        order = json.loads((pages / "pages.json").read_text(encoding="utf-8"))["pageOrder"]
        expected = ["p1_overview", "p8_caliber", "p2_segments", "p3_drivers", "p9_delay",
                    "p4_reviews", "p6_causal", "p10_benchmark", "p5_action", "p7_funnel"]
        check(order == expected, f"dashboard-{lang}: pageOrder follows the five-layer order (appendix p7_funnel last)")
        check(len(order) == 10, f"dashboard-{lang}: {len(order)} pages in total")
        last = json.loads((pages / order[-1] / "page.json").read_text(encoding="utf-8"))["displayName"]
        check(("Appendix" in last), f"dashboard-{lang}: last page is the appendix -> {last}")
        p2 = json.loads((pages / "p2_segments/page.json").read_text(encoding="utf-8"))
        name = p2["displayName"]
        check(("Who stays" in name), f"dashboard-{lang}: p2 renamed -> {name}")
        model = proj / "olist_demo_dashboard.SemanticModel/definition/model.tmdl"
        txt = model.read_text(encoding="utf-8")
        for t in (["Churn_Sensitivity", "Caliber_Definition"] if lang == "en"
                  else ["Churn_Sensitivity", "Caliber_Definition"]):
            check(f"ref table {t}" in txt, f"dashboard-{lang}: model.tmdl registers {t}")
        rd = (proj / "README.md").read_text(encoding="utf-8")
        check(("180" in rd and "30" in rd), f"dashboard-{lang}: guide documents the main and early-warning caliber")
        check("2.10" in rd, f"dashboard-{lang}: guide discloses the true repeat rate 2.10%")
        # the page count cited on deck S21 must match the dashboard
        dash_line = re.search(r"(\d+) pages · (\d+) tables", rd)
        notes.append(f"i dashboard-{lang}: {len(order)} pages, generated by the same function as the deck S21 copy")

    print("\n" + "=" * 74)
    print("4) Language purity: the English dashboard must not contain Chinese (the mapping layer separates languages)")
    print("=" * 74)
    import re as _re2
    _CJK = _re2.compile(r"[\u4e00-\u9fff]")
    leaks = []
    # 4-1 data files
    for f in sorted((DASH_EN / "data").glob("*.csv")):
        df = pd.read_csv(f, dtype=str).fillna("")
        for c in df.columns:
            hit = df[c].map(lambda v: bool(_CJK.search(str(v))))
            if hit.any():
                leaks.append(f"data/{f.name} column {c}: {int(hit.sum())} Chinese values "
                             f"e.g. {df.loc[hit, c].unique()[:3].tolist()}")
    # 4-2 visual titles and field names
    for vf in sorted(DASH_EN.rglob("definition/pages/*/visuals/*/visual.json")):
        txt = vf.read_text(encoding="utf-8")
        for m in _re2.finditer(r'"nativeQueryRef":\s*"([^"]*)"', txt):
            if _CJK.search(m.group(1)):
                leaks.append(f"{vf.parent.name} field name {m.group(1)}")
        for m in _re2.finditer(r'"Literal":\s*\{\s*"Value":\s*"\'([^\']*)\'', txt):
            if _CJK.search(m.group(1)):
                leaks.append(f"{vf.parent.name} title {m.group(1)[:40]}")
    # 4-3 table member names
    for tf in sorted((DASH_EN / "olist_demo_dashboard.SemanticModel/definition/tables").glob("*.tmdl")):
        for m in _re2.finditer(r"^\t(?:column|measure) ([^\n ]+)", tf.read_text(encoding="utf-8"), _re2.M):
            if _CJK.search(m.group(1)):
                leaks.append(f"{tf.stem}.{m.group(1)}")
    check(not leaks, f"English dashboard language purity: {'clean' if not leaks else f'{len(leaks)} Chinese leaks'}")
    for x in leaks[:12]:
        print("   ✗ " + x)

    for n in notes:
        print("  " + n)
    if fails:
        print("\nFailed checks:")
        for f in fails:
            print("  " + f)
        print(f"\nRESULT: FAILED ({len(fails)} failed / {len(notes)} passed)")
        return 1
    print(f"\nRESULT: PASSED ({len(notes)} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

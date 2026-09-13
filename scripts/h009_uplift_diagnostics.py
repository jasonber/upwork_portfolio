#!/usr/bin/env python3
"""
H009-E/F - uplift diagnostics (Qini / AUUC / deciles / event counts)
=========================================================
Why this script exists
------------------
h009_uplift_cate.json kept only the two AUUC and Qini scalars - no curve, no decile table and no event
counts. As a result neither the deck nor the dashboard could explain why Qini is negative, leaving only
two options: omit it (hiding a core output) or show it and be unable to answer questions about it.

This script recomputes every diagnostic from h009_cate_by_segment.csv (the T-Learner 30% hold-out),
using exactly the same caliber as compute_auuc() in h009_causal_pipeline.py, and reconciles AUUC against
the value already stored in h009_uplift_cate.json (it raises if they disagree).

Output
----
data/snapshots/h009_uplift_qini_curve.csv      Qini / AUUC curve points (for the deck and dashboard line charts)
data/snapshots/h009_uplift_deciles.csv         decile table (predicted uplift vs actual repurchase)
data/snapshots/h009_uplift_diagnostics.json    event counts / class balance / Qini / AUUC / verdict
data/visualizations/h009_qini_curve.png        curve plot (for the Power BI image visual)

Usage:  python3 scripts/h009_uplift_diagnostics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# repository root (run from there); data/ and outputs/ paths below are relative to it
BASE = Path(__file__).resolve().parents[1]
SNAP = BASE / "data" / "snapshots"
VIZ = BASE / "data" / "visualizations"

# diagnostic thresholds: below these, the ranking is declared unusable for targeting
AUUC_TARGET = 0.05          # same caliber as the print in h009_causal_pipeline.py
QINI_END_TARGET = 0.0       # the curve must end above the baseline (0)


def qini_curve(uplift: np.ndarray, treat: np.ndarray, outcome: np.ndarray):
    """Cumulative incremental-gain curve, ordered by uplift descending.

    Same caliber as compute_auuc() in h009_causal_pipeline.py:
      cum_treat(x) = events in the truncated treatment group / total treatment
      cum_ctrl(x)  = events in the truncated control group / total control
      qini(x)      = cum_treat(x) - cum_ctrl(x)
    """
    n = len(uplift)
    order = np.argsort(-uplift)
    t, y = treat[order], outcome[order]
    n_t, n_c = max(int(t.sum()), 1), max(int((1 - t).sum()), 1)
    cum_treat = np.cumsum(t * y) / n_t
    cum_ctrl = np.cumsum((1 - t) * y) / n_c
    qini = cum_treat - cum_ctrl
    pop = np.arange(1, n + 1) / n
    return pop, qini, cum_treat, cum_ctrl


def main() -> int:
    df = pd.read_csv(SNAP / "h009_cate_by_segment.csv")
    T = df["T"].values.astype(int)
    Y = df["Y"].values.astype(int)
    U = df["uplift"].values.astype(float)

    n1, n0 = int(T.sum()), int((1 - T).sum())
    e1, e0 = int(Y[T == 1].sum()), int(Y[T == 0].sum())
    r1, r0 = e1 / n1, e0 / n0

    pop, qini, cum_treat, cum_ctrl = qini_curve(U, T, Y)
    auuc = float(np.trapezoid(qini, pop))
    qini_end = float(qini[-1])

    # ---- reconcile against the existing artefact (calibers must match)
    with open(SNAP / "h009_uplift_cate.json", encoding="utf-8") as f:
        baseline_auuc = float(json.load(f)["auuc"])
    delta = abs(auuc - baseline_auuc)
    if delta > 1e-9:
        raise SystemExit(f"AUUC reconciliation failed: recomputed {auuc:.9f} vs stored {baseline_auuc:.9f} (diff {delta:.2e})")

    # ---- decile table
    dec = pd.qcut(U, 10, labels=False)
    rows = []
    for k in range(10):
        m = dec == k
        m1, m0 = m & (T == 1), m & (T == 0)
        rows.append({
            "decile": k + 1,
            "pred_uplift_mean": round(float(U[m].mean()), 4),
            "n_total": int(m.sum()),
            "n_treated": int(m1.sum()),
            "events_treated": int(Y[m1].sum()),
            "repeat_treated_pct": round(float(Y[m1].mean() * 100) if m1.sum() else np.nan, 2),
            "n_control": int(m0.sum()),
            "events_control": int(Y[m0].sum()),
            "repeat_control_pct": round(float(Y[m0].mean() * 100) if m0.sum() else np.nan, 2),
        })
    d10 = pd.DataFrame(rows)
    d10.to_csv(SNAP / "h009_uplift_deciles.csv", index=False, encoding="utf-8")

    # ---- curve points (1% resolution, 101 points; the deck line chart reads every 10%)
    step = max(1, len(pop) // 100)
    idx = np.unique(np.append(np.arange(0, len(pop), step), len(pop) - 1))
    pd.DataFrame({
        "cum_pop_pct": np.round(pop[idx] * 100, 3),
        "qini": np.round(qini[idx], 6),
        "cum_y_treated": np.round(cum_treat[idx], 6),
        "cum_y_control": np.round(cum_ctrl[idx], 6),
        "uplift_threshold": np.round(np.sort(U)[::-1][idx], 4),
    }).to_csv(SNAP / "h009_uplift_qini_curve.csv", index=False, encoding="utf-8")

    # ---- monotonicity check (higher uplift deciles should perform better)
    hi = d10[d10.decile >= 8]
    lo = d10[d10.decile <= 3]
    hi_events, hi_treated = int(hi.events_treated.sum()), int(hi.n_treated.sum())
    lo_events, lo_treated = int(lo.events_treated.sum()), int(lo.n_treated.sum())
    hi_rate = hi_events / hi_treated * 100 if hi_treated else float("nan")
    lo_rate = lo_events / lo_treated * 100 if lo_treated else float("nan")

    verdict_reasons = []
    if auuc <= 0:
        verdict_reasons.append(f"AUUC {auuc:+.4f} is below the 0 baseline")
    if qini_end < QINI_END_TARGET:
        verdict_reasons.append(f"the Qini curve ends negative at {qini_end:+.4f}")
    if hi_treated == 0 or hi_events == 0:
        verdict_reasons.append(f"the top deciles (D8-D10) hold only {hi_treated} treated customers and {hi_events} events - unverifiable")
    if hi_rate < lo_rate:
        verdict_reasons.append(f"actual repurchase in the top deciles {hi_rate:.2f}% is below the bottom deciles {lo_rate:.2f}% (rank reversed)")
    if e1 < 30:
        verdict_reasons.append(f"only {e1} repurchase events in the treatment group (ratio {e0/max(e1,1):.0f}:1) - insufficient precision")

    diag = {
        "sample": "T-Learner 30% holdout test split",
        "n_total": len(df), "n_treated": n1, "n_control": n0,
        "treated_share_pct": round(n1 / len(df) * 100, 2),
        "events_treated": e1, "events_control": e0,
        "event_rate_treated_pct": round(r1 * 100, 3),
        "event_rate_control_pct": round(r0 * 100, 3),
        "event_ratio_control_to_treated": round(e0 / max(e1, 1), 1),
        "qini_end": round(qini_end, 6),
        "qini_end_pct": round(qini_end * 100, 4),
        "auuc_area": round(auuc, 6),
        "auuc_target": AUUC_TARGET,
        "auuc_passed": bool(auuc > AUUC_TARGET),
        "high_decile_n_treated": hi_treated,
        "high_decile_events": hi_events,
        "high_decile_repeat_pct": round(hi_rate, 2) if hi_treated else None,
        "low_decile_n_treated": lo_treated,
        "low_decile_events": lo_events,
        "low_decile_repeat_pct": round(lo_rate, 2) if lo_treated else None,
        "ranking_usable": bool(auuc > AUUC_TARGET and qini_end > 0),
        "verdict": "not_usable_for_targeting",
        "verdict_reasons": verdict_reasons,
        "root_cause": ("The outcome is extremely sparse and the treated group has too few events: repurchase is 2.96% "
                       f"overall, and only {e1} of the {n1:,} treated customers repurchase (ratio {e0/max(e1,1):.0f}:1). "
                       "Class imbalance makes the T-Learner CATE estimates unstable, so Qini/AUUC fall below the baseline. "
                       "This is a precision problem, not evidence that the effect is absent (PSM/DML/IV all agree on direction)."),
        "method_caveat": ("Training used class_weight='balanced_subsample': with only "
                          f"{e1} positive cases it systematically inflates the treated-group probabilities, which is the direct source of the ~+31pp of spurious uplift; "
                          "any re-estimate should drop the class weighting or use a calibrated learner."),
        "root_cause_en": ("The outcome is extremely sparse: repeat runs at 2.96% overall and only "
                          f"{e1} of {n1:,} treated customers repeat \u2014 a {e0/max(e1,1):.0f}:1 event imbalance "
                          f"against {e0} control events. The T-Learner's CATE is therefore unstable and Qini/AUUC "
                          "fall below baseline: a precision problem, not an absent effect (PSM, DML and IV agree)."),
        "method_caveat_en": ("Caveat: class_weight='balanced_subsample' on 12 positives inflates treated-arm "
                             "probabilities \u2014 the source of the +31pp phantom uplift."),
        "next_steps_en": [
            "Widen the event base: use a compound repeat/retention outcome inside a 180-day window, or extend "
            "the observation window",
            "Re-estimate CATE on a randomised holdout (A/B) instead of observational counterfactuals",
            "Drop class_weight and move to an X-Learner / DR-Learner with probability calibration",
            "Re-accept on both the Qini/UAUUC curve and decile monotonicity before any list ships",
        ],
        "next_steps": [
            "Enlarge the event base: use a composite repurchase/retention outcome inside a 180-day window, or extend the observation window",
            "Re-estimate CATE from a randomised hold-out (A/B) instead of observational counterfactuals",
            "Drop class_weight; use an X-Learner or DR-Learner with probability calibration",
            "Accept the re-estimate only if both the Qini/UAUUC curve and decile monotonicity pass; only then publish a list",
        ],
        "top100_note": ("Every name in the h009_top100_uplift.csv top-100 list is an early/on-time customer - that is, customers"
                        "not harmed by delay rank highest on uplift. The list cannot be used for targeting and is kept only as a "
                        " as a counter-example of failed ranking."),
    }
    with open(SNAP / "h009_uplift_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    # ---- curve plot (for the Power BI image visual)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140, facecolor="white")
    x = pop * 100
    ax.axhline(0, color="#66717D", lw=1.2, ls="--")
    ax.fill_between(x, qini, 0, where=(qini < 0), color="#D94B3D", alpha=0.18)
    ax.fill_between(x, qini, 0, where=(qini >= 0), color="#2F7D5B", alpha=0.18)
    ax.plot(x, qini, color="#0B2D52", lw=2.6, label=f"Qini curve  (end {qini_end:+.3f})")
    ax.set_xlabel("Cumulative population, ranked by predicted uplift (%)",
                  fontsize=10, color="#111820")
    ax.set_ylabel("Cumulative incremental gain", fontsize=10, color="#111820")
    ax.set_title(f"Qini / AUUC — AUUC {auuc:+.4f} (target > {AUUC_TARGET}), ranking not usable",
                 fontsize=11.5, color="#0B2D52", fontweight="bold", loc="left")
    ax.tick_params(labelsize=9, colors="#66717D")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C7CED6")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(VIZ / "h009_qini_curve.png")
    plt.close(fig)

    # ---- console summary
    print("=" * 78)
    print("H009-E/F - uplift diagnostics")
    print("=" * 78)
    print(f"sample {len(df):,} (30% hold-out) - treated {n1:,} ({n1/len(df)*100:.0f}%) - control {n0:,}")
    print(f"repurchase events: treated {e1} ({r1*100:.2f}%) - control {e0} ({r0*100:.2f}%)"
          f" - event ratio {e0/max(e1,1):.0f}:1")
    print(f"Qini end {qini_end:+.6f} - AUUC {auuc:+.6f} (target > {AUUC_TARGET})"
          f" - reconciled against h009_uplift_cate.json OK")
    print(f"top deciles D8-D10: {hi_treated} treated customers / {hi_events} events"
          f"| bottom deciles D1-D3: {lo_treated} customers / {lo_events} events")
    print(f"verdict: {'ranking usable for targeting' if diag['ranking_usable'] else 'NOT usable for targeting'}")
    for r in verdict_reasons:
        print(f"   · {r}")
    print()
    print(d10.to_string(index=False))
    print()
    print("OK  saved: h009_uplift_qini_curve.csv / h009_uplift_deciles.csv / "
          "h009_uplift_diagnostics.json / h009_qini_curve.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

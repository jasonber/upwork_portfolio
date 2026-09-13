#!/usr/bin/env python3
"""
H009 ROI scenarios - turning point estimates into ranges
=========================================================
Background
----
In h009_causal_pipeline.py the scenario model's repurchase elasticity is a hard-coded
business assumption:

    repeat_rate_lift = s['sat_boost'] * 4.0      # 4.0 pp of repeat rate per satisfaction point

But the dose-response curve fitted in H009 (h009_marginal_effect.json) shows only about
1.46 pp per point in the low-score band, peaking at 1.56 pp, and turning negative above sat>3.4
(a quadratic extrapolation artefact).

This script leaves the upper bound untouched (the 4.0 business assumption) and fills in the lower
bound with the average marginal effect of the fitted curve over the low-score band (sat 1.0-2.5)
where the intervention actually operates, so the deck can present a range instead of a single number.

Output
----
data/snapshots/h009_intervention_scenarios_range.csv
data/snapshots/h009_roi_range.json

Usage:  python3 scripts/h009_roi_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# repository root (run from there); data/ and outputs/ paths below are relative to it
BASE = Path(__file__).resolve().parents[1]
SNAP = BASE / "data" / "snapshots"

# the satisfaction band in which the intervention actually operates: high-delay customers sit at a
# treated mean of 2.57, so the intervention moves them from the low band into the middle band.
BAND_LO, BAND_HI = 1.0, 2.5

# upper bound: the business assumption in the scenario model (kept identical, not modified)
E_HIGH = 4.0


def identified_elasticity() -> tuple[float, dict]:
    """Average marginal effect (pp per satisfaction point) over the low-score band."""
    d = json.load(open(SNAP / "h009_marginal_effect.json", encoding="utf-8"))
    rows = []
    for k, v in d["marginal_effects"].items():
        a, b = (float(x) for x in k.split("->"))
        rows.append((a, b, float(v)))
    sel = [v for a, b, v in rows if a >= BAND_LO - 1e-9 and b <= BAND_HI + 1e-9]
    e = float(np.mean(sel)) * 100  # pp / point
    detail = {
        "band": [BAND_LO, BAND_HI],
        "n_intervals": len(sel),
        "elasticity_pp": round(e, 4),
        "peak_marginal_pp": round(d["peak_marginal_effect"] * 100, 4),
        "peak_at_satisfaction": round(d["peak_satisfaction"], 3),
        "curve_negative_above": 3.4,
        "method": "mean marginal effect of fitted quadratic logit over the low-satisfaction band",
    }
    return e, detail


def main() -> int:
    scen = pd.read_csv(SNAP / "h009_intervention_scenarios.csv")
    caus = pd.read_csv(SNAP / "h009_causal_data.csv")

    # same caliber as h009_c_counterfactual in h009_causal_pipeline.py
    baseline_churn = caus["Y_churn"].mean()
    n_active = len(caus) * (1 - baseline_churn)
    avg_repeat_gmv = caus[caus["repeat"] == 1]["total_spent"].mean()

    e_low, detail = identified_elasticity()

    print("=" * 78)
    print("H009 ROI - elasticity as a range")
    print("=" * 78)
    print(f"active customers n_active = {n_active:,.0f} - avg spend per repeat buyer = R$ {avg_repeat_gmv:,.2f}")
    print(f"upper elasticity (business assumption, unchanged) = {E_HIGH:.2f} pp/point")
    print(f"lower elasticity (fitted curve sat {BAND_LO}-{BAND_HI}) = {e_low:.3f} pp/point "
          f"({e_low / E_HIGH * 100:.0f}% of the upper bound; peak {detail['peak_marginal_pp']:.2f})")
    print()

    rows = []
    for _, s in scen.iterrows():
        boost = float(s["sat_boost"])
        cpc = float(s["cost_per_customer"])
        gmv_high = float(s["annual_gmv_lift"])
        cost = float(s["cost"])

        # reconciliation: reproduce the existing upper bound (proves the ceiling is untouched)
        calc_high = n_active * (boost * E_HIGH / 100) * 12 * avg_repeat_gmv
        assert abs(calc_high - gmv_high) < 1.0, f"ceiling reconciliation failed {calc_high} vs {gmv_high}"

        gmv_low = n_active * (boost * e_low / 100) * 12 * avg_repeat_gmv
        roi_low = (gmv_low - cost) / cost if cost > 0 else None
        roi_high = (gmv_high - cost) / cost if cost > 0 else None
        rows.append({
            "scenario": s["name"],
            "sat_boost": boost,
            "cost_per_customer": cpc,
            "elasticity_low_pp": round(e_low, 4),
            "elasticity_high_pp": E_HIGH,
            "gmv_low": round(gmv_low, 2),
            "gmv_high": round(gmv_high, 2),
            "cost": round(cost, 2),
            "roi_low": None if roi_low is None else round(roi_low, 4),
            "roi_high": None if roi_high is None else round(roi_high, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(SNAP / "h009_intervention_scenarios_range.csv", index=False, encoding="utf-8")

    print(f"{'scenario':<26s}{'GMV low':>12s}{'GMV high':>12s}{'cost':>11s}{'ROI low':>9s}{'ROI high':>9s}")
    print("-" * 78)
    for r in rows:
        if r["roi_low"] is None:
            print(f"{r['scenario']:<26s}{'-':>12s}{'-':>12s}{'-':>11s}{'base':>9s}{'base':>9s}")
            continue
        flag = "  <- rank unchanged" if r["roi_low"] < 1 <= r["roi_high"] else ""
        print(f"{r['scenario']:<26s}{r['gmv_low']/1e3:>10.1f}k{r['gmv_high']/1e3:>10.1f}k"
              f"{r['cost']/1e3:>9.1f}k{r['roi_low']:>+8.2f}x{r['roi_high']:>8.2f}x{flag}")

    payload = {
        "unit": "R$ / year",
        "elasticity_low": round(e_low, 4),
        "elasticity_high": E_HIGH,
        "elasticity_detail": detail,
        "n_active": round(n_active, 2),
        "avg_repeat_gmv": round(avg_repeat_gmv, 2),
        "scenarios": rows,
        "note": ("Lower bound = fitted dose-response elasticity over the low-satisfaction band where the "
                 "interventions operate; upper bound = the business-assumption elasticity used by the "
                 "original scenario model. The repeat-rate link is not causally identified (IV p = 0.61), "
                 "so both bounds are scenarios, not identified effects."),
    }
    with open(SNAP / "h009_roi_range.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("OK  saved: h009_intervention_scenarios_range.csv")
    print("OK  saved: h009_roi_range.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

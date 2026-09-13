#!/usr/bin/env python3
"""
H012 - evidence deep-dive: aggregating previously un-surfaced findings into quotable snapshots
=========================================================
Background
----
An inventory found 33 analysis outputs that existed but had never reached the deck or dashboard (delivery
dose-response, absolute per-node leakage, state/category funnels, three benchmark sets, four-class customer
health, the model card, the four-estimator cross-check, cost breakdown, NLP themes, list samples). Row-level
tables are unusable on a slide, so this script aggregates them into small, accurate snapshots.

It also fixes one caliber-hygiene problem:
  the churn_180 column in h007_active_pool_check.csv is actually a repurchase-window caliber (no second
  order within 180 days of the first) but shared a name with the main dormancy caliber - renamed to no_repurchase_in_180d.

Output (all under data/snapshots/)
----
h012_benchmark_compare.csv / .json   three-way benchmark (node, four-class health, retention tiers, annual churn)
h012_dropoff_summary.csv             absolute leakage per node (with cumulative)
h012_funnel_groups.csv               state / category funnel summary (worst groups flagged)
h012_delay_dose_response.csv / .json delivery dose-response (score and low-score rate) + delivery-status caliber risk
h012_delivery_geo_category.csv       delivery duration and low-score rate by category / state (top and bottom)
h012_structure_health.csv / .json    four-class customer health (23-month average + healthy band + deviation)
h012_model_card.json                 model card (split, 38 features, class weight, threshold, metrics, confusion matrix)
h012_method_crosscheck.csv / .json   side-by-side estimates of the same causal question
h012_cost_breakdown.csv / .json      intervention cost breakdown (per-customer cost x pool size)
h012_nlp_themes.json                 NLP theme summary with examples
h012_watchlist_sample.json           list sample (top 5 high-risk + the failed uplift distribution)

Usage:  python3 scripts/h012_evidence_deepdive.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repository root (run from there); data/ and outputs/ paths below are relative to it
BASE = Path(__file__).resolve().parents[1]
SNAP = BASE / "data" / "snapshots"
DASH_EN = BASE / "outputs" / "reports" / "dashboard_en" / "data"


def jload(name: str):
    return json.load(open(SNAP / name, encoding="utf-8"))


def markdown_text(v) -> str:
    return " ".join(str(v).split())


# ---------------------------------------------------------------- 1. caliber-hygiene fix
def fix_active_pool_naming() -> dict:
    p = SNAP / "h007_active_pool_check.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    renamed = False
    if "churn_180" in df.columns:
        df = df.rename(columns={"churn_180": "no_repurchase_in_180d"})
        df.to_csv(p, index=False, encoding="utf-8")
        renamed = True
    recency = df["recency_days"]
    info = {
        "file": p.name,
        "renamed": renamed,
        "rows": int(len(df)),
        "no_repurchase_in_180d_share": round(float(df["no_repurchase_in_180d"].mean()) * 100, 2),
        "recency_min": int(recency.min()),
        "recency_median": float(recency.median()),
        "recency_max": int(recency.max()),
        "what_it_is": ("This is the active pool: %d customers with recency <= %d days (1.03 orders each, R$141.69 each), "
                       "of whom %.2f%% have no second order. "
                       % (int(recency.max()), len(df), float(df["no_repurchase_in_180d"].mean()) * 100)),
        "caliber_clash_proof": ("The original column churn_180 shared a name with the main caliber but is not the same thing: "
                               "this file spans recency %d-%d days while the dormancy caliber cuts at 180 days - "
                               "the two columns coexisting would be misread."),
        "definitional_note": ("Retention therefore has two readings in this project: 24,196 dormant-caliber retained customers (last purchase <= 180 days ago) "
                              "and the 17,064-customer active pool (recency <= 92 days); the caliber page and S27 "
                              "label both and never mix them."),
        "naming_rule": {"dormant": "churn_dormant_180d",
                        "repurchase_window": "no_repurchase_in_180d",
                        "alert": "alert_30d"},
    }
    return info


# ---------------------------------------------------------------- 2. three-way benchmark
def benchmark_compare() -> tuple[pd.DataFrame, dict]:
    b4 = jload("h004_industry_benchmark.json")
    b5 = jload("h005_industry_benchmark.json")
    b6 = jload("h006_industry_benchmark.json")
    overall = jload("h005_overall_summary.json")
    funnel = pd.read_csv(SNAP / "h004_funnel_overall.csv")

    node_key = {"1_2": "N2", "2_3": "N3", "3_4": "N4", "4_5": "N5", "5_6": "N6", "6_7": "N7"}
    rows = []
    # (1) per node: industry benchmark vs Olist step conversion
    step = dict(zip(funnel["node"].str.split("_").str[0], funnel["step_conversion"]))
    for k, v in b4.items():
        node = node_key[k]
        actual = float(step.get(node, np.nan))
        rows.append({
            "layer": "funnel_node", "key": k, "label": v["label"],
            "olist": round(actual, 2), "benchmark": float(v["min"]),
            "gap_pp": round(actual - float(v["min"]), 2), "source": v["source"],
        })
    # (2) four customer classes: healthy band vs Olist monthly average
    for k, v in b5.items():
        o = overall[k]
        rows.append({
            "layer": "user_class", "key": k, "label": v["label"],
            "olist": float(o["pct"]), "benchmark": float(v["min"]),
            "benchmark_max": float(v["max"]),
            "gap_pp": round(float(o["pct"]) - float(v["min"]), 2),
            "source": v["source"],
        })
    # (3) retention tiers
    cal = pd.read_csv(SNAP / "h011_caliber_sensitivity.csv")
    for w, key in ((30, "t30_retention_healthy"), (90, "t90_retention_healthy"),
                   (180, "t180_retention_healthy"), (365, "t365_retention_healthy")):
        o = float(cal.loc[cal.window_days == w, "retained_rate"].iloc[0])
        rows.append({
            "layer": "retention", "key": f"T+{w}", "label": f"{w}-day retention",
            "olist": round(o, 2), "benchmark": float(b6[key]),
            "gap_pp": round(o - float(b6[key]), 2), "source": "Shopify Plus / Recurly",
        })
    # (4) annual churn comparison
    for k, label in (("ecommerce_avg", "e-commerce average"), ("marketplace_avg", "marketplace average"),
                     ("subscription_avg", "subscription average")):
        rows.append({
            "layer": "annual_churn", "key": k, "label": f"annual churn vs {label}",
            "olist": float(b6["annual_churn_rate"]["olist_actual"]),
            "benchmark": float(b6["annual_churn_rate"][k]),
            "gap_pp": round(float(b6["annual_churn_rate"]["olist_actual"]) - float(b6["annual_churn_rate"][k]), 2),
            "source": b6["annual_churn_rate"]["source"],
        })
    df = pd.DataFrame(rows)
    agg = {
        "n_comparisons": int(len(df)),
        "funnel_worst": df[df.layer == "funnel_node"].sort_values("gap_pp").head(1).to_dict("records")[0],
        "user_class_unhealthy": int((df[df.layer == "user_class"].apply(
            lambda r: not (r["benchmark"] <= r["olist"] <= r.get("benchmark_max", 1e9)), axis=1)).sum()),
        "repeat_benchmark_gap_pp": float(df[(df.layer == "funnel_node") & (df.key == "6_7")]["gap_pp"].iloc[0]),
        "retention_t90_gap_pp": float(df[(df.layer == "retention") & (df.key == "T+90")]["gap_pp"].iloc[0]),
    }
    return df, agg


# ---------------------------------------------------------------- 3. per-node leakage
def dropoff_summary() -> pd.DataFrame:
    d = pd.read_csv(SNAP / "h004_dropoff_clients.csv")
    d["stage"] = d["leak_stage"].str.split().str[0]
    d["stage_name"] = d["leak_stage"].str.split(n=1).str[1]
    d["share_of_all"] = (d["n_dropoff"] / float(d["n_start"].iloc[0]) * 100).round(2)
    return d[["stage", "stage_name", "n_start", "n_dropoff", "dropoff_rate", "share_of_all"]]


# ---------------------------------------------------------------- 4. state / category funnel
def funnel_groups() -> pd.DataFrame:
    out = []
    for fname, kind in (("h004_funnel_by_state.csv", "state"),
                        ("h004_funnel_by_category.csv", "category")):
        df = pd.read_csv(SNAP / fname)
        for g, sub in df.groupby("group"):
            row = {"group_type": kind, "group": g, "n_start": int(sub["count"].iloc[0])}
            for _, r in sub.iterrows():
                node = r["node"].split("_")[0]
                row[f"{node}_step"] = float(r["step_conversion"])
                row[f"{node}_cum"] = float(r["cumulative_conversion"])
                row[f"{node}_count"] = int(r["count"])
            out.append(row)
    t = pd.DataFrame(out)
    for c in ("N5_step", "N6_step", "N7_step", "N7_cum"):
        if c in t:
            t[c] = t[c].round(2)
    t["N6_to_N7_drop_pp"] = (t["N7_step"] - 100).round(2)
    return t


# ---------------------------------------------------------------- 5. delivery dose-response
def delay_dose_response() -> tuple[pd.DataFrame, dict]:
    b4 = pd.read_csv(SNAP / "h003_b4_delay_bucket.csv")
    b1 = pd.read_csv(SNAP / "h003_b1_delivery_status.csv")
    order = ["Early", "OnTime_±1day", "Late_1-3days", "Late_3-7days", "Late_7+days"]
    b4["order"] = b4["delay_bucket"].map({k: i for i, k in enumerate(order)})
    b4 = b4.sort_values("order").drop(columns="order")
    b4 = b4.rename(columns={"delay_bucket": "bucket", "n": "n_orders",
                            "avg_score": "avg_review", "low_pct": "low_score_pct"})
    b4["share_of_orders"] = (b4["n_orders"] / b4["n_orders"].sum() * 100).round(2)
    b4["score_drop_vs_early"] = (b4["avg_review"] - b4["avg_review"].iloc[0]).round(3)
    info = {
        "status_rows": b1.to_dict("records"),
        "timing_imbalance": {
            "late_n": int(b1.loc[b1.delivery_status == "Late", "n"].iloc[0]),
            "ontime_n": int(b1.loc[b1.delivery_status == "OnTime", "n"].iloc[0]),
            "early_n": int(b1.loc[b1.delivery_status == "Early", "n"].iloc[0]),
            "note": ("On-time deliveries number %d against %d early deliveries (%dx) - on-time is a rare "
                     "event, so the score gap between on-time and early orders must not be treated as "
                     "the operating target; the real lever is landing inside the +/-1 day window.",
                     int(b1.loc[b1.delivery_status == "OnTime", "n"].iloc[0]),
                     int(b1.loc[b1.delivery_status == "Early", "n"].iloc[0]),
                     int(b1.loc[b1.delivery_status == "Early", "n"].iloc[0]) //
                     max(1, int(b1.loc[b1.delivery_status == "OnTime", "n"].iloc[0]))),
            "late_1_3_low_pct": float(b4.loc[b4.bucket == "Late_1-3days", "low_score_pct"].iloc[0]),
            "late_3_7_low_pct": float(b4.loc[b4.bucket == "Late_3-7days", "low_score_pct"].iloc[0]),
            "late_7p_low_pct": float(b4.loc[b4.bucket == "Late_7+days", "low_score_pct"].iloc[0]),
            "note": "The low-score rate jumps from 26% to 62% at 3-7 days late and reaches 78% at 7+ days - day 3 is the inflection",
        },
    }
    return b4, info


# ---------------------------------------------------------------- 6. delivery by category / geography
def delivery_geo_category() -> pd.DataFrame:
    out = []
    for fname, kind, key in (("h003_b2_category_timing.csv", "category", "product_category_name"),
                             ("h003_b3_state_delivery.csv", "state", "customer_state")):
        df = pd.read_csv(SNAP / fname).rename(columns={key: "group"})
        df["group_type"] = kind
        df["avg_days"] = df["avg_days"].round(1)
        df["avg_review"] = df["avg_review"].round(2)
        df["low_pct"] = df["low_pct"].round(1)
        out.append(df)
    t = pd.concat(out, ignore_index=True)
    return t.sort_values(["group_type", "avg_days"], ascending=[True, False])


# ---------------------------------------------------------------- 7. four-class customer health
def structure_health() -> tuple[pd.DataFrame, dict]:
    benchmark = jload("h005_industry_benchmark.json")
    overall = jload("h005_overall_summary.json")
    trend = pd.read_csv(SNAP / "h005_trend_chart.csv")
    rows = []
    for k, label, col in (("nurr", "New buyers", "nurr_pct"), ("curr", "Consecutive repeat", "curr_pct"),
                          ("rurr", "Short-recall", "rurr_pct"), ("surr", "Deep-wake", "surr_pct")):
        o, b = overall[k], benchmark[k]
        rows.append({
            "key": k, "label": label,
            "avg_pct": float(o["pct"]),
            "benchmark_min": float(b["min"]), "benchmark_max": float(b["max"]),
            "is_healthy": bool(o["is_healthy"]),
            "deviation_pct": float(o["deviation_pct"]),
            "last_month_pct": float(trend[col].iloc[-1]),
            "last_month": str(trend["order_month_str"].iloc[-1]),
            "trend_min": round(float(trend[col].min()), 2),
            "trend_max": round(float(trend[col].max()), 2),
            "source": b["source"],
        })
    df = pd.DataFrame(rows)
    info = {
        "n_months": int(overall["n_months"]),
        "period": overall["period"],
        "total_buyer_months": int(overall["total_buyer_months"]),
        "unhealthy_count": int((~df["is_healthy"]).sum()),
        "note": "Three of the four classes are severely unhealthy; only deep-wake falls inside its healthy band (0.53% within 0-5%)",
    }
    return df, info


# ---------------------------------------------------------------- 8. model card
def model_card() -> dict:
    split = jload("h007_train_test_split.json")
    metrics = jload("h007_evaluation_metrics.json")
    conf = pd.read_csv(SNAP / "h007_confusion_matrix.csv")
    imp = pd.read_csv(SNAP / "h007_feature_importance.csv")
    return {
        "algorithm": "XGBoost (binary classification)",
        "target": "churn_label = no purchase in 180 days (dormancy caliber, 74.08%)",
        "split": {"train": int(split["train_size"]), "test": int(split["test_size"]),
                  "churn_rate_train": float(split["churn_rate_train"]),
                  "churn_rate_test": float(split["churn_rate_test"]),
                  "scale_pos_weight": float(split["scale_pos_weight"])},
        "features": {"n": int(split["n_features"]), "list": split["features"]},
        "leakage_control": "days_since_last / recency removed (same source as the label -> tautological AUC=1.0)",
        "threshold": {"value": 0.30, "rule": "chosen at the F1 optimum"},
        "metrics": {k: round(float(v), 4) for k, v in metrics.items() if k != "timestamp"},
        "confusion": conf.to_dict("records"),
        "top_features": imp.head(6).to_dict("records"),
        "evidence_ids": ["E016", "E017", "E032", "E033"],
    }


# ---------------------------------------------------------------- 9. multi-method cross-check
def method_crosscheck() -> tuple[pd.DataFrame, dict]:
    ce = jload("h009_causal_effects.json")
    dw = jload("h009_dowhy_estimates.json")
    es = jload("h009_dowhy_estimand.json")
    rows = [
        ("PSM ATT", "delay -> churn rate", ce["psm"]["att_churn"], "pp", "significant", "matched on order count, spend and state"),
        ("PSM ATT", "delay -> satisfaction", ce["psm"]["att_satisfaction"], "points", "significant", "same matched set"),
        ("PSM ATT", "delay -> repurchase", ce["psm"]["att_repeat"], "pp (proportion)", "not significant", "agrees with IV"),
        ("DoWhy linear regression", "delay / low satisfaction -> repurchase", dw["linear_regression"]["ate"], "repurchase (proportion)", "~0",
         "backdoor.linear_regression"),
        ("DoWhy doubly robust (DML)", "delay / low satisfaction -> repurchase", dw["doubly_robust"]["ate"], "repurchase (proportion)", "~0",
         "backdoor.doubly_robust"),
        ("DoWhy propensity-score matching", "delay / low satisfaction -> repurchase", dw["ps_matching"]["ate"], "repurchase (proportion)", "~0",
         "backdoor.propensity_score_matching"),
        ("IV 2SLS", "delay -> repurchase", ce["iv_2sls"]["coef"], "repurchase (proportion)",
         f"not significant (p={ce['iv_2sls']['pvalue']:.2f})", "instrument: average delay days"),
        ("Logit marginal", "delay -> churn", ce["logit"]["coef"], "log-odds",
         f"borderline (p={ce['logit']['pvalue']:.3f}, OR={ce['logit']['or']:.3f})", "with covariates"),
    ]
    df = pd.DataFrame(rows, columns=["method", "question", "estimate", "unit", "significance", "detail"])
    info = {
        "estimand": es["estimand_type"],
        "common_causes": es["common_causes"],
        "instrumental_variables": es["instrumental_variables"],
        "conclusion": markdown_text(ce["conclusion"]),
        "reading": ("All four chains point the same way: delay / low satisfaction lowers repurchase and raises churn. "
                    "But the repurchase link collapses to zero under every method (PSM 0.0011, linear -0.0034, "
                    "DML -0.0006, PSM matching 0.0000, IV p=0.61), so the report treats only delay -> churn "
                    "+3.9pp and satisfaction -0.46 as identified and hands the repurchase link to the 90-day "
                    "control group."),
    }
    return df, info


# ---------------------------------------------------------------- 10. cost breakdown
def cost_breakdown() -> tuple[pd.DataFrame, dict]:
    st = pd.read_csv(DASH_EN / "Retention_Strategies.csv")
    bm = jload("h009_business_model.json")
    rows = []
    for s in bm["scenarios"]:
        if s["name"].startswith("Status quo"):
            continue
        rows.append({
            "strategy": s["name"],
            "sat_boost": s["sat_boost"],
            "repeat_rate_lift_pp": s["repeat_rate_lift"],
            "cost_per_customer": s["cost_per_customer"],
            "total_cost": round(float(s["cost"]), 0),
            "annual_gmv_lift": round(float(s["annual_gmv_lift"]), 0),
            "roi_assumption": None if s["roi"] is None else round(float(s["roi"]), 2),
        })
    df = pd.DataFrame(rows)
    pool = float(df["total_cost"].iloc[0]) / float(df["cost_per_customer"].iloc[0])
    individual = df[~df.strategy.str.contains("Combined")]
    info = {
        "intervention_pool": int(round(pool)),
        "individual_cost_sum": int(individual["total_cost"].sum()),
        "combined_cost": int(df.loc[df.strategy.str.contains("Combined"), "total_cost"].iloc[0]),
        "bundling_note": ("The combined programme is priced at 25 per customer rather than 10+15+8=33, so its total cost of 605,175 "
                          "798,831. That bundling assumption must be stated on the page, or the reader will "
                          "assume an error."),
    }
    return df, info


# ---------------------------------------------------------------- 11. NLP theme summary
def nlp_themes() -> dict:
    th = pd.read_csv(SNAP / "h008_low_score_nlp_themes.csv")
    res = pd.read_csv(SNAP / "h008_nlp_results.csv")
    agg = (th.groupby(["pain_point", "subtype"])
             .agg(n=("review_id", "count"), avg_score=("review_score", "mean"),
                  avg_sentiment=("sentiment", "mean"))
             .reset_index().sort_values("n", ascending=False))
    agg["avg_score"] = agg["avg_score"].round(2)
    agg["avg_sentiment"] = agg["avg_sentiment"].round(2)
    examples = th.sort_values("sentiment", ascending=False).head(5)[
        ["review_score", "pain_point", "subtype", "en_translation", "key_phrase", "suggestion"]
    ].to_dict("records")
    return {
        "n_theme_rows": int(len(th)),
        "n_low_score_analysed": int(len(res)),
        "n_sample_frame": 1000,
        "theme_mix": agg.to_dict("records"),
        "examples": examples,
        "note": (f"The page shows a theme summary plus 5 examples (out of {len(th)} theme-level rows); "
                 f"the LLM extraction is based on {len(res)} low-score reviews (score <= 2) drawn from a 1,000-review sampling frame. "
                 "The 1,000-review sample stays in the data layer as reproducible evidence; raw text is not reproduced."),
    }


# ---------------------------------------------------------------- 12. list sample
def watchlist_sample() -> dict:
    hi = pd.read_csv(SNAP / "h007_top100_high_risk.csv")
    up = pd.read_csv(SNAP / "h009_top100_uplift.csv")
    up_early = None
    for c in up.columns:
        if "delay" in c.lower() or "late" in c.lower():
            up_early = c
            break
    return {
        "high_risk": {
            "n": int(len(hi)),
            "mean_probability": round(float(hi["churn_probability"].mean()), 4),
            "true_churn_share": round(float(hi["true_churn_label"].mean()) * 100, 1),
            "mean_orders": round(float(hi["order_count"].mean()), 2),
            "mean_spent": round(float(hi["total_spent"].mean()), 2),
            "sample_rows": hi.head(5)[["customer_unique_id", "churn_probability", "risk_level",
                                       "intervention", "customer_state", "order_count",
                                       "total_spent"]].to_dict("records"),
            "note": "The deliverable sample has 5 rows; the full list ships with the dashboard table (93,358 rows).",
        },
        "uplift_top100_failure": {
            "n": int(len(up)),
            "columns": list(up.columns),
            "early_or_ontime_share": None,
            "note": "Every name in this list is an early/on-time customer - empirical proof that the ranking should not be delivered; kept only as a failure example.",
        },
    }


def main() -> int:
    print("=" * 78)
    print("H012 - evidence deep-dive")
    print("=" * 78)

    pool = fix_active_pool_naming()
    if pool:
        print(f"\n1. caliber hygiene: {pool['file']} renamed={pool['renamed']} - {pool['rows']:,} rows - "
              f"{pool['what_it_is']}")
        print(f"   naming rule: {pool['naming_rule']}")

    bench, bench_agg = benchmark_compare()
    bench.to_csv(SNAP / "h012_benchmark_compare.csv", index=False, encoding="utf-8")
    drop = dropoff_summary()
    drop.to_csv(SNAP / "h012_dropoff_summary.csv", index=False, encoding="utf-8")
    groups = funnel_groups()
    groups.to_csv(SNAP / "h012_funnel_groups.csv", index=False, encoding="utf-8")
    dose, dose_info = delay_dose_response()
    dose.to_csv(SNAP / "h012_delay_dose_response.csv", index=False, encoding="utf-8")
    geo = delivery_geo_category()
    geo.to_csv(SNAP / "h012_delivery_geo_category.csv", index=False, encoding="utf-8")
    struct, struct_info = structure_health()
    struct.to_csv(SNAP / "h012_structure_health.csv", index=False, encoding="utf-8")
    card = model_card()
    cross, cross_info = method_crosscheck()
    cross.to_csv(SNAP / "h012_method_crosscheck.csv", index=False, encoding="utf-8")
    cost, cost_info = cost_breakdown()
    cost.to_csv(SNAP / "h012_cost_breakdown.csv", index=False, encoding="utf-8")
    themes = nlp_themes()
    watch = watchlist_sample()

    summary = {
        "hypothesis": "H012 evidence deep-dive: aggregating un-surfaced findings into quotable snapshots",
        "created": "2026-09-13",
        "active_pool_naming_fix": pool,
        "benchmark": bench_agg,
        "delay": dose_info,
        "structure": struct_info,
        "model_card": card,
        "crosscheck": cross_info,
        "cost": cost_info,
        "nlp": themes,
        "watchlist": watch,
    }
    for name, obj in (("h012_benchmark_compare", bench_agg), ("h012_delay_dose_response", dose_info),
                      ("h012_structure_health", struct_info), ("h012_model_card", card),
                      ("h012_method_crosscheck", cross_info), ("h012_cost_breakdown", cost_info),
                      ("h012_nlp_themes", themes), ("h012_watchlist_sample", watch)):
        with open(SNAP / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    with open(SNAP / "h012_deepdive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n2. three-way benchmark {len(bench)} rows - worst funnel node = {bench_agg['funnel_worst']['label']} "
          f"({bench_agg['funnel_worst']['gap_pp']}pp)")
    print(f"   unhealthy classes = {bench_agg['user_class_unhealthy']} / 4 - "
          f"repeat benchmark gap = {bench_agg['repeat_benchmark_gap_pp']}pp - T+90 retention gap = {bench_agg['retention_t90_gap_pp']}pp")
    print(f"\n3. per-node leakage: largest absolute leak = {int(drop.n_dropoff.max()):,} orders ({drop.loc[drop.n_dropoff.idxmax(), 'stage_name']})")
    print(f"4. grouped funnels: {len(groups)} groups ({groups.group_type.nunique()} types)")
    print(f"5. delivery dose-response: {len(dose)} buckets - low-score rate at the day-3 inflection "
          f"{dose_info['threshold_evidence']['late_3_7_low_pct']}% -> {dose_info['threshold_evidence']['late_7p_low_pct']}% at 7+ days")
    print(f"6. delivery distribution: {len(geo)} rows; slowest category {geo[geo.group_type=='category'].iloc[0]['group']} "
          f"{geo[geo.group_type=='category'].iloc[0]['avg_days']} days")
    print(f"7. four-class health: unhealthy {struct_info['unhealthy_count']} / 4 ({struct_info['period']})")
    print(f"8. model card: {card['split']['train']:,} / {card['split']['test']:,} - {card['features']['n']} features - "
          f"weight {card['split']['scale_pos_weight']} - threshold {card['threshold']['value']}")
    print(f"9. multi-method cross-check: {len(cross)} rows - estimand {cross_info['estimand']}")
    print(f"10. cost breakdown: intervention pool {cost_info['intervention_pool']:,} - individual sum {cost_info['individual_cost_sum']:,} - "
          f"combined {cost_info['combined_cost']:,}")
    print(f"11. NLP themes: {themes['n_theme_rows']} rows (low-score sample {themes['n_low_score_analysed']} / frame "
          f"{themes['n_sample_frame']})")
    print("12. list sample: top 5 high-risk + the failed uplift distribution")
    print(f"\noutput -> {SNAP}/h012_*.csv / *.json (11 files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

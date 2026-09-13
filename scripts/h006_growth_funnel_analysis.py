#!/usr/bin/env python3
"""
H006: growth funnel - multi-dimensional churn diagnosis
=========================================
H006 covers four analyses (XGBoost was split out into H007):

  6.1 time-window churn curve (T+30/60/90/180/365)
  6.2 cohort retention matrix (grouped by acquisition month)
  6.3 pre-churn behaviour of lost customers (last-order features)
  6.4 industry benchmark comparison

Usage:
    python3 scripts/h006_growth_funnel_analysis.py

Output:
    data/snapshots/h006_time_window_churn.csv
    data/snapshots/h006_cohort_retention_matrix.csv
    data/snapshots/h006_last_order_features.csv
    data/snapshots/h006_industry_benchmark.json
    data/snapshots/h006_churn_by_state.csv
    data/snapshots/h006_churn_by_category.csv
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import mysql.connector
import pandas as pd
import numpy as np

BASE = Path(__file__).parent.parent.parent
SNAPSHOTS = BASE / "data" / "snapshots"
SNAPSHOTS.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("SQLPUB_HOST", "mysql6.sqlpub.com"),
    "port": int(os.getenv("SQLPUB_PORT", "3311")),
    "user": os.getenv("SQLPUB_USER", "zz0008"),
    "password": os.getenv("SQLPUB_PASSWORD", "yjom5GVTLAzPC3O6"),
    "database": os.getenv("SQLPUB_DATABASE", "zz_free"),
    "connect_timeout": 30,
}

# industry benchmarks
INDUSTRY_BENCHMARK = {
    "annual_churn_rate": {
        "olist_actual": 74.1,  # from H003-C
        "ecommerce_avg": 25.0,  # Shopify Plus 2024
        "subscription_avg": 5.0,  # subscription models churn less
        "marketplace_avg": 40.0,  # marketplaces sit in between
        "source": "Shopify Plus 2024, Recurly Research",
    },
    "t30_retention_healthy": 60.0,  # 60% should still be active at 30 days
    "t90_retention_healthy": 40.0,  # 40% at 90 days
    "t180_retention_healthy": 30.0,  # 30% at 180 days
    "t365_retention_healthy": 25.0,  # 25% at 365 days
}


def fetch_orders(conn) -> pd.DataFrame:
    """Fetch all delivered orders with customer details and reviews."""
    print("Fetching order data ...")
    query = """
    SELECT
        o.order_id,
        c.customer_unique_id,
        c.customer_state,
        o.order_purchase_timestamp,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        r.review_score
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
    """
    df = pd.read_sql(query, conn)

    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col])

    # delivery delay in days
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    df["is_late"] = (df["delivery_delay_days"] > 0).astype(int)

    print(f"   OK  {len(df):,} orders / {df['customer_unique_id'].nunique():,} distinct buyers")
    return df


def compute_time_window_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the T+30/60/90/180/365 time-window churn curve."""
    print("Computing the time-window churn curve ...")

    # 1) each customer's first order date
    first_orders = (
        df.groupby("customer_unique_id")["order_purchase_timestamp"]
        .min()
        .reset_index()
        .rename(columns={"order_purchase_timestamp": "first_order_ts"})
    )
    first_orders["first_order_month"] = first_orders["first_order_ts"].dt.to_period("M").dt.to_timestamp()

    # 2) each customer's later order dates (second order onwards)
    subsequent = (
        df.sort_values(["customer_unique_id", "order_purchase_timestamp"])
        .groupby("customer_unique_id")
        .agg(
            first_order_ts=("order_purchase_timestamp", "min"),
            second_order_ts=("order_purchase_timestamp", lambda x: x.iloc[1] if len(x) > 1 else pd.NaT),
        )
        .reset_index()
    )

    # 3) did a second order arrive within N days of the first?
    windows = [30, 60, 90, 180, 365]
    results = []
    for w in windows:
        subsequent[f"retained_{w}d"] = (
            (subsequent["second_order_ts"].notna()) &
            ((subsequent["second_order_ts"] - subsequent["first_order_ts"]).dt.days <= w)
        ).astype(int)

    # 4) aggregate
    n_total = len(subsequent)
    summary = []
    for w in windows:
        retained = subsequent[f"retained_{w}d"].sum()
        churned = n_total - retained
        summary.append({
            "window_days": int(w),
            "n_total_customers": int(n_total),
            "n_retained": int(retained),
            "n_churned": int(churned),
            "retention_rate": round(retained / n_total * 100, 2),
            "churn_rate": round(churned / n_total * 100, 2),
        })

    summary_df = pd.DataFrame(summary)
    summary_df["window_days"] = summary_df["window_days"].astype(int)
    print("   time-window churn curve:")
    for _, r in summary_df.iterrows():
        wd = int(r['window_days'])
        rr = float(r['retention_rate'])
        cr = float(r['churn_rate'])
        print(f"   T+{wd:>3d}: retention {rr:5.2f}% | churn {cr:5.2f}%")
    return summary_df


def compute_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Cohort retention matrix."""
    print("Computing the cohort retention matrix ...")

    # 1) each customer's acquisition month
    first_orders = (
        df.groupby("customer_unique_id")["order_purchase_timestamp"]
        .min()
        .reset_index()
    )
    first_orders["cohort_month"] = first_orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    # 2) all orders per customer per month
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    user_months = (
        df.groupby(["customer_unique_id", "order_month"])
        .size()
        .reset_index(name="order_count")
    )

    # 3) attach the cohort month
    user_months = user_months.merge(first_orders[["customer_unique_id", "cohort_month"]], on="customer_unique_id")
    user_months["months_since_acquisition"] = (
        (user_months["order_month"].dt.year - user_months["cohort_month"].dt.year) * 12 +
        (user_months["order_month"].dt.month - user_months["cohort_month"].dt.month)
    )

    # 4) active customers per cohort at M+N
    cohort_data = (
        user_months.groupby(["cohort_month", "months_since_acquisition"])
        ["customer_unique_id"].nunique()
        .reset_index(name="active_customers")
    )

    # 5) cohort size at M+0
    cohort_sizes = (
        cohort_data[cohort_data["months_since_acquisition"] == 0]
        .set_index("cohort_month")["active_customers"]
        .to_dict()
    )

    # 6) retention rate
    cohort_data["cohort_size"] = cohort_data["cohort_month"].map(cohort_sizes)
    cohort_data["retention_rate"] = (
        cohort_data["active_customers"] / cohort_data["cohort_size"] * 100
    ).round(2)

    print(f"   OK  {len(cohort_sizes)} cohorts x {cohort_data['months_since_acquisition'].max() + 1} months")
    return cohort_data


def compute_last_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Last-order features of churned customers."""
    print("Analysing last-order features of churned customers ...")

    # 1) each customer's last order
    last_orders = (
        df.sort_values(["customer_unique_id", "order_purchase_timestamp"])
        .groupby("customer_unique_id")
        .agg(
            first_order_ts=("order_purchase_timestamp", "min"),
            last_order_ts=("order_purchase_timestamp", "max"),
            last_review_score=("review_score", "last"),
            last_is_late=("is_late", "last"),
            last_delivery_delay=("delivery_delay_days", "last"),
            order_count=("order_id", "count"),
            state=("customer_state", "first"),
        )
        .reset_index()
    )

    # 2) days since the snapshot reference date (2018-10-31)
    cutoff = pd.Timestamp("2018-10-31")
    last_orders["days_since_last"] = (cutoff - last_orders["last_order_ts"]).dt.days
    last_orders["churn_label"] = (last_orders["days_since_last"] > 180).astype(int)

    # 3) churned vs active comparison
    comparison = []
    for label, group_name in [(0, "active"), (1, "churned")]:
        sub = last_orders[last_orders["churn_label"] == label]
        comparison.append({
            "segment": group_name,
            "n_customers": len(sub),
            "avg_days_since_last": round(sub["days_since_last"].mean(), 1),
            "avg_last_review_score": round(sub["last_review_score"].mean(), 2),
            "pct_low_score": round((sub["last_review_score"] <= 2).mean() * 100, 2),
            "pct_late": round(sub["last_is_late"].mean() * 100, 2),
            "avg_delivery_delay_days": round(sub["last_delivery_delay"].mean(), 2),
            "pct_one_time_buyer": round((sub["order_count"] == 1).mean() * 100, 2),
        })

    comp_df = pd.DataFrame(comparison)

    # 4) churn rate by state
    churn_by_state = (
        last_orders.groupby("state")
        .agg(
            n_customers=("customer_unique_id", "count"),
            n_churned=("churn_label", "sum"),
            avg_last_review=("last_review_score", "mean"),
        )
        .reset_index()
    )
    churn_by_state["churn_rate"] = round(churn_by_state["n_churned"] / churn_by_state["n_customers"] * 100, 2)
    churn_by_state = churn_by_state.sort_values("churn_rate", ascending=False)
    churn_by_state["avg_last_review"] = churn_by_state["avg_last_review"].round(2)

    print("   churned vs active customers:")
    print(comp_df.to_string(index=False))

    return last_orders, comp_df, churn_by_state


def main():
    print("=" * 60)
    print("H006: growth funnel - multi-dimensional churn diagnosis")
    print("=" * 60)

    # 1) database connection
    conn = mysql.connector.connect(**DB_CONFIG)
    print(f"OK  connected: {DB_CONFIG['host']}/{DB_CONFIG['database']}\n")

    # 2) fetch data
    df = fetch_orders(conn)
    conn.close()

    # 3) time-window churn curve
    time_window = compute_time_window_churn(df)
    time_window.to_csv(SNAPSHOTS / "h006_time_window_churn.csv", index=False, encoding="utf-8")
    print(f"   💾 h006_time_window_churn.csv\n")

    # 4) cohort retention matrix
    cohort = compute_cohort_retention(df)
    cohort_pivot = cohort.pivot(index="cohort_month", columns="months_since_acquisition", values="retention_rate")
    cohort_pivot.to_csv(SNAPSHOTS / "h006_cohort_retention_matrix.csv", encoding="utf-8")
    print(f"   💾 h006_cohort_retention_matrix.csv ({cohort_pivot.shape})\n")

    # 5) pre-churn behaviour
    last_orders, comp_df, churn_by_state = compute_last_order_features(df)
    comp_df.to_csv(SNAPSHOTS / "h006_last_order_features.csv", index=False, encoding="utf-8")
    churn_by_state.head(27).to_csv(SNAPSHOTS / "h006_churn_by_state.csv", index=False, encoding="utf-8")
    print(f"   💾 h006_last_order_features.csv")
    print(f"   💾 h006_churn_by_state.csv\n")

    # 6) industry benchmarks
    with open(SNAPSHOTS / "h006_industry_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(INDUSTRY_BENCHMARK, f, ensure_ascii=False, indent=2)
    print(f"   💾 h006_industry_benchmark.json\n")

    print("=" * 60)
    print("OK  H006 data generated")
    print("=" * 60)


if __name__ == "__main__":
    main()

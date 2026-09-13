#!/usr/bin/env python3
"""
H005: monthly buyer structure split into four classes (nurr / curr / rurr / surr)
================================================================
For each (customer_unique_id, month) pair, classifies the buyer's order in that month:

  nurr  first order ever, placed this month
  curr  ordered this month and last month (gap <= 31 days)
  rurr  ordered this month, not last month, gap 31-180 days
  surr  ordered this month after a gap longer than 180 days

Usage:
    python3 scripts/h005_monthly_user_structure.py

Output:
    data/snapshots/h005_customer_month_classified.csv   (per-customer-month labels)
    data/snapshots/h005_monthly_user_structure.csv      (monthly four-class share matrix)
    data/snapshots/h005_overall_summary.json           (period summary + industry benchmark)
    data/snapshots/h005_trend_chart.csv                (trend chart data)
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import mysql.connector
import pandas as pd

BASE = Path(__file__).parent.parent.parent
SNAPSHOTS = BASE / "data" / "snapshots"
SNAPSHOTS.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("SQLPUB_HOST", "mysql6.sqlpub.com"),
    "port": int(os.getenv("SQLPUB_PORT", "3311")),
    "user": os.getenv("SQLPUB_USER", "zz0008"),
    "password": os.getenv("SQLPUB_PASSWORD", "yjom5GVTLAzPC3O6"),
    "database": os.getenv("SQLPUB_DATABASE", "zz_free"),
    "connect_timeout": 15,
}

# e-commerce industry benchmarks (healthy platform)
INDUSTRY_BENCHMARK = {
    "nurr": {"min": 30.0, "max": 40.0, "label": "New buyers, monthly average share", "source": "Shopify Plus 2024"},
    "curr": {"min": 30.0, "max": 100.0, "label": "Consecutive repeat buyers, monthly average share", "source": "Cohort Analysis Best Practice"},
    "rurr": {"min": 15.0, "max": 25.0, "label": "Short-recall buyers, monthly average share", "source": "E-commerce Retention Benchmark"},
    "surr": {"min": 0.0, "max": 5.0, "label": "Deep-wake buyers, monthly average share", "source": "Healthy E-commerce Standard"},
}


def fetch_orders(conn) -> pd.DataFrame:
    """Fetch all delivered orders with customer_unique_id."""
    print("Fetching order data ...")
    query = """
    SELECT
        o.order_id,
        c.customer_unique_id,
        o.order_purchase_timestamp
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    print(f"   OK  {len(df):,} orders / {df['customer_unique_id'].nunique():,} distinct buyers")
    return df


def build_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Assign one of four classes to each (customer, month) pair."""
    print("Building (customer, month) four-class labels ...")

    # 1) each customer's first order within each month
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    first_in_month = (
        df.groupby(["customer_unique_id", "order_month"])["order_purchase_timestamp"]
        .min()
        .reset_index()
        .rename(columns={"order_purchase_timestamp": "first_order_in_month"})
    )

    # 2) the customer's previous order date before that first-in-month order
    first_in_month = first_in_month.sort_values(["customer_unique_id", "first_order_in_month"])
    first_in_month["prev_order_date"] = first_in_month.groupby("customer_unique_id")[
        "first_order_in_month"
    ].shift(1)

    # 3) days since that previous order
    first_in_month["days_since_prev"] = (
        first_in_month["first_order_in_month"] - first_in_month["prev_order_date"]
    ).dt.days

    # 4) four-class assignment
    def classify(row) -> str:
        if pd.isna(row["prev_order_date"]):
            return "nurr"
        gap = row["days_since_prev"]
        if gap <= 31:  # within the previous month
            return "curr"
        if gap <= 180:
            return "rurr"
        return "surr"

    first_in_month["user_type"] = first_in_month.apply(classify, axis=1)
    print(f"   OK  {len(first_in_month):,} (customer, month) records")
    print(f"   four-class distribution: {first_in_month['user_type'].value_counts().to_dict()}")
    return first_in_month


def aggregate_monthly(df_classified: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the monthly four-class share matrix."""
    print("Aggregating monthly four-class shares ...")

    # long to wide
    pivot = (
        df_classified.groupby(["order_month", "user_type"])["customer_unique_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    # make sure all four classes are present
    for col in ["nurr", "curr", "rurr", "surr"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[["nurr", "curr", "rurr", "surr"]]
    pivot["total_buyers"] = pivot.sum(axis=1)
    for col in ["nurr", "curr", "rurr", "surr"]:
        pivot[f"{col}_pct"] = (pivot[col] / pivot["total_buyers"] * 100).round(2)

    pivot = pivot.reset_index().sort_values("order_month")
    print(f"   OK  {len(pivot)} months ({pivot['order_month'].min()} ~ {pivot['order_month'].max()})")
    return pivot


def compute_overall_summary(monthly: pd.DataFrame) -> dict:
    """Period summary with industry benchmark comparison."""
    print("Summarising the period against industry benchmarks ...")

    total = monthly["total_buyers"].sum()
    summary = {
        "period": f"{monthly['order_month'].min().strftime('%Y-%m')} ~ {monthly['order_month'].max().strftime('%Y-%m')}",
        "n_months": int(len(monthly)),
        "total_buyer_months": int(total),
    }

    for col in ["nurr", "curr", "rurr", "surr"]:
        cnt = int(monthly[col].sum())
        pct = round(cnt / total * 100, 2)
        bench = INDUSTRY_BENCHMARK[col]
        is_healthy = bool(bench["min"] <= pct <= bench["max"])  # cast to a Python bool
        # deviation = (actual - benchmark midpoint) / benchmark midpoint
        benchmark_mid = (bench["min"] + bench["max"]) / 2
        deviation = round((pct - benchmark_mid) / benchmark_mid * 100, 1) if benchmark_mid > 0 else 0

        summary[col] = {
            "count": int(cnt),
            "pct": float(pct),
            "benchmark_min": float(bench["min"]),
            "benchmark_max": float(bench["max"]),
            "benchmark_label": bench["label"],
            "is_healthy": is_healthy,
            "deviation_pct": float(deviation),
            "source": bench["source"],
        }

    return summary


def build_trend_chart(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build the data for the stacked area chart."""
    print("Building the trend chart data ...")
    chart = monthly[["order_month", "nurr_pct", "curr_pct", "rurr_pct", "surr_pct"]].copy()
    chart["order_month_str"] = chart["order_month"].dt.strftime("%Y-%m")
    return chart


def main():
    print("=" * 60)
    print("H005: monthly buyer structure split into four classes")
    print("=" * 60)

    # 1) database connection
    conn = mysql.connector.connect(**DB_CONFIG)
    print(f"OK  connected: {DB_CONFIG['host']}/{DB_CONFIG['database']}\n")

    # 2) fetch data
    df = fetch_orders(conn)
    conn.close()

    # 3) four-class assignment
    classified = build_classification(df)

    # 4) detailed classification table
    classified_out = classified[
        ["customer_unique_id", "order_month", "first_order_in_month", "prev_order_date", "days_since_prev", "user_type"]
    ].copy()
    classified_out.to_csv(SNAPSHOTS / "h005_customer_month_classified.csv", index=False, encoding="utf-8")
    print(f"   saved h005_customer_month_classified.csv: {len(classified_out):,} rows\n")

    # 5) monthly aggregation
    monthly = aggregate_monthly(classified)
    monthly.to_csv(SNAPSHOTS / "h005_monthly_user_structure.csv", index=False, encoding="utf-8")
    print(f"   saved h005_monthly_user_structure.csv: {len(monthly)} months\n")
    print("   monthly four-class share preview:")
    print(monthly[["order_month", "total_buyers", "nurr_pct", "curr_pct", "rurr_pct", "surr_pct"]].to_string(index=False))
    print()

    # 6) period summary
    summary = compute_overall_summary(monthly)
    summary["timestamp"] = datetime.now().isoformat()
    with open(SNAPSHOTS / "h005_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"   💾 h005_overall_summary.json")
    print(f"   period four-class shares: nurr={summary['nurr']['pct']}% / curr={summary['curr']['pct']}% / rurr={summary['rurr']['pct']}% / surr={summary['surr']['pct']}%")
    print()

    # 7) trend chart data
    trend = build_trend_chart(monthly)
    trend.to_csv(SNAPSHOTS / "h005_trend_chart.csv", index=False, encoding="utf-8")
    print(f"   saved h005_trend_chart.csv: {len(trend)} months\n")

    # 8) industry benchmarks
    with open(SNAPSHOTS / "h005_industry_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(INDUSTRY_BENCHMARK, f, ensure_ascii=False, indent=2)
    print(f"   💾 h005_industry_benchmark.json\n")

    print("=" * 60)
    print("OK  H005 data generated")
    print("=" * 60)


if __name__ == "__main__":
    main()

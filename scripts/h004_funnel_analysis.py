#!/usr/bin/env python3
"""
H004: user-journey conversion funnel - 7-node order-lifecycle analysis
=================================================
Walks the full Olist order lifecycle (created -> approved -> shipped -> delivered ->
reviewed -> satisfied -> repurchased), computes each step conversion and locates the leak.

Node definitions:
  N1 created   order_purchase_timestamp IS NOT NULL
  N2 approved  order_approved_at IS NOT NULL
  N3 shipped   order_delivered_carrier_date IS NOT NULL
  N4 delivered order_delivered_customer_date IS NOT NULL
  N5 reviewed  review_creation_date IS NOT NULL
  N6 satisfied review_score >= 4
  N7 repeat    the same customer_unique_id has a later order

Usage:
    python3 scripts/h004_funnel_analysis.py

Output:
    data/snapshots/h004_funnel_wide_table.csv     order-level 7-node wide table
    data/snapshots/h004_funnel_overall.csv        overall order funnel
    data/snapshots/h004_funnel_by_state.csv       broken down by state
    data/snapshots/h004_funnel_by_category.csv    broken down by category
    data/snapshots/h004_dropoff_clients.csv       drop-off customer samples per stage
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
    "connect_timeout": 30,
}

# industry benchmarks (healthy e-commerce)
INDUSTRY_BENCHMARK = {
    "1_2": {"min": 99.0, "label": "Approval rate", "source": "E-commerce Standard"},
    "2_3": {"min": 95.0, "label": "Shipment start rate", "source": "E-commerce Standard"},
    "3_4": {"min": 95.0, "label": "Delivery rate", "source": "E-commerce Standard"},
    "4_5": {"min": 50.0, "label": "Review rate", "source": "Trustpilot Industry 50-70%"},
    "5_6": {"min": 70.0, "label": "Satisfaction rate (>=4)", "source": "Healthy E-commerce 70-80%"},
    "6_7": {"min": 30.0, "label": "Repurchase rate", "source": "Shopify Plus 2024"},
}


def fetch_data(conn) -> pd.DataFrame:
    """Fetch orders + customers + reviews + products (products only for the category split)."""
    print("Fetching order data ...")

    # order-level data (includes customer_unique_id and review)
    query = """
    SELECT
        o.order_id,
        o.customer_id,
        c.customer_unique_id,
        c.customer_state,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        r.review_score,
        r.review_creation_date
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    """
    df = pd.read_sql(query, conn)

    # parse timestamp columns
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_creation_date",
    ]:
        df[col] = pd.to_datetime(df[col])

    print(f"   OK  {len(df):,} orders fetched")

    # order -> category mapping (for the category split)
    print("Fetching order-category mapping ...")
    category_query = """
    SELECT
        oi.order_id,
        COALESCE(
            TRIM(BOTH '﻿' FROM p.product_category_name),
            'unknown'
        ) AS product_category_name
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    """
    df_cat = pd.read_sql(category_query, conn)
    print(f"   OK  {len(df_cat):,} order-category rows fetched")

    # merge: keep the first category of each order
    df_cat_first = df_cat.groupby("order_id")["product_category_name"].first().reset_index()
    df = df.merge(df_cat_first, on="order_id", how="left")
    df["product_category_name"] = df["product_category_name"].fillna("unknown")

    return df


def build_funnel_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the order-level 7-node wide table and the repeat flag."""
    print("Building the order-level 7-node wide table ...")

    # nodes 1-6
    df["n1_created"] = 1  # every order was created
    df["n2_approved"] = df["order_approved_at"].notna().astype(int)
    df["n3_shipped"] = df["order_delivered_carrier_date"].notna().astype(int)
    df["n4_delivered"] = df["order_delivered_customer_date"].notna().astype(int)
    df["n5_reviewed"] = (df["review_creation_date"].notna() & (df["order_status"] == "delivered")).astype(int)
    df["n6_satisfied"] = ((df["review_score"] >= 4) & (df["review_score"].notna())).astype(int)

    # node 7: repeat flag (the same customer_unique_id has an order after this one)
    # detected with groupby + shift
    df_sorted = df.sort_values(["customer_unique_id", "order_purchase_timestamp"]).copy()
    df_sorted["next_order_ts"] = df_sorted.groupby("customer_unique_id")["order_purchase_timestamp"].shift(-1)
    df_sorted["n7_repeat"] = df_sorted["next_order_ts"].notna().astype(int)

    # strict funnel flags (a node counts only if every upstream node was passed)
    df_sorted["reach_n2"] = df_sorted["n2_approved"]
    df_sorted["reach_n3"] = (df_sorted["n2_approved"] & df_sorted["n3_shipped"]).astype(int)
    df_sorted["reach_n4"] = (df_sorted["n2_approved"] & df_sorted["n3_shipped"] & df_sorted["n4_delivered"]).astype(int)
    df_sorted["reach_n5"] = (df_sorted["n2_approved"] & df_sorted["n3_shipped"] & df_sorted["n4_delivered"] & df_sorted["n5_reviewed"]).astype(int)
    df_sorted["reach_n6"] = (df_sorted["n2_approved"] & df_sorted["n3_shipped"] & df_sorted["n4_delivered"] & df_sorted["n5_reviewed"] & df_sorted["n6_satisfied"]).astype(int)
    df_sorted["reach_n7"] = (df_sorted["n2_approved"] & df_sorted["n3_shipped"] & df_sorted["n4_delivered"] & df_sorted["n5_reviewed"] & df_sorted["n6_satisfied"] & df_sorted["n7_repeat"]).astype(int)

    wide = df_sorted[[
        "order_id",
        "customer_unique_id",
        "customer_state",
        "product_category_name",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "review_creation_date",
        "review_score",
        "n1_created",
        "n2_approved",
        "n3_shipped",
        "n4_delivered",
        "n5_reviewed",
        "n6_satisfied",
        "n7_repeat",
        "reach_n2",
        "reach_n3",
        "reach_n4",
        "reach_n5",
        "reach_n6",
        "reach_n7",
    ]].copy()

    print(f"   OK  7-node wide table built: {len(wide):,} rows")
    return wide


def compute_funnel(wide: pd.DataFrame, group_col: str = None, top_n: int = None) -> pd.DataFrame:
    """Compute the funnel (6 step conversions)."""
    if group_col is None:
        # overall funnel
        node_counts = {
            "N1_created": int(wide["n1_created"].sum()),
            "N2_approved": int(wide["reach_n2"].sum()),
            "N3_shipped": int(wide["reach_n3"].sum()),
            "N4_delivered": int(wide["reach_n4"].sum()),
            "N5_reviewed": int(wide["reach_n5"].sum()),
            "N6_satisfied": int(wide["reach_n6"].sum()),
            "N7_repeat": int(wide["reach_n7"].sum()),
        }
        rows = [{"group": "all", "node": k, "count": v} for k, v in node_counts.items()]
    else:
        # grouped funnel
        if top_n:
            top_values = wide[group_col].value_counts().head(top_n).index.tolist()
            wide = wide[wide[group_col].isin(top_values)]
        grouped = wide.groupby(group_col).agg(
            n1=("n1_created", "sum"),
            n2=("reach_n2", "sum"),
            n3=("reach_n3", "sum"),
            n4=("reach_n4", "sum"),
            n5=("reach_n5", "sum"),
            n6=("reach_n6", "sum"),
            n7=("reach_n7", "sum"),
        ).reset_index()
        # reshape to long format
        rows = []
        for _, r in grouped.iterrows():
            for i, node in enumerate(["N1_created", "N2_approved", "N3_shipped", "N4_delivered", "N5_reviewed", "N6_satisfied", "N7_repeat"], start=1):
                rows.append({
                    "group": r[group_col],
                    "node": node,
                    "count": int(r[f"n{i}"]),
                })

    funnel = pd.DataFrame(rows)

    # step conversion rates
    funnel["step_conversion"] = 0.0
    funnel["cumulative_conversion"] = 0.0
    for grp in funnel["group"].unique():
        mask = funnel["group"] == grp
        sub = funnel[mask].sort_values("node", key=lambda x: x.map({n: i for i, n in enumerate(["N1_created", "N2_approved", "N3_shipped", "N4_delivered", "N5_reviewed", "N6_satisfied", "N7_repeat"])})).reset_index(drop=True)
        for i in range(1, len(sub)):
            if sub.loc[i - 1, "count"] > 0:
                sub.loc[i, "step_conversion"] = round(sub.loc[i, "count"] / sub.loc[i - 1, "count"] * 100, 2)
        if sub.loc[0, "count"] > 0:
            for i in range(1, len(sub)):
                sub.loc[i, "cumulative_conversion"] = round(sub.loc[i, "count"] / sub.loc[0, "count"] * 100, 2)
        funnel.loc[mask, "step_conversion"] = sub["step_conversion"].values
        funnel.loc[mask, "cumulative_conversion"] = sub["cumulative_conversion"].values

    return funnel


def identify_dropoffs(wide: pd.DataFrame) -> pd.DataFrame:
    """Identify customer samples lost at each stage."""
    print("Identifying drop-off customers per stage ...")

    dropoffs = []

    # leak 1->2: created but never approved
    drop_12 = wide[(wide["n1_created"] == 1) & (wide["n2_approved"] == 0)]
    dropoffs.append({
        "leak_stage": "1->2 created, not approved",
        "n_dropoff": len(drop_12),
        "n_start": len(wide),
        "dropoff_rate": round(len(drop_12) / len(wide) * 100, 2),
    })

    # leak 2->3: approved but never shipped
    drop_23 = wide[(wide["reach_n2"] == 1) & (wide["n3_shipped"] == 0)]
    dropoffs.append({
        "leak_stage": "2->3 approved, not shipped",
        "n_dropoff": len(drop_23),
        "n_start": int(wide["reach_n2"].sum()),
        "dropoff_rate": round(len(drop_23) / wide["reach_n2"].sum() * 100, 2) if wide["reach_n2"].sum() > 0 else 0,
    })

    # leak 3->4: shipped but never delivered
    drop_34 = wide[(wide["reach_n3"] == 1) & (wide["n4_delivered"] == 0)]
    dropoffs.append({
        "leak_stage": "3->4 shipped, not delivered",
        "n_dropoff": len(drop_34),
        "n_start": int(wide["reach_n3"].sum()),
        "dropoff_rate": round(len(drop_34) / wide["reach_n3"].sum() * 100, 2) if wide["reach_n3"].sum() > 0 else 0,
    })

    # leak 4->5: delivered but never reviewed
    drop_45 = wide[(wide["reach_n4"] == 1) & (wide["n5_reviewed"] == 0)]
    dropoffs.append({
        "leak_stage": "4->5 delivered, not reviewed",
        "n_dropoff": len(drop_45),
        "n_start": int(wide["reach_n4"].sum()),
        "dropoff_rate": round(len(drop_45) / wide["reach_n4"].sum() * 100, 2) if wide["reach_n4"].sum() > 0 else 0,
    })

    # leak 5->6: reviewed but dissatisfied
    drop_56 = wide[(wide["reach_n5"] == 1) & (wide["n6_satisfied"] == 0)]
    dropoffs.append({
        "leak_stage": "5->6 reviewed, not satisfied",
        "n_dropoff": len(drop_56),
        "n_start": int(wide["reach_n5"].sum()),
        "dropoff_rate": round(len(drop_56) / wide["reach_n5"].sum() * 100, 2) if wide["reach_n5"].sum() > 0 else 0,
    })

    # leak 6->7: satisfied but never repurchased
    drop_67 = wide[(wide["reach_n6"] == 1) & (wide["n7_repeat"] == 0)]
    dropoffs.append({
        "leak_stage": "6->7 satisfied, not repurchased",
        "n_dropoff": len(drop_67),
        "n_start": int(wide["reach_n6"].sum()),
        "dropoff_rate": round(len(drop_67) / wide["reach_n6"].sum() * 100, 2) if wide["reach_n6"].sum() > 0 else 0,
    })

    return pd.DataFrame(dropoffs)


def main():
    print("=" * 60)
    print("H004: user-journey conversion funnel (7-node order lifecycle)")
    print("=" * 60)

    # 1) database connection
    conn = mysql.connector.connect(**DB_CONFIG)
    print(f"OK  connected: {DB_CONFIG['host']}/{DB_CONFIG['database']}\n")

    # 2) fetch data
    df = fetch_data(conn)
    conn.close()

    # 3) build the wide table
    wide = build_funnel_wide_table(df)
    wide.to_csv(SNAPSHOTS / "h004_funnel_wide_table.csv", index=False, encoding="utf-8")
    print(f"   saved h004_funnel_wide_table.csv: {len(wide):,} rows\n")

    # 4) overall funnel
    print("Computing the overall funnel ...")
    funnel_overall = compute_funnel(wide, group_col=None)
    funnel_overall.to_csv(SNAPSHOTS / "h004_funnel_overall.csv", index=False, encoding="utf-8")
    print(funnel_overall.to_string(index=False))
    print(f"   💾 h004_funnel_overall.csv\n")

    # 5) split by state (top 10)
    print("Splitting the funnel by state (top 10) ...")
    funnel_state = compute_funnel(wide, group_col="customer_state", top_n=10)
    funnel_state.to_csv(SNAPSHOTS / "h004_funnel_by_state.csv", index=False, encoding="utf-8")
    print(f"   saved h004_funnel_by_state.csv: {funnel_state['group'].nunique()} states x 7 nodes = {len(funnel_state)} rows\n")

    # 6) split by category (top 10)
    print("Splitting the funnel by category (top 10) ...")
    funnel_cat = compute_funnel(wide, group_col="product_category_name", top_n=10)
    funnel_cat.to_csv(SNAPSHOTS / "h004_funnel_by_category.csv", index=False, encoding="utf-8")
    print(f"   saved h004_funnel_by_category.csv: {funnel_cat['group'].nunique()} categories x 7 nodes = {len(funnel_cat)} rows\n")

    # 7) identify drop-off points
    print("Identifying drop-offs per stage ...")
    dropoffs = identify_dropoffs(wide)
    dropoffs.to_csv(SNAPSHOTS / "h004_dropoff_clients.csv", index=False, encoding="utf-8")
    print(dropoffs.to_string(index=False))
    print(f"   💾 h004_dropoff_clients.csv\n")

    # 8) industry benchmarks
    with open(SNAPSHOTS / "h004_industry_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(INDUSTRY_BENCHMARK, f, ensure_ascii=False, indent=2)
    print(f"   💾 h004_industry_benchmark.json\n")

    print("=" * 60)
    print("OK  H004 data generated")
    print("=" * 60)


if __name__ == "__main__":
    main()

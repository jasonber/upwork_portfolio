#!/usr/bin/env python3
"""
H002-B/C/D: Olist data-feasibility validation
====================================
Validates:
  H002-B: delivery delay -> low review score (is the causal chain plausible?)
  H002-C: the data can support churn modelling (is a label definable?)
  H002-D: review text is usable for NLP analysis

Usage:
    python3 scripts/h002_data_validation.py

Requires:
    pip install mysql-connector-python pandas

Environment variables (optional; defaults in DB_CONFIG):
    SQLPUB_HOST, SQLPUB_PORT, SQLPUB_USER, SQLPUB_PASSWORD, SQLPUB_DATABASE
"""

from __future__ import annotations

import os
import json
import mysql.connector
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
SNAPSHOTS = BASE / "data" / "snapshots"
SNAPSHOTS.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("SQLPUB_HOST", "mysql6.sqlpub.com"),
    "port": int(os.getenv("SQLPUB_PORT", "3311")),
    "user": os.getenv("SQLPUB_USER", "zz0008"),
    "password": os.getenv("SQLPUB_PASSWORD", "yjom5GVTLAzPC3O6"),
    "database": os.getenv("SQLPUB_DATABASE", "zz_free"),
    "connect_timeout": 10,
}


def get_conn():
    return mysql.connector.connect(**DB_CONFIG)


# ============================================================
# H002-B: delivery delay -> satisfaction
# ============================================================
def validate_h003_b(conn) -> dict:
    """Test H002-B: delivery delay depresses satisfaction."""

    # B1: group by delivery status (late / on time / early)
    q_b1 = """
    SELECT
        CASE
            WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) > 0 THEN 'Late'
            WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) = 0 THEN 'OnTime'
            ELSE 'Early'
        END AS delivery_status,
        COUNT(*) AS n,
        ROUND(AVG(r.review_score), 3) AS avg_score,
        ROUND(STD(r.review_score), 3) AS std_score,
        ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS low_score_pct,
        ROUND(MIN(r.review_score), 1) AS min_score,
        ROUND(MAX(r.review_score), 1) AS max_score
    FROM orders o
    JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND r.review_score IS NOT NULL
    GROUP BY delivery_status
    ORDER BY avg_score
    """
    df_b1 = pd.read_sql(q_b1, conn)
    df_b1.to_csv(SNAPSHOTS / "h003_b1_delivery_status.csv", index=False, encoding="utf-8")
    print(f"  ✅ b1_delivery_status: {len(df_b1)} rows")

    # B2: category x delivery performance
    q_b2 = """
    SELECT
        p.product_category_name,
        COUNT(*) AS n,
        ROUND(AVG(TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp, o.order_delivered_customer_date))/24, 1) AS avg_days,
        ROUND(AVG(r.review_score), 2) AS avg_review,
        ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END)*100.0/COUNT(*), 1) AS low_pct
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY p.product_category_name
    HAVING n >= 200
    ORDER BY avg_days DESC
    LIMIT 20
    """
    df_b2 = pd.read_sql(q_b2, conn)
    df_b2.to_csv(SNAPSHOTS / "h003_b2_category_timing.csv", index=False, encoding="utf-8")
    print(f"  ✅ b2_category_timing: {len(df_b2)} rows")

    # B3: delivery performance by state
    q_b3 = """
    SELECT
        c.customer_state,
        COUNT(*) AS n,
        ROUND(AVG(TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp, o.order_delivered_customer_date))/24, 1) AS avg_days,
        ROUND(AVG(r.review_score), 2) AS avg_review,
        ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END)*100.0/COUNT(*), 1) AS low_pct
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_state
    HAVING n >= 100
    ORDER BY avg_days DESC
    """
    df_b3 = pd.read_sql(q_b3, conn)
    df_b3.to_csv(SNAPSHOTS / "h003_b3_state_delivery.csv", index=False, encoding="utf-8")
    print(f"  ✅ b3_state_delivery: {len(df_b3)} rows")

    # B4: delay-day buckets vs review score
    q_b4 = """
    SELECT
        CASE
            WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) < 0 THEN 'Early'
            WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) <= 24 THEN 'OnTime_±1day'
            WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) <= 72 THEN 'Late_1-3days'
            WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) <= 168 THEN 'Late_3-7days'
            ELSE 'Late_7+days'
        END AS delay_bucket,
        COUNT(*) AS n,
        ROUND(AVG(r.review_score), 3) AS avg_score,
        ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS low_pct
    FROM orders o
    JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY delay_bucket
    ORDER BY MIN(TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date))
    """
    df_b4 = pd.read_sql(q_b4, conn)
    df_b4.to_csv(SNAPSHOTS / "h003_b4_delay_bucket.csv", index=False, encoding="utf-8")
    print(f"  ✅ b4_delay_bucket: {len(df_b4)} rows")

    return {"b1": df_b1.to_dict(orient="records"), "b2": df_b2.to_dict(orient="records"),
            "b3": df_b3.to_dict(orient="records"), "b4": df_b4.to_dict(orient="records")}


# ============================================================
# H002-C: feasibility of churn modelling
# ============================================================
def validate_h003_c(conn) -> dict:
    """Test H002-C: the data supports churn modelling."""

    # C1: distribution of orders per customer
    q_c1 = """
    SELECT customer_unique_id, COUNT(DISTINCT o.order_id) AS order_cnt
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY customer_unique_id
    """
    df_c1 = pd.read_sql(q_c1, conn)

    # C2: RFM feature table
    q_c2 = """
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT o.order_id) AS order_count,
        ROUND(SUM(oi.price), 2) AS total_spent,
        DATEDIFF('2018-10-31', MAX(o.order_purchase_timestamp)) AS days_since_last,
        ROUND(AVG(r.review_score), 2) AS avg_review,
        COUNT(DISTINCT p.product_category_name) AS unique_categories
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
    """
    df_c2 = pd.read_sql(q_c2, conn)

    # build the churn label and measure feature coverage
    df_c2["churn_label"] = (df_c2["days_since_last"] > 180).astype(int)
    df_c2["high_value"] = (df_c2["total_spent"] > df_c2["total_spent"].quantile(0.75)).astype(int)

    feature_coverage = {
        col: round(float(df_c2[col].notna().mean() * 100), 2)
        for col in ["total_spent", "order_count", "days_since_last", "avg_review", "unique_categories"]
    }

    df_c2.to_csv(SNAPSHOTS / "h003_c_rfm_features.csv", index=False, encoding="utf-8")
    print(f"  ✅ c_rfm_features: {len(df_c2)} rows, churn rate = {df_c2['churn_label'].mean()*100:.1f}%")

    repeat_rate = round(float((df_c1["order_cnt"] > 1).mean() * 100), 2)

    return {
        "n_customers": len(df_c2),
        "feature_coverage": feature_coverage,
        "churn_label_pct": round(float(df_c2["churn_label"].mean() * 100), 2),
        "repeat_rate": repeat_rate,
    }


# ============================================================
# H002-D: feasibility of NLP on review text
# ============================================================
def validate_h003_d(conn) -> dict:
    """Test H002-D: review text is usable for NLP analysis."""

    # D1: share of non-null text fields
    q_d1 = """
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN review_comment_message IS NOT NULL AND review_comment_message != '' THEN 1 ELSE 0 END) AS has_text,
        SUM(CASE WHEN review_comment_title IS NOT NULL AND review_comment_title != '' THEN 1 ELSE 0 END) AS has_title,
        SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END) AS low_score,
        SUM(CASE WHEN review_score >= 4 THEN 1 ELSE 0 END) AS high_score
    FROM order_reviews
    """
    row = pd.read_sql(q_d1, conn).iloc[0]

    # D2: low-score review examples
    q_d2 = """
    SELECT review_comment_message, review_score, order_id
    FROM order_reviews
    WHERE review_score <= 2
      AND review_comment_message IS NOT NULL
      AND review_comment_message != ''
    LIMIT 20
    """
    df_d2 = pd.read_sql(q_d2, conn)

    # D3: high-score review examples
    q_d3 = """
    SELECT review_comment_message, review_score, order_id
    FROM order_reviews
    WHERE review_score >= 4
      AND review_comment_message IS NOT NULL
      AND review_comment_message != ''
    LIMIT 20
    """
    df_d3 = pd.read_sql(q_d3, conn)

    df_d2.to_csv(SNAPSHOTS / "h003_d_negative_samples.csv", index=False, encoding="utf-8")
    df_d3.to_csv(SNAPSHOTS / "h003_d_positive_samples.csv", index=False, encoding="utf-8")
    print(f"  ✅ d_negative_samples: {len(df_d2)} rows")
    print(f"  ✅ d_positive_samples: {len(df_d3)} rows")

    total = int(row["total"])
    return {
        "total_reviews": total,
        "has_text_pct": round(float(row["has_text"]) / total * 100, 2),
        "has_title_pct": round(float(row["has_title"]) / total * 100, 2),
        "low_score_pct": round(float(row["low_score"]) / total * 100, 2),
        "high_score_pct": round(float(row["high_score"]) / total * 100, 2),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("H002: Olist data-feasibility validation")
    print("=" * 60)

    conn = get_conn()
    print(f"OK  connected: {DB_CONFIG['host']}/{DB_CONFIG['database']}")

    print("\n--- H002-B: delivery delay -> satisfaction ---")
    b_results = validate_h003_b(conn)

    print("\n--- H002-C: churn modelling feasibility ---")
    c_results = validate_h003_c(conn)

    print("\n--- H002-D: review-text NLP feasibility ---")
    d_results = validate_h003_d(conn)

    conn.close()

    # write the summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "h003_b": b_results,
        "h003_c": c_results,
        "h003_d": d_results,
    }
    summary_path = SNAPSHOTS / "h002_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nOK  summary report: {summary_path}")
    print("Done.")

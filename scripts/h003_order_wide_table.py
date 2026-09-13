#!/usr/bin/env python3
"""
H003: Olist order-level wide table (ETL)
================================
Joins several Olist tables into an analysis-ready wide table used by the later
hypothesis tests and by the churn model.

Usage:
    python3 scripts/h003_order_wide_table.py

Output:
    data/snapshots/h003_order_wide_table.csv   (~99K rows)
"""

from __future__ import annotations

import os
import mysql.connector
import pandas as pd
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
    "connect_timeout": 15,
}

QUERY = """
SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    p.product_category_name,
    oi.order_item_id,
    oi.price AS item_price,
    oi.quantity AS item_quantity,
    oi.seller_id,
    r.review_score,
    r.review_comment_title,
    r.review_comment_message,
    r.review_creation_date,
    TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) / 24 AS delivery_delay_days,
    TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp, o.order_delivered_customer_date) / 24 AS actual_delivery_days,
    CASE
        WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) > 0 THEN 'Late'
        WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date, o.order_delivered_customer_date) = 0 THEN 'OnTime'
        ELSE 'Early'
    END AS delivery_status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
"""


def main():
    print("H003: building the order wide table...")
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(QUERY, conn)
    conn.close()

    out_path = SNAPSHOTS / "h003_order_wide_table.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"OK  wide table saved: {out_path}")
    print(f"   rows: {len(df):,} | columns: {len(df.columns)}")
    print(f"   columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

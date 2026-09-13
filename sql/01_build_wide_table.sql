-- ============================================================================
-- 01_build_wide_table.sql
-- Olist churn & retention analysis — order-level wide table + customer-level RFM
-- ----------------------------------------------------------------------------
-- Dialect : MySQL 8.0+ (originally run against mysql6.sqlpub.com; any MySQL 8
--           instance works after re-importing the 9 Olist CSVs)
-- Sources : h003_order_wide_table.py · h009_causal_pipeline.py (prepare_data)
--           · h011_caliber_sensitivity.py (recency / churn label)
--
-- Grain   :
--   step 1  VIEW  v_order_wide        -> one row per order item (≈112K)
--   step 2  TABLE customer_features   -> one row per customer_unique_id (93,358)
--
-- Reference date: 2018-10-31 (last purchase in the snapshot) — every
--                 recency / churn label is measured against it.
-- ============================================================================


-- ---------------------------------------------------------------------------
-- STEP 1 · Order-level wide table  (source: h003_order_wide_table.py, QUERY)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_order_wide AS
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
    oi.price                        AS item_price,
    oi.quantity                     AS item_quantity,
    oi.seller_id,
    r.review_score,
    r.review_comment_title,
    r.review_comment_message,
    r.review_creation_date,
    -- positive = delivered after the promised date
    TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date,
                        o.order_delivered_customer_date) / 24 AS delivery_delay_days,
    TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp,
                        o.order_delivered_customer_date) / 24 AS actual_delivery_days,
    CASE
        WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date,
                                o.order_delivered_customer_date) > 0 THEN 'Late'
        WHEN TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date,
                                o.order_delivered_customer_date) = 0 THEN 'OnTime'
        ELSE 'Early'
    END                             AS delivery_status
FROM orders          o
JOIN customers       c  ON o.customer_id = c.customer_id
JOIN order_items     oi ON o.order_id    = oi.order_id
JOIN products        p  ON oi.product_id = p.product_id
LEFT JOIN order_reviews r ON o.order_id  = r.order_id
WHERE o.order_status = 'delivered';


-- ---------------------------------------------------------------------------
-- STEP 2 · Customer-level feature table
--   (source: h009_causal_pipeline.py · prepare_data, plus recency from h011)
--   delivered orders only, purchase timestamp required
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS customer_features;
CREATE TABLE customer_features AS
SELECT
    c.customer_unique_id,
    c.customer_state,
    COUNT(DISTINCT o.order_id)                                 AS order_count,
    ROUND(SUM(oi.price), 2)                                    AS total_spent,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date
                        > o.order_estimated_delivery_date
                   THEN 1 ELSE 0 END), 4)                      AS late_rate,
    ROUND(AVG(TIMESTAMPDIFF(HOUR, o.order_estimated_delivery_date,
                                 o.order_delivered_customer_date)) / 24.0, 2)
                                                               AS avg_delay_days,
    ROUND(AVG(r.review_score), 3)                              AS avg_review,
    MAX(o.order_purchase_timestamp)                            AS last_order_ts,
    MIN(o.order_purchase_timestamp)                            AS first_order_ts,
    -- recency against the snapshot reference date
    DATEDIFF('2018-10-31', MAX(o.order_purchase_timestamp))    AS days_since_last,
    -- MAIN churn label: dormant > 180 days (caliber defined in 03_*.sql)
    CASE WHEN DATEDIFF('2018-10-31', MAX(o.order_purchase_timestamp)) > 180
         THEN 1 ELSE 0 END                                     AS churn_label
FROM customers   c
JOIN orders      o  ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id    = oi.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp IS NOT NULL
GROUP BY c.customer_unique_id, c.customer_state;


-- ---------------------------------------------------------------------------
-- STEP 3 · First-touch category per order (used by the by-category funnel)
--   source: h004_funnel_analysis.py · fetch_data (category_query)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_order_primary_category AS
SELECT
    order_id,
    SUBSTRING_INDEX(
        GROUP_CONCAT(COALESCE(TRIM(LEADING '\ufeff' FROM p.product_category_name),
                              'unknown')
                     ORDER BY oi.order_item_id SEPARATOR '|'),
        '|', 1) AS product_category_name
FROM order_items oi
JOIN products    p ON oi.product_id = p.product_id
GROUP BY order_id;


-- ---------------------------------------------------------------------------
-- Sanity checks
-- ---------------------------------------------------------------------------
-- 1. customer count must equal the portfolio figure 93,358
--    SELECT COUNT(*) FROM customer_features;                       -- 93358
-- 2. main churn rate must equal 74.1 %
--    SELECT ROUND(AVG(churn_label) * 100, 2) FROM customer_features; -- 74.08
-- 3. recency span (h011: min 63 · p50 175 · p75 179 · max 366)
--    SELECT MIN(days_since_last), MAX(days_since_last) FROM customer_features;

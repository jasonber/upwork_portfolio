-- ============================================================================
-- 02_funnel_and_cohort.sql
-- 7-node order-lifecycle funnel · cohort retention matrix · time-window churn
-- ----------------------------------------------------------------------------
-- Dialect : MySQL 8.0+ (window functions required)
-- Sources : h004_funnel_analysis.py (build_funnel_wide_table, compute_funnel)
--           h006_growth_funnel_analysis.py (compute_cohort_retention,
--                                           compute_time_window_churn)
--
-- PREREQUISITE: run 01_build_wide_table.sql first — this file uses
--   v_order_primary_category. 03_churn_and_uplift.sql depends on
--   v_time_window_churn created at the end of this file.
--
-- Node definitions (identical to h004):
--   N1 created   order_purchase_timestamp IS NOT NULL
--   N2 approved  order_approved_at          IS NOT NULL
--   N3 shipped   order_delivered_carrier_date IS NOT NULL
--   N4 delivered order_delivered_customer_date IS NOT NULL
--   N5 reviewed  review_creation_date IS NOT NULL AND order_status='delivered'
--   N6 satisfied review_score >= 4
--   N7 repeat    same customer_unique_id has a later order
--
-- NOTE h004 reads ALL order statuses (no delivered filter) so that the
--      approval/shipping leakage is visible. Cohort section uses delivered only.
-- ============================================================================


-- ---------------------------------------------------------------------------
-- STEP 1 · Node flags + strict "reach" flags (customer-ordered window LEAD)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_funnel_wide AS
WITH node_flags AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        c.customer_state,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        r.review_creation_date,
        r.review_score,
        1                                                       AS n1_created,
        (o.order_approved_at            IS NOT NULL)            AS n2_approved,
        (o.order_delivered_carrier_date IS NOT NULL)            AS n3_shipped,
        (o.order_delivered_customer_date IS NOT NULL)           AS n4_delivered,
        (r.review_creation_date IS NOT NULL
            AND o.order_status = 'delivered')                   AS n5_reviewed,
        (r.review_score >= 4)                                   AS n6_satisfied
    FROM orders          o
    JOIN customers       c ON o.customer_id = c.customer_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
),
pairing AS (
    SELECT
        f.*,
        -- N7: is there a LATER order for the same buyer?
        LEAD(order_purchase_timestamp) OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp
        ) IS NOT NULL                                           AS n7_repeat
    FROM node_flags f
)
SELECT
    order_id,
    customer_unique_id,
    customer_state,
    order_status,
    order_purchase_timestamp,
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    review_creation_date,
    review_score,
    n1_created, n2_approved, n3_shipped, n4_delivered,
    n5_reviewed, n6_satisfied, n7_repeat,
    -- strict funnel: a node counts only when every upstream node was passed
    n2_approved                                             AS reach_n2,
    (n2_approved AND n3_shipped)                            AS reach_n3,
    (n2_approved AND n3_shipped AND n4_delivered)           AS reach_n4,
    (n2_approved AND n3_shipped AND n4_delivered
                 AND n5_reviewed)                           AS reach_n5,
    (n2_approved AND n3_shipped AND n4_delivered
                 AND n5_reviewed AND n6_satisfied)          AS reach_n6,
    (n2_approved AND n3_shipped AND n4_delivered
                 AND n5_reviewed AND n6_satisfied
                 AND n7_repeat)                             AS reach_n7
FROM pairing;


-- ---------------------------------------------------------------------------
-- STEP 2 · Overall funnel + step conversion
--   expected: N1 99,992 → N7 2,930 (cumulative 2.93 %)
-- ---------------------------------------------------------------------------
SELECT
    'N1_created'   AS node, 1 AS step, COUNT(*)                        AS reached FROM v_funnel_wide
UNION ALL SELECT 'N2_approved',  2, SUM(reach_n2)          FROM v_funnel_wide
UNION ALL SELECT 'N3_shipped',   3, SUM(reach_n3)          FROM v_funnel_wide
UNION ALL SELECT 'N4_delivered', 4, SUM(reach_n4)          FROM v_funnel_wide
UNION ALL SELECT 'N5_reviewed',  5, SUM(reach_n5)          FROM v_funnel_wide
UNION ALL SELECT 'N6_satisfied', 6, SUM(reach_n6)          FROM v_funnel_wide
UNION ALL SELECT 'N7_repeat',    7, SUM(reach_n7)          FROM v_funnel_wide
ORDER BY step;


-- ---------------------------------------------------------------------------
-- STEP 3 · Funnel broken down by state and by primary category
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_funnel_by_state AS
SELECT
    customer_state AS grp,
    COUNT(*)                    AS n1_created,
    SUM(reach_n2)               AS n2_approved,
    SUM(reach_n3)               AS n3_shipped,
    SUM(reach_n4)               AS n4_delivered,
    SUM(reach_n5)               AS n5_reviewed,
    SUM(reach_n6)               AS n6_satisfied,
    SUM(reach_n7)               AS n7_repeat,
    ROUND(SUM(reach_n6) / NULLIF(SUM(reach_n5), 0) * 100, 2)  AS pct_5_to_6,
    ROUND(SUM(reach_n7) / NULLIF(SUM(reach_n6), 0) * 100, 2)  AS pct_6_to_7
FROM v_funnel_wide
GROUP BY customer_state;

CREATE OR REPLACE VIEW v_funnel_by_category AS
SELECT
    t.product_category_name    AS grp,
    COUNT(*)                   AS n1_created,
    SUM(f.reach_n2)            AS n2_approved,
    SUM(f.reach_n3)            AS n3_shipped,
    SUM(f.reach_n4)            AS n4_delivered,
    SUM(f.reach_n5)            AS n5_reviewed,
    SUM(f.reach_n6)            AS n6_satisfied,
    SUM(f.reach_n7)            AS n7_repeat,
    ROUND(SUM(f.reach_n6) / NULLIF(SUM(f.reach_n5), 0) * 100, 2)  AS pct_5_to_6,
    ROUND(SUM(f.reach_n7) / NULLIF(SUM(f.reach_n6), 0) * 100, 2)  AS pct_6_to_7
FROM v_funnel_wide f
JOIN v_order_primary_category t ON f.order_id = t.order_id
GROUP BY t.product_category_name;


-- ---------------------------------------------------------------------------
-- STEP 4 · Cohort retention matrix (month of acquisition × months since)
--   source: h006 · compute_cohort_retention — any order in month M+N counts
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_cohort_retention AS
WITH delivered AS (
    SELECT DISTINCT
        c.customer_unique_id,
        DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01') AS order_month
    FROM orders    o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
),
cohort AS (   -- each buyer's acquisition month = month of first order
    SELECT customer_unique_id, MIN(order_month) AS cohort_month
    FROM delivered
    GROUP BY customer_unique_id
),
activity AS (
    SELECT
        d.customer_unique_id,
        d.order_month,
        k.cohort_month,
        TIMESTAMPDIFF(MONTH, k.cohort_month, d.order_month) AS months_since_acquisition
    FROM delivered d
    JOIN cohort    k ON d.customer_unique_id = k.customer_unique_id
),
sized AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_unique_id) AS cohort_size
    FROM cohort
    GROUP BY cohort_month
)
SELECT
    a.cohort_month,
    a.months_since_acquisition,
    COUNT(DISTINCT a.customer_unique_id)                       AS active_customers,
    s.cohort_size,
    ROUND(COUNT(DISTINCT a.customer_unique_id)
          / s.cohort_size * 100, 2)                            AS retention_rate
FROM activity a
JOIN sized    s ON a.cohort_month = s.cohort_month
GROUP BY a.cohort_month, a.months_since_acquisition, s.cohort_size;

-- pivot for the dashboard heat-map:
--   SELECT cohort_month,
--          MAX(CASE WHEN months_since_acquisition = 0 THEN retention_rate END) AS m0,
--          MAX(CASE WHEN months_since_acquisition = 1 THEN retention_rate END) AS m1,
--          ... FROM v_cohort_retention GROUP BY cohort_month;


-- ---------------------------------------------------------------------------
-- STEP 5 · Time-window churn curve (T+30/60/90/180/365)
--   source: h006 · compute_time_window_churn — "no second order within N days
--   of the first order". This is a DIFFERENT caliber from 03_*.sql (dormancy),
--   which is exactly why the deck shows both side by side.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_time_window_churn AS
WITH ordered AS (
    SELECT
        c.customer_unique_id,
        o.order_purchase_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp
        ) AS order_idx
    FROM orders    o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
),
firsts AS (
    SELECT customer_unique_id,
           MIN(order_purchase_timestamp) AS first_order_ts
    FROM ordered
    GROUP BY customer_unique_id
),
seconds AS (
    SELECT f.customer_unique_id,
           f.first_order_ts,
           MIN(o.order_purchase_timestamp) AS second_order_ts
    FROM firsts f
    LEFT JOIN ordered o
           ON o.customer_unique_id = f.customer_unique_id
          AND o.order_idx = 2
    GROUP BY f.customer_unique_id, f.first_order_ts
)
SELECT 30 AS window_days,
       COUNT(*) AS n_total_customers,
       SUM(second_order_ts IS NOT NULL
           AND DATEDIFF(second_order_ts, first_order_ts) <= 30)  AS n_retained,
       ROUND(SUM(second_order_ts IS NOT NULL
           AND DATEDIFF(second_order_ts, first_order_ts) <= 30)
             / COUNT(*) * 100, 2)                                AS retention_rate
FROM seconds
UNION ALL SELECT 60,  COUNT(*),
       SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 60),
       ROUND(SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 60) / COUNT(*) * 100, 2)
FROM seconds
UNION ALL SELECT 90,  COUNT(*),
       SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 90),
       ROUND(SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 90) / COUNT(*) * 100, 2)
FROM seconds
UNION ALL SELECT 180, COUNT(*),
       SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 180),
       ROUND(SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 180) / COUNT(*) * 100, 2)
FROM seconds
UNION ALL SELECT 365, COUNT(*),
       SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 365),
       ROUND(SUM(second_order_ts IS NOT NULL AND DATEDIFF(second_order_ts, first_order_ts) <= 365) / COUNT(*) * 100, 2)
FROM seconds
ORDER BY window_days;


-- ---------------------------------------------------------------------------
-- STEP 6 · Leakage attribution — the two break points the deck highlights
--   N5→N6 (satisfaction collapse) and N6→N7 (repurchase collapse)
-- ---------------------------------------------------------------------------
SELECT
    ROUND(SUM(reach_n2) / COUNT(*)                              * 100, 2) AS pct_1_2,
    ROUND(SUM(reach_n3) / SUM(reach_n2)                         * 100, 2) AS pct_2_3,
    ROUND(SUM(reach_n4) / SUM(reach_n3)                         * 100, 2) AS pct_3_4,
    ROUND(SUM(reach_n5) / SUM(reach_n4)                         * 100, 2) AS pct_4_5,
    ROUND(SUM(reach_n6) / SUM(reach_n5)                         * 100, 2) AS pct_5_6,
    ROUND(SUM(reach_n7) / SUM(reach_n6)                         * 100, 2) AS pct_6_7
FROM v_funnel_wide;

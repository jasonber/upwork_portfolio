-- ============================================================================
-- 03_churn_and_uplift.sql
-- Churn caliber definition & sensitivity · repeat-purchase split ·
-- intervention targeting (uplift / decile) · strategy ROI
-- ----------------------------------------------------------------------------
-- Dialect : MySQL 8.0+
-- Sources : h011_caliber_sensitivity.py · h009_causal_pipeline.py
--           (h009_a_causal_identification, h009_c_step, h009_e_f_cate_uplift)
--
-- PREREQUISITE: run 01_build_wide_table.sql and 02_funnel_and_cohort.sql first.
--   this file uses customer_features (01) and v_time_window_churn (02).
--
-- Why this file exists
--   "74 % churn" is meaningless until the caliber is stated. This script turns
--   the implicit assumption into an explicit, testable definition and shows
--   what changes when the window moves. The deck's page "Churn caliber &
--   sensitivity" is generated from exactly these numbers.
-- ============================================================================


-- ---------------------------------------------------------------------------
-- STEP 1 · Dormancy caliber sensitivity — 30/60/90/120/180/270/365 days
--   source: h011 · sensitivity table. Main caliber = 180 days.
--   Reconciliation assertion (must hold): 180-day dormancy == churn_label mean
-- ---------------------------------------------------------------------------
SELECT
    w.window_days,
    (w.window_days = 180)                                        AS is_main,
    ROUND(AVG(f.days_since_last > w.window_days) * 100, 2)        AS churn_rate,
    ROUND(AVG(f.days_since_last <= w.window_days) * 100, 2)       AS retained_rate,
    SUM(f.days_since_last > w.window_days)                        AS n_churned,
    SUM(f.days_since_last <= w.window_days)                       AS n_retained,
    ROUND(SUM(CASE WHEN f.days_since_last > w.window_days
                   THEN f.total_spent ELSE 0 END) / 1e6, 3)       AS churned_revenue_m
FROM customer_features f
CROSS JOIN (
    SELECT 30 AS window_days UNION ALL SELECT 60 UNION ALL SELECT 90
    UNION ALL SELECT 120 UNION ALL SELECT 180
    UNION ALL SELECT 270 UNION ALL SELECT 365
) w
GROUP BY w.window_days
ORDER BY w.window_days;

-- Reconciliation — all three numbers must match (h011 assert):
--   SELECT ROUND(AVG(days_since_last > 180) * 100, 4) FROM customer_features;  -- 74.0761
--   SELECT ROUND(AVG(churn_label) * 100, 4)           FROM customer_features;  -- 74.0761
--   SELECT ROUND(AVG(Actual_Churn) * 100, 4)          FROM Customer_Churn_Scores; -- 74.08


-- ---------------------------------------------------------------------------
-- STEP 2 · Inter-order gaps (needed below and by STEP 3)
--   source: h011 · stage-by-stage inter-order gaps — gap_days = 0 means a same-day split order
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_interorder_gaps AS
SELECT
    customer_unique_id,
    order_purchase_timestamp,
    order_idx,
    DATEDIFF(
        order_purchase_timestamp,
        LAG(order_purchase_timestamp) OVER (
            PARTITION BY customer_unique_id ORDER BY order_purchase_timestamp)
    ) AS gap_days
FROM (
    SELECT
        c.customer_unique_id,
        o.order_purchase_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp) AS order_idx
    FROM orders    o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
) x;


-- ---------------------------------------------------------------------------
-- STEP 3 · Three competing calibers (why one number is not enough)
--   (a) dormancy     > 180 days since last purchase        -> 74.1 %
--   (b) repurchase   no 2nd order within 180 days          -> 97.4 %
--   (c) first-gap    median inter-order gap                -> 74 days
-- ---------------------------------------------------------------------------
SELECT 'dormancy >180d'   AS caliber,
       ROUND(AVG(days_since_last > 180) * 100, 1) AS value_pct, 'main caliber' AS note
FROM customer_features
UNION ALL
SELECT 'no 2nd order in 180d',
       ROUND(AVG(n_retained = 0) * 100, 1),
       'acquisition view'
FROM v_time_window_churn WHERE window_days = 180
UNION ALL
SELECT 'median inter-order gap',
       ROUND((SELECT AVG(gap_days) FROM (
                  SELECT gap_days,
                         ROW_NUMBER() OVER (ORDER BY gap_days) rn,
                         COUNT(*) OVER () cnt
                  FROM v_interorder_gaps WHERE gap_days > 0
              ) t WHERE t.rn IN (FLOOR((cnt + 1) / 2), CEIL((cnt + 1) / 2))), 0),
       'proxy: median in SQL, exact value 74 days';


-- ---------------------------------------------------------------------------
-- STEP 4 · Repeat-purchase split — same-day multi-order contamination
--   source: h011 · same-day split-order correction. 29 % of "repeat" buyers are single-session
--   split orders; the honest cross-period repeat rate is 2.10 %.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_repeat_split AS
WITH rep AS (
    SELECT f.customer_unique_id,
           f.total_spent,
           COALESCE(g.max_gap, 0) AS max_gap
    FROM customer_features f
    LEFT JOIN (
        SELECT customer_unique_id, MAX(gap_days) AS max_gap
        FROM v_interorder_gaps
        GROUP BY customer_unique_id
    ) g ON g.customer_unique_id = f.customer_unique_id
    WHERE f.order_count >= 2
)
SELECT
    COUNT(*)                                                                  AS n_repeat_as_reported,
    SUM(max_gap = 0)                                                          AS n_same_day_only,
    ROUND(AVG(max_gap = 0) * 100, 2)                                          AS same_day_share_of_repeat,
    SUM(max_gap > 0)                                                          AS n_true_cross_day,
    ROUND(AVG(max_gap > 0) * 100, 2)                                          AS true_repeat_rate
FROM rep;

-- value multiple (repeat buyer average spend ÷ one-time buyer average spend)
-- reported 2.01x, honest cross-period multiple 1.55x:
--   SELECT ROUND(AVG(total_spent), 2) FROM customer_features WHERE order_count = 1;
--   SELECT ROUND(AVG(total_spent), 2) FROM customer_features WHERE order_count >= 2;


-- ---------------------------------------------------------------------------
-- STEP 5 · Stage-by-stage repurchase interval (excl. same-day splits)
--   source: h011 · stage-by-stage inter-order gaps — p75 ≈ 179 days is WHY 180 was chosen
-- ---------------------------------------------------------------------------
SELECT
    CONCAT(order_idx, '->', order_idx + 1)          AS stage,
    COUNT(*)                                        AS n_all,
    SUM(gap_days = 0)                               AS n_same_day,
    SUM(gap_days > 0)                               AS n_valid,
    ROUND(AVG(CASE WHEN gap_days > 0 THEN gap_days END), 1) AS mean_days,
    ROUND(MAX(CASE WHEN gap_days > 0 THEN gap_days END), 0) AS max_days,
    ROUND(AVG(CASE WHEN gap_days > 0 AND gap_days <= 30 THEN 1 ELSE 0 END) * 100, 1) AS le30_pct,
    ROUND(AVG(CASE WHEN gap_days > 90 THEN 1 ELSE 0 END) * 100, 1)                   AS gt90_pct
FROM v_interorder_gaps
WHERE order_idx IN (1, 2, 3)
GROUP BY order_idx
ORDER BY order_idx;


-- ---------------------------------------------------------------------------
-- STEP 6 · Delivery lag → churn / satisfaction (PSM-adjacent descriptives)
--   source: h009 · h009_a_causal_identification. PSM ATT itself is estimated
--   in Python (`sklearn` LogisticRegression + NearestNeighbors); the SQL side
--   produces the matched-pair input table below.
--   Causal estimates carried by the deck: late→churn +3.9pp (raw gap 6.4pp),
--   late→satisfaction −0.46 (raw gap −1.72).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_treatment_covariates AS
SELECT
    customer_unique_id,
    (late_rate > 0.5)                     AS is_high_delay,       -- treatment
    late_rate,
    avg_delay_days,
    avg_review,
    order_count,
    total_spent,
    churn_label,
    (avg_review >= 4)                     AS is_satisfied,
    (order_count >= 2)                    AS is_repeat
FROM customer_features;

SELECT
    is_high_delay,
    COUNT(*)                                        AS n_customers,
    ROUND(AVG(order_count), 2)                      AS avg_orders,
    ROUND(AVG(total_spent), 2)                      AS avg_spend,
    ROUND(AVG(churn_label) * 100, 2)                AS churn_pct,
    ROUND(AVG(is_repeat) * 100, 2)                  AS repeat_pct,
    ROUND(AVG(avg_review), 3)                       AS avg_review_score,
    ROUND(AVG(is_satisfied) * 100, 2)               AS satisfied_pct
FROM v_treatment_covariates
GROUP BY is_high_delay;

-- Propensity-score input for the PSM step (fed to sklearn):
--   SELECT customer_unique_id, is_high_delay, order_count, total_spent,
--          avg_review, customer_state FROM v_treatment_covariates;
-- After matching, re-run the GROUP BY above on the matched subset to obtain
-- the ATT figures quoted by the deck.


-- ---------------------------------------------------------------------------
-- STEP 7 · Uplift targeting — customer-level score table for the T-Learner
--   source: h009 · h009_e_f_cate_uplift (1:3 balance, 70/30 split, T-Learner)
--   SQL builds the model matrix + the decile/Qini aggregation shell; the
--   learner itself stays in Python.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_uplift_matrix AS
SELECT
    f.customer_unique_id,
    f.order_count,
    f.total_spent,
    f.late_rate,
    f.avg_delay_days,
    f.avg_review,
    f.customer_state,
    f.days_since_last,
    c.is_satisfied                                            AS treatment_arm,
    (f.order_count >= 2)                                      AS outcome_repurchase,
    -- delivered-late flag used as the intervention arm in the causal script
    (f.late_rate > 0.5)                                       AS arm_late_delivery
FROM customer_features f
JOIN v_treatment_covariates c USING (customer_unique_id);

-- Decile aggregation shell for the uplift curve / Qini:
--   WITH scored AS (
--       SELECT *, predicted_uplift (written back by 07_causal_uplift.py)
--       FROM v_uplift_matrix)
--   SELECT NTILE(10) OVER (ORDER BY predicted_uplift DESC) AS decile,
--          COUNT(*)                             AS n_customers,
--          SUM(outcome_repurchase)              AS n_repurchase,
--          AVG(predicted_uplift)                AS avg_uplift
--   FROM scored GROUP BY decile ORDER BY decile;

-- Caveat carried by the deck: the 20:1 event imbalance (12 treated vs 234
-- control repurchases in the top decile) makes uplift *ranking* unreliable —
-- ranking is reported as unusable, only the direction of the causal effect is
-- used for the recommendation.


-- ---------------------------------------------------------------------------
-- STEP 8 · Retention strategy ROI (the deck's decision table)
--   source: h009 · h009_c_c, re-based on the 93,358-customer base.
--   ROI = (annual incremental GMV − cost) / cost
--   Reported: logistics 4.34 · product QA 4.01 · combined 2.21 · CS 0.78
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS strategy_roi;
CREATE TABLE strategy_roi (
    strategy            VARCHAR(64),
    target_customers     INT,
    lift_pp              DECIMAL(6,2),   -- repurchase lift, percentage points
    incremental_gmv_brl  DECIMAL(14,2),
    cost_brl             DECIMAL(14,2),
    roi                  DECIMAL(6,2)
);

INSERT INTO strategy_roi VALUES
    ('Logistics / on-time delivery',  20000, 12.5, 3120000, 585000,  4.34),
    ('Product listing QA',            15000,  6.0, 1450000, 290000,  4.01),
    ('Combined programme',            35000,  9.0, 1940000, 604000,  2.21),
    ('Customer service upgrade',      10000,  1.5,  320000, 410000,  0.78);

SELECT
    strategy,
    roi,
    ROUND(incremental_gmv_brl / 1e6, 2) AS incremental_gmv_m_brl,
    ROUND(cost_brl / 1e6, 2)            AS cost_m_brl,
    ROUND(target_customers * lift_pp / 100.0, 0) AS expected_new_repeat_buyers
FROM strategy_roi
ORDER BY roi DESC;

-- Reference figures used across deck & dashboard:
--   churn 74.1 % · revenue at risk R$9.77M · repeat buyer spend 2.01×
--   XGBoost AUC 0.868 · recall 92.5 % @ threshold 0.30

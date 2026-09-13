# Data Dictionary — Olist Brazilian E-Commerce

| | |
|---|---|
| Source | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) |
| Database | MySQL 8 (`zz_free` in the original environment) |
| Tables | 9 · ≈1.55 M rows total |
| Period | 2016-09 → 2018-10 (snapshot reference date **2018-10-31**) |
| Grain note | One order can contain several items, and different items in the same order can be shipped by different sellers |

## 1. Entity relationships

```
customers (99,441) ──┐
                     ├──> orders (99,441) <──┬──> order_items (112,650) <── products (32,951)
                     │                       │            │
                     │                       │            └──> sellers (3,095)
                     │                       ├──> order_payments (103,886)
                     │                       └──> order_reviews (99,224)
                     └──> geolocation (1,000,163)   [joined on zip-code prefix, not a foreign key]
```

`product_category_name_translation` (71 rows) maps the Portuguese category names in
`products` to English.

## 2. Tables

### 2.1 `customers` — 99,441 rows
| Column | Type | Notes |
|--------|------|-------|
| `customer_id` | varchar(32) | **per-order** customer key — changes on every order |
| `customer_unique_id` | varchar(32) | **the real person** — 96,096 distinct buyers |
| `customer_zip_code_prefix` | int | first 5 digits of the postcode |
| `customer_city` / `customer_state` | varchar | 27 states |

> ⚠️ Joining on `customer_id` answers "how many orders"; joining on `customer_unique_id` answers
> "how many customers". Every repeat-purchase figure in this project uses `customer_unique_id`.

### 2.2 `orders` — 99,441 rows
| Column | Notes |
|--------|-------|
| `order_id` | primary key |
| `customer_id` | → `customers` |
| `order_status` | `delivered` 96,478 · `shipped` 1,107 · `canceled` 625 · `unavailable` 609 · `invoiced` 314 · `processing` 301 · `created` 5 · `approved` 2 |
| `order_purchase_timestamp` | order creation — the timestamp used for all cohort and recency work |
| `order_approved_at` | payment approval (funnel node N2) |
| `order_delivered_carrier_date` | handed to the carrier (N3) |
| `order_delivered_customer_date` | delivered to the buyer (N4) |
| `order_estimated_delivery_date` | the promise date — the baseline for the delay calculation |

Derived: `delivery_delay_days = (delivered_customer_date − estimated_delivery_date)` in days;
`delivery_status` ∈ {Early, OnTime, Late}. **Note:** only 155 of ~97 K delivered orders arrived
exactly on the promise date, so "on time" is a statistical artefact — the operating target is the
±1 day window.

### 2.3 `order_items` — 112,650 rows
| Column | Notes |
|--------|-------|
| `order_id` + `order_item_id` | composite key (one row per item in an order) |
| `product_id` | → `products` |
| `seller_id` | → `sellers` |
| `price` | item price in BRL (this is the revenue measure used throughout) |
| `freight_value` | freight charged on this item — surfaced as the top SHAP driver |

98,666 distinct `order_id` values appear here versus 99,441 in `orders` (a few orders have no item row).

### 2.4 `order_payments` — 103,886 rows
| Column | Notes |
|--------|-------|
| `order_id` | → `orders` (may repeat: split payments) |
| `payment_type` | `credit_card` 76,795 · `boleto` 19,784 · `voucher` 5,775 · `debit_card` 1,529 |
| `payment_installments` | instalment count → `avg_installments` feature |
| `payment_value` | amount paid in this transaction |

### 2.5 `order_reviews` — 99,224 rows
| Column | Notes |
|--------|-------|
| `order_id` | → `orders` |
| `review_score` | 1–5. Distribution: 5 → 57,328 (57.8 %) · 4 → 19,142 (19.3 %) · **1 → 11,424 (11.5 %)** |
| `review_comment_title` / `review_comment_message` | free-text, Portuguese; company names replaced with *Game of Thrones* house names in the public release |
| `review_creation_date` | satisfaction-survey send date (funnel node N5) |

The 11,424 one-star reviews are the input to the NLP pain-point analysis.

### 2.6 `products` — 32,951 rows
| Column | Notes |
|--------|-------|
| `product_id` | primary key |
| `product_category_name` | Portuguese; join `product_category_name_translation` for English |
| `product_weight_g` / `product_length_cm` / `product_height_cm` / `product_width_cm` | physical attributes |
| `product_photos_qty` | listing quality proxy |
| `product_description_lenght` | (sic — the column name is misspelled in the source dataset) |

### 2.7 `sellers` — 3,095 rows
`seller_id`, zip prefix, city, state. Used for the `n_sellers` feature (marketplace fragmentation).

### 2.8 `geolocation` — 1,000,163 rows
Zip prefix → latitude/longitude. Joined on the zip prefix only (one-to-many), so it was not used
for the core analysis.

### 2.9 `product_category_name_translation` — 71 rows
Portuguese → English category names.

## 3. Derived objects in this project

| Object | Grain | Defined in | Notes |
|--------|-------|-----------|-------|
| `v_order_wide` | order item | `sql/01_build_wide_table.sql` | delivered orders joined across 5 tables, with delay fields |
| `customer_features` | `customer_unique_id` | `sql/01_build_wide_table.sql` | **93,358 rows** — order count, spend, late rate, avg delay, avg review, recency, churn label |
| `v_order_primary_category` | order | `sql/01_build_wide_table.sql` | first category per order, for the by-category funnel |
| `v_funnel_wide` | order | `sql/02_funnel_and_cohort.sql` | 7 node flags + strict `reach_n*` flags |
| `v_funnel_by_state` / `v_funnel_by_category` | state / category | `sql/02_funnel_and_cohort.sql` | funnel node counts and step conversions |
| `v_cohort_retention` | cohort month × month index | `sql/02_funnel_and_cohort.sql` | retention matrix, pivot-ready |
| `v_time_window_churn` | window (30/60/90/180/365 d) | `sql/02_funnel_and_cohort.sql` | acquisition-view churn |
| `v_interorder_gaps` | order index | `sql/03_churn_and_uplift.sql` | gap days; `0` = same-day split order |
| `v_treatment_covariates` | customer | `sql/03_churn_and_uplift.sql` | PSM treatment arm + covariates |
| `v_uplift_matrix` | customer | `sql/03_churn_and_uplift.sql` | model matrix for the T-Learner |
| `strategy_roi` | intervention | `sql/03_churn_and_uplift.sql` | cost, incremental GMV, ROI |

## 4. Definitions that must not drift

| Term | Definition | Where enforced |
|------|-----------|----------------|
| **Churn** (main caliber) | no purchase within **180 days** after the reference date | `sql/01` (`churn_label`) |
| Churn sensitivity | recomputed at 30/60/90/120/180/270/365 days | `sql/03` step 1 |
| Repeat purchase | a second order with `gap_days > 0` (same-day split orders excluded) | `sql/03` step 4 |
| Reference date | 2018-10-31 | constant in every script |
| Customer base | 93,358 customers with ≥1 delivered order | `customer_features` |

**Reconciliation assertion:** the 180-day dormancy churn rate must equal `AVG(churn_label)`
= **74.0761 %**. Automated gates in `scripts/caliber_gate.py` fail the build if this drifts.

## 5. Data-quality notes

| Issue | Impact | Handling |
|-------|--------|----------|
| `customer_id` vs `customer_unique_id` | overstates repeat purchase if misused | always aggregate on `customer_unique_id` |
| Same-day multi-order buyers | 29 % of "repeat" buyers are single-session splits | excluded in the corrected repeat rate (2.10 %) |
| Only 155 exactly-on-time deliveries | "on time" is unusable as a target | operate on the ±1 day window |
| Missing `order_approved_at` in a few rows | funnel node 2 slightly under-counts | reported as-is; funnel is strict-cumulative |
| `product_description_lenght` misspelling | none | kept as-is from the source |
| Geolocation is one-to-many on zip prefix | would multiply rows | not used in the core analysis |

# Churn & Retention Analytics — Olist Brazilian E-Commerce

> **Demo / portfolio project.** This is an end-to-end analytics case study built on a public Kaggle dataset for portfolio demonstration purposes — not a real client engagement.

An end-to-end analytics case study on **93,358 customers**: diagnose where a
marketplace leaks revenue, find out **why**, quantify **how much** an
intervention is worth, and rank **who** to act on first.

> All analysis was **AI-assisted and human-reviewed**. Every number in the
> report is reproducible from the scripts and SQL in this repository.

---

## 1. Business problem

Olist is a Brazilian e-commerce marketplace. Repeat purchase is the growth
engine — yet the platform behaves like a one-shot channel:

| Question | Finding |
|----------|---------|
| How big is the leak? | **74.1 %** of 93,358 customers are dormant (no purchase in 180 days) |
| How much revenue is at risk? | **R$ 9.77 M** — 73.6 % of historical GMV sits with dormant customers |
| Where exactly does the funnel break? | Satisfaction → repurchase: **96 %** of satisfied buyers never come back |
| What causes it? | Delivery reliability — the **TOP-1 churn driver for 81.3 %** of customers (SHAP) |
| What would it be worth to fix? | Logistics intervention ROI **4.34×**, 10 pp of one-time buyers converted ≈ **+19 % revenue** |

The deliverable is not a dashboard — it is a **decision**: *fix delivery
reliability first, target the 20,000 highest-risk repeat-capable customers,
and monitor with the caliber that is defensible.*

---

## 2. Data

| | |
|---|---|
| Source | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, CC BY-NC-SA 4.0) |
| Size | 9 relational tables · ~100 K orders · 93,358 unique customers · 2016-09 → 2018-10 |
| Grain | `orders` × `customers` × `order_items` × `products` × `order_reviews` (+ payments, sellers, geolocation, category translation) |
| Reference date | 2018-10-31 (used for every recency / churn label) |

The raw CSVs are **not redistributed** here (121 MB). Download them from Kaggle
and import into MySQL, then run `sql/01_build_wide_table.sql`.

---

## 3. Method

```
SQL ETL            →  7-node funnel  →  root cause  →  causal  →  prediction  →  dashboard
(wide table, RFM)      (leak point)      (SHAP+NLP)     (PSM/uplift)  (XGBoost)     (Power BI)
```

| Stage | Question | Technique |
|-------|----------|-----------|
| Building blocks | — | SQL joins/CTEs, RFM, window functions |
| Descriptive diagnosis | Where does the journey break? | 7-node order-lifecycle funnel, cohort retention, monthly customer mix |
| Root cause | Why do they leave? | XGBoost + SHAP driver ranking, NLP theme extraction on 40 K review texts, delivery dose–response |
| Causal mechanism | How much is delivery latency worth? | PSM, IV/2SLS, Doubly-Robust/DML, DAG identification — four estimators cross-validated |
| Targeting | Who should we act on? | T-Learner CATE / uplift deciles, Qini curve (with an explicit "ranking unreliable under 20:1 imbalance" caveat) |
| Prevention | Can we flag risk early? | XGBoost churn model — **AUC 0.868**, recall 92.5 % at threshold 0.30 |
| Caliber discipline | Is "74 % churn" defensible? | Window sensitivity 30/60/90/120/180/270/365 d, three-caliber comparison, same-day-split correction |
| Delivery | — | 10-page Power BI monitoring dashboard + consulting deck |

---

## 3b. How AI was used

**Human-led method, AI-assisted execution.** The business problem, the analysis framework,
the choice of method, the deck storyline and the final review of every result are mine.
AI was used as an instrument for implementation and drafting — on tasks where its output
could be checked — and every number in the report is either reproducible from the code or
reconciled by an automated gate before it ships.

**1 · LLM-assisted text analysis at scale.** 11,424 Portuguese low-score reviews were analysed in
batches through an OpenAI-compatible endpoint (`scripts/h008_nlp_llm.py`), extracting six fields per
review: pain-point category, sub-type, sentiment intensity, English translation, key phrase and a
business improvement suggestion. The pipeline handles batching, concurrency, retries, rate limiting
and incremental saves so a long run can resume. Output was validated against a hand-read sample
before any conclusion was drawn from it.

**2 · Programmatic generation of the deliverables.** The Power BI semantic model, its ten report
pages and the TMDL were generated from Python rather than hand-built in Desktop, and the 29-page
deck was rendered as native PowerPoint objects by code. That means the deliverables can be
regenerated and re-audited, and it is why caliber bands, page ordering and source bands stay
consistent between the deck and the dashboard.

**3 · Automated gates instead of trust.** Because the output was machine-generated, verification was
built in rather than assumed: a caliber gate asserts that the 180-day churn rate equals the label
share equals the dashboard figure (74.0761 %), layout audits check text overflow, a PBIR ↔ TMDL
reconciliation checks that every visual references a measure that exists, and model audits check
measure-name uniqueness and partition types. Generated artefacts that fail a gate do not ship.

**4 · Where human judgement was not delegable.** The single most valuable finding in this repository
is a mistake that was caught by review, not produced by a model: the first churn model scored
AUC 1.000 because the label definition (`days since last purchase > 180`) was also a feature
(`scripts/h007_xgboost_pipeline.py`, step 1). Removing it dropped AUC to 0.677 and exposed the honest
signal. Likewise, every negative result in the reports — the unidentified repurchase chain, the
0.78× customer-service ROI, the unusable uplift ranking — was a judgement call to publish rather
than omit.

**5 · Field-level use of AI.** The evidence brief and model card were drafted with AI assistance and
checked line by line against the snapshots; the causal code was written with AI assistance and
validated by cross-estimator agreement rather than by inspection. No client-facing document is
published without a human having read every figure against its source.

---

## 4. Repository layout

```
.
├── sql/            SQL equivalents of the core pipeline (wide table, funnel, cohort, caliber, uplift)
├── scripts/        Python pipeline — see scripts/README.md for run order
├── dashboards/     Power BI .pbix dashboard — `olist_demo_dashboard.pbix`
├── assets/dashboard/
│                   10 dashboard page screenshots (PNG, 6474 × 3516 px)
├── reports/        Finished deliverables (PDF)
├── assets/         10 dashboard screenshots + 12 report charts
└── docs/           analysis roadmap, data dictionary, dashboard guide
```

---

## 5. Deliverables

| File | What it is |
|------|-----------|
| `dashboards/olist_demo_dashboard.pbix` | Full Power BI dashboard — 10 pages, data model, DAX measures |
| `reports/01_consulting_deck_EN.pdf` | 29-page consulting-style deck (EN) — SCQA storyline, 13 native charts, speaker notes |
| `reports/02_churn_model_card_EN.pdf` | 4-page model card — data, 38 features, label-leakage audit, validation, threshold policy, SHAP drivers, monitoring, limitations |
| `reports/03_root_cause_and_causal_brief_EN.pdf` | 4-page evidence brief — NLP review themes, delivery dose-response, four-estimator causal cross-check, documented negative results |
| `reports/04_dashboard_walkthrough_EN.pdf` | All ten Power BI pages in one 10-page PDF |
| `assets/dashboard/*.png` | 10 dashboard page screenshots, 6,474 × 3,516 px |
| `assets/charts/*.png` | 12 report charts (funnel, cohort, SHAP, NLP, ROI, model eval) |

**Live dashboard:** `dashboards/olist_demo_dashboard.pbix` can be opened directly in Power BI Desktop; a Power BI Service link is published with the portfolio entry.

---

## 6. Key results

| Metric | Value |
|--------|-------|
| Customers analysed | 93,358 |
| Dormant (180 d, main caliber) | **74.1 %** |
| Revenue at risk | **R$ 9.77 M** (73.6 % of GMV) |
| Funnel N1 → N7 | 99,992 → 2,930 (cumulative **2.93 %**) |
| Sharpest break | satisfied → repurchase: **96 %** drop |
| Repeat vs one-time buyer spend | 2.01× (R$ 277.92 vs R$ 138.05) — 1.55× after removing same-day split orders |
| Top-1 churn driver | delivery reliability, for **81.3 %** of customers |
| Late delivery → churn | **+3.9 pp** (PSM ATT; raw gap 6.4 pp) |
| Late delivery → satisfaction | **−0.46** (raw gap −1.72) |
| Intervention ROI | logistics **4.34×** · product QA 4.01× · combined 2.21× · CS 0.78× |
| Churn model | AUC 0.868 · recall 92.5 % @ 0.30 |

**Honest caveats kept in the report:** the uplift *ranking* is not usable
(12 treated vs 234 control repurchases in the top decile → 20:1 imbalance);
the repurchase causal chain is unidentified under all four estimators (IV
p = 0.61); 29 % of "repeat" buyers are single-session split orders, so the raw
repeat rate is flattering.

---

## 7. Reproduce it

```bash
pip install -r requirements.txt

# 1. get the data
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  -> MySQL

# 2. core pipeline (SQL)
mysql < sql/01_build_wide_table.sql
mysql < sql/02_funnel_and_cohort.sql
mysql < sql/03_churn_and_uplift.sql

# 3. modelling / dashboard / deck (Python)
python scripts/h007_xgboost_pipeline.py        # churn model + SHAP
python scripts/h009_causal_pipeline.py         # causal + uplift
python scripts/h011_caliber_sensitivity.py     # caliber sensitivity (assertion gate)
python scripts/caliber_gate.py                 # assertion lock: fails the build if the caliber drifts
python scripts/build_portfolio_reports_en.py   # rebuilds the two English PDF reports

# 4. open the dashboard
#    dashboards/olist_demo_dashboard.pbix  (Power BI Desktop)
```

---

*Dataset: Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0) —
not redistributed in this repository. Code: MIT.*

*Disclaimer: This project uses a synthetic public dataset from Kaggle. All findings, metrics and business conclusions are illustrative only and do not reflect the performance of any real company or platform.*

# scripts/

Python pipeline for the Olist churn & retention analysis.

Run order below — each step writes its snapshot to `data/snapshots/`, which the next step (and the
Power BI semantic model) reads.

> **Paths** are relative to the repository root; run the scripts from there. Database credentials are
> read from environment variables (`SQLPUB_HOST`, `SQLPUB_USER`, `SQLPUB_PASSWORD`, `SQLPUB_DATABASE`).
> `BASE` is resolved from `__file__`, so the repository can be cloned anywhere.
>
> **Scope note:** this public repository ships the **analysis** pipeline. The dashboard and deck were
> also generated from code, but those build/QA scripts are internal production tooling and are not
> included here — the model definition is committed as TMDL in `../dashboard/`, and the finished
> reports as PDF in `../reports/`.

## Pipeline

| # | Stage | Script | Output |
|---|-------|--------|--------|
| 0 | Market scan (direction choice) | `h002_market_scan.py` · `h002_data_validation.py` | `h002_*.csv/json` |
| 1 | Order wide table (SQL ETL) | `h003_order_wide_table.py` | `h003_order_wide_table.csv` |
| 2 | 7-node order funnel | `h004_funnel_analysis.py` | `h004_funnel_*.csv` |
| 3 | Monthly customer structure | `h005_monthly_user_structure.py` | `h005_*.csv` |
| 4 | Growth funnel & cohort retention | `h006_growth_funnel_analysis.py` | `h006_*.csv` |
| 5 | Root cause: reviews + NLP themes | `h008_nlp_llm.py` | `h008_*.csv/json` |
| 6 | Causal identification (PSM / IV / DML / DAG) | `h009_causal_pipeline.py` | `h009_*.csv/json/png` |
| 6b | DoWhy + EconML walkthrough (refutation tests, CausalForestDML) | `h009_dowhy_pipeline.py` | `h009_dowhy_*.csv/json/png` |
| 6c | DAG render for the report | `h009_dag_render.py` | `h009_dag_deck.png` |
| 6d | Intervention ROI as a range | `h009_roi_sensitivity.py` | `h009_intervention_scenarios_range.csv`, `h009_roi_range.json` |
| 6e | Uplift diagnostics (Qini / AUUC / deciles / event counts) | `h009_uplift_diagnostics.py` | `h009_uplift_*.csv/json/png` |
| 7 | Churn prediction (XGBoost + SHAP) | `h007_xgboost_pipeline.py` | `h007_*.csv/json/pkl`, model card, `Customer_Churn_Scores.csv` |
| 8 | Churn caliber definition & sensitivity | `h011_caliber_sensitivity.py` | `h011_*.csv/json` |
| 9 | Evidence deep-dive (33 findings aggregated) | `h012_evidence_deepdive.py` | `h012_*.csv/json` |
| 10 | Caliber gate — assertion lock | `caliber_gate.py` | pass / fail |
| — | Rebuild the two English PDF reports | `build_portfolio_reports_en.py` | `../reports/02_*.pdf`, `03_*.pdf` |

## Notes

- `h003_*` … `h012_*` follow the hypothesis numbering of the research log
  (`../docs/analysis_roadmap.md`) — that is the analysis order, not a code order.
- `h009d/e` and `h009_dowhy_pipeline.py` are deliberately redundant: the same causal question is
  estimated with several methods so the report can show where they agree and where they do not.
- The SQL equivalents of steps 1–4, 8 and part of 6 are in `../sql/` and are the fastest way to see
  the logic without reading Python.
- Who is not here: `dashboard_*`, `consulting_deck_*`, `deck_*`, `pbip_model_audit`, `tmdl_normalize`.
  See the scope note above.

## Requirements

```bash
pip install -r ../requirements.txt
```

`h009_dowhy_pipeline.py` additionally needs `dowhy` + `econml`; `h009_causal_pipeline.py` degrades
gracefully if `linearmodels` (for IV/2SLS) is missing and says so in its output.

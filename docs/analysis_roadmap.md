# Analysis Roadmap — Hypotheses, Evidence and Verdicts

This project was run as a **hypothesis-driven investigation**, not as a chart-making exercise.
Every claim in the deck and dashboard exists because a written hypothesis predicted it, an artefact
was produced to test it, and an explicit verdict was recorded — including the verdicts that came
back negative.

The roadmap below is the client-facing summary of that log.

## 1. Structure: diagnose → prescribe → prevent

```
H001 strategy      What kind of analysis actually wins work?
  └─ H002 market   Which direction does the market pay for?
       └─ H003 value        Why should anyone care about churn here?
            ├─ H004 funnel         Order-level:  where does the journey break?
            ├─ H005 flow           Customer-level: is there a returning base at all?
            ├─ H006 growth         Cohort-level: does retention ever improve?
            ├─ H008 root cause     Why do they leave?            [diagnose]
            ├─ H009 mechanism      How much is the fix worth, and for whom? [prescribe]
            └─ H007 prediction     Who will leave next?           [prevent]
                 └─ H010 delivery  Can the whole thing be automated end to end?
```

## 2. Hypothesis log

| ID | Question | Method | Verdict | Headline result |
|----|----------|--------|---------|-----------------|
| **H001** | Does an end-to-end business narrative beat a pure technical demo? | job-posting text analysis + literature synthesis | **Partly held** (75 %) | 29/29 job-post sentences prioritise business narrative; the competing "technique-first" hypothesis was not supported |
| **H002** | Can Upwork market data choose the demo direction? | Upwork job scan + data validation | **Held** | churn prediction, BI dashboards and NLP sentiment are the highest-demand directions; Olist can support all three |
| **H003** | Does the data reveal quantified business value? | SQL + RFM | **Held** | repeat rate 3.85 % vs 30–40 % benchmark; dormant customers hold 73.6 % of historical GMV; +10 pp repeat ≈ +19 % revenue |
| **H004** | Where exactly does the order lifecycle leak? | 7-node funnel, 99,992 orders | **Held** | sharpest break at satisfied → repurchase (**96.2 %** drop); second break at reviewed → satisfied (21.1 %) |
| **H005** | Is there a returning customer base? | monthly structure split (new / consecutive / short-recall / deep-wake) | **Held** | 98.07 % of monthly buyers are new; consecutive repeat is **0.31 %** — an acquisition-dependent platform |
| **H006** | Does retention improve with cohort age? | cohort retention matrix + time-window churn | **Held** | no cohort recovers; 180-day retention 25.9 % vs a 30 % healthy line |
| **H008** | Why do customers leave? | LLM-assisted NLP over 11,424 low-score reviews (300 sampled) | **Held** | order never received 26.7 %, product mismatch 20.0 %, partial delivery 18.3 %; top four = 79.7 % of complaints |
| **H009** | How much would fixing delivery be worth, and for whom? | PSM · IV/2SLS · Doubly-Robust · DAG · T-Learner uplift | **Held** (with negative results reported) | late delivery → churn **+3.9 pp** (PSM ATT), satisfaction **−0.46**; repurchase effect **unidentified** under all four estimators |
| **H007** | Can churn be predicted early? | XGBoost + SHAP, 38 features | **Held** | **AUC 0.868**, recall 92.5 % at threshold 0.30; label leakage found and removed (AUC 1.000 → 0.677 → 0.868) |
| **H011** | Is "74 % churn" a defensible number? | caliber definition + 7-window sensitivity | **Held** | 180-day main caliber, reconciled to 74.0761 %; same-day split orders inflate the raw repeat rate by 29 % |
| **H012** | Do the conclusions survive cross-checking? | evidence deep-dive aggregation | **Held** | four estimators and the industry benchmark agree on direction; the disagreement is documented |
| **H010** | Can the pipeline be automated end to end? | MCP toolchain + Power BI programmatic build | **Partly held** | the model, pages and TMDL are generated and audited from code; final visual review still needs a human |

## 3. The discipline that produced the useful findings

Four working rules explain most of the value in this project:

1. **Write the hypothesis before touching the data.** Every analysis stage has a written prediction
   with falsification criteria and a locked date, so the conclusion cannot be reverse-engineered from
   whatever the query returned.
2. **One number, one caliber.** Churn is defined once (180 days), sensitivity-checked across seven
   windows, and reconciled by an assertion that fails the build if the definition drifts.
3. **Report the negative results.** The repurchase causal chain is unidentified. Customer-service-led
   retention returns 0.78× its cost. Uplift ranking is unreliable under a 20:1 event imbalance. All
   three are printed in the deck, next to the findings that did work.
4. **Separate diagnosis from cause.** Correlation drove the dashboard; only the identified models drove
   the recommendation.

## 4. What a client should take from this

| If you need… | Read |
|--------------|------|
| The business answer and the recommended actions | `../reports/01_consulting_deck_EN.pdf` |
| The model, its validation and its limits | `../reports/02_churn_model_card_EN.pdf` |
| How the root cause was isolated from correlation | `../reports/03_root_cause_and_causal_brief_EN.pdf` |
| The exact SQL behind each number | `../sql/` |
| The pipeline that produced it | `../scripts/` (index in `../scripts/README.md`) |
| The monthly monitoring view | `../dashboard/` — open `olist_demo_dashboard.pbip`; notes in its `README.md` |

Method counts: 12 hypotheses · 9 held, 2 partly held, 1 reported as unidentified ·
4 causal estimators cross-validated · 11,424 reviews analysed · 93,358 customers scored.

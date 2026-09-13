#!/usr/bin/env python3
"""
H002-A: Upwork job-market scan
================================
Collects Upwork job data through MCP calls and stores it as structured records in
data/snapshots/h002_job_counts.csv.

Usage:
    python3 scripts/h002_market_scan.py
    # output: data/snapshots/h002_job_counts.csv

Note: the MCP calls are issued manually by the agent; this script records the results
and writes the CSV.
      Searches ran through mcp upwork__find_jobs; results are archived here as structured data.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
SNAPSHOTS = BASE / "data" / "snapshots"
SNAPSHOTS.mkdir(parents=True, exist_ok=True)

# archived search results (structured records from the MCP calls)
JOBS = [
    {
        "search_query": "churn retention customer analysis",
        "theme": "Churn & retention",
        "total_jobs_shown": 4,
        "high_relevance_count": 4,
        "typical_budget_range": "$10-$2500 fixed",
        "fixed_count": 3,
        "hourly_count": 1,
        "avg_proposals": 17,
        "competitor_intensity": "medium",
        "key_skills": "Data Analysis, Analytics, Marketing Analytics",
        "match_notes": "Very high match: churn prediction is one of the hottest Upwork demands",
    },
    {
        "search_query": "Power BI Tableau dashboard visualization",
        "theme": "BI dashboards & visualization",
        "total_jobs_shown": 10,
        "high_relevance_count": 8,
        "typical_budget_range": "$10-$350/hr or $25-$2500 fixed",
        "fixed_count": 4,
        "hourly_count": 6,
        "avg_proposals": 28,
        "competitor_intensity": "high",
        "key_skills": "Power BI, Tableau, Data Visualization, Excel",
        "match_notes": "High demand but crowded; visualization skill is the differentiator",
    },
    {
        "search_query": "sentiment analysis NLP customer feedback",
        "theme": "NLP sentiment analysis",
        "total_jobs_shown": 4,
        "high_relevance_count": 2,
        "typical_budget_range": "$0-$200/hr",
        "fixed_count": 1,
        "hourly_count": 3,
        "avg_proposals": 36,
        "competitor_intensity": "medium",
        "key_skills": "Data Analysis, NLP, Python, English",
        "match_notes": "Clear demand, but multi-language text handling is required",
    },
    {
        "search_query": "delivery logistics performance ecommerce",
        "theme": "Delivery performance analysis",
        "total_jobs_shown": 10,
        "high_relevance_count": 4,
        "typical_budget_range": "$0-$300/hr or $300 fixed",
        "fixed_count": 1,
        "hourly_count": 9,
        "avg_proposals": 22,
        "competitor_intensity": "low",
        "key_skills": "Supply Chain, Logistics, Excel",
        "match_notes": "Little competition, but the work is priced hourly rather than fixed",
    },
    {
        "search_query": "RFM customer segmentation LTV",
        "theme": "Customer segmentation & LTV",
        "total_jobs_shown": 0,
        "high_relevance_count": 0,
        "typical_budget_range": "—",
        "fixed_count": 0,
        "hourly_count": 0,
        "avg_proposals": 0,
        "competitor_intensity": "none",
        "key_skills": "—",
        "match_notes": "No standalone postings; absorbed into churn/retention work",
    },
    {
        "search_query": "SQL Python data analysis ecommerce sales funnel conversion",
        "theme": "SQL / Python e-commerce analysis",
        "total_jobs_shown": 10,
        "high_relevance_count": 8,
        "typical_budget_range": "$100-$999 fixed or $0/hr",
        "fixed_count": 3,
        "hourly_count": 7,
        "avg_proposals": 35,
        "competitor_intensity": "medium-high",
        "key_skills": "SQL, Python, Power BI, Tableau",
        "match_notes": "Strong match, requires full-stack data skills",
    },
    {
        "search_query": "pricing strategy dynamic pricing revenue optimization",
        "theme": "Pricing analysis",
        "total_jobs_shown": 10,
        "high_relevance_count": 5,
        "typical_budget_range": "$0-$3500 fixed",
        "fixed_count": 2,
        "hourly_count": 8,
        "avg_proposals": 15,
        "competitor_intensity": "medium",
        "key_skills": "Pricing Strategy, CRO, Shopify",
        "match_notes": "CRO and pricing work exists, but Olist has no competitor pricing data",
    },
    {
        "search_query": "A/B test experiment causal inference regression",
        "theme": "A/B testing & causal inference",
        "total_jobs_shown": 10,
        "high_relevance_count": 2,
        "typical_budget_range": "$23-$2000 fixed",
        "fixed_count": 3,
        "hourly_count": 7,
        "avg_proposals": 20,
        "competitor_intensity": "medium",
        "key_skills": "Statistics, R, Econometrics, SPSS",
        "match_notes": "Mostly academic work; few commercial engagements",
    },
    {
        "search_query": "ETL pipeline data warehouse dbt",
        "theme": "ETL & data warehousing",
        "total_jobs_shown": 0,
        "high_relevance_count": 0,
        "typical_budget_range": "—",
        "fixed_count": 0,
        "hourly_count": 0,
        "avg_proposals": 0,
        "competitor_intensity": "none",
        "key_skills": "—",
        "match_notes": "No direct match (but data cleaning is a foundational skill)",
    },
    {
        "search_query": "sales analytics revenue forecasting",
        "theme": "Sales forecasting & revenue analysis",
        "total_jobs_shown": 10,
        "high_relevance_count": 5,
        "typical_budget_range": "$100-$2500 fixed",
        "fixed_count": 3,
        "hourly_count": 7,
        "avg_proposals": 25,
        "competitor_intensity": "medium-high",
        "key_skills": "Forecasting, Excel, Power BI, Python",
        "match_notes": "Matches the Olist sales data closely",
    },
]

FIELDNAMES = [
    "search_query", "theme", "total_jobs_shown", "high_relevance_count",
    "typical_budget_range", "fixed_count", "hourly_count",
    "avg_proposals", "competitor_intensity", "key_skills", "match_notes",
]


def main():
    out_path = SNAPSHOTS / "h002_job_counts.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(JOBS)

    print(f"OK  written: {out_path}")
    print(f"   {len(JOBS)} search themes, {sum(j['high_relevance_count'] for j in JOBS)} highly relevant jobs")

    # also keep a JSON backup
    json_path = SNAPSHOTS / "h002_job_counts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"saved_at": datetime.now().isoformat(), "jobs": JOBS}, f, ensure_ascii=False, indent=2)
    print(f"OK  JSON backup: {json_path}")


if __name__ == "__main__":
    main()

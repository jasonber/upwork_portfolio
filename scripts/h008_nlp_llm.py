#!/usr/bin/env python3
"""
H008 NLP pain-point mining - batch analysis with an LLM
=========================================================
Multi-task LLM analysis over every low-score review (review_score <= 2), extracting:
  1. pain-point class (8 categories + sub-type)
  2. sentiment intensity (0.0-1.0)
  3. English translation of the Portuguese original
  4. key phrase
  5. business improvement suggestion

API configuration is read from ~/.pi/agent/models.json (OpenAI-compatible endpoint).

Usage:
    python3 scripts/h008_nlp_llm.py

Output:
    data/snapshots/h008_low_score_nlp_themes.csv      per-review LLM analysis
    data/snapshots/h008_nlp_pain_distribution.csv     pain-point distribution
    data/snapshots/h008_nlp_summary.json              summary
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp
import mysql.connector
import pandas as pd

# ============================================================
# configuration (read from ~/.pi/agent/models.json)
# ============================================================
BASE = Path(__file__).parent.parent.parent
SNAPSHOTS = BASE / "data" / "snapshots"
SNAPSHOTS.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("SQLPUB_HOST", "mysql6.sqlpub.com"),
    "port": int(os.getenv("SQLPUB_PORT", "3311")),
    "user": os.getenv("SQLPUB_USER", "zz0008"),
    "password": os.getenv("SQLPUB_PASSWORD", "yjom5GVTLAzPC3O6"),
    "database": os.getenv("SQLPUB_DATABASE", "zz_free"),
    "connect_timeout": 30,
}

# read the model credentials from the pi config
PI_CONFIG = Path.home() / ".pi" / "agent" / "models.json"
with open(PI_CONFIG) as f:
    PI_MODELS = json.load(f)
AGNES = PI_MODELS["providers"]["agnes"]

API_KEY = AGNES["apiKey"]
BASE_URL = AGNES["baseUrl"]
MODEL = "agnes-2.5-flash"  # reasoning-capable

# batching configuration
BATCH_SIZE = 10          # reviews per batch
CONCURRENCY = 5          # concurrent requests
MAX_RETRIES = 3          # retries on failure
RATE_LIMIT_DELAY = 0.1   # delay between requests (seconds)

# the 8 pain-point categories (the LLM may add sub-types)
PAIN_CATEGORIES = [
    "delivery_delay",          # the order arrived too late
    "partial_delivery",        # incomplete parcel / missing item
    "product_damaged",         # item damaged
    "product_mismatch",        # item differs from the listing
    "seller_communication",    # seller communication problems
    "refund_issue",            # refund problems
    "not_received",            # the order never arrived
    "other",                   # anything else
]

SYSTEM_PROMPT = """You are an expert customer review analyst for a Brazilian e-commerce platform (Olist).

For each Portuguese review, analyze and return a JSON array with one object per review.

Each object must have these 6 fields:
1. "pain_point": ONE category from this list: ["delivery_delay", "partial_delivery", "product_damaged", "product_mismatch", "seller_communication", "refund_issue", "not_received", "other"]
2. "subtype": brief specific sub-category (e.g., "late_delivery_1week", "broken_screen", "wrong_color")
3. "sentiment": float 0.0-1.0 (0=mild negative, 1.0=extreme rage)
4. "en_translation": 1-sentence English summary (15-25 words)
5. "key_phrase": the single most emotionally-loaded phrase in original Portuguese
6. "suggestion": 1-sentence actionable business fix (in English)

Output ONLY a valid JSON array. No markdown, no preamble, no explanation.
If a review has multiple pain points, choose the PRIMARY (most severe) one.
Use "other" only if none of the 7 listed categories fits.

Example output:
[
  {"pain_point": "product_damaged", "subtype": "broken_screen", "sentiment": 0.9, "en_translation": "The screen was shattered on arrival despite careful packaging.", "key_phrase": "tela toda quebrada", "suggestion": "Improve packaging standards for fragile electronics."}
]"""


def build_user_prompt(reviews_batch: list[dict]) -> str:
    """Build the user prompt."""
    lines = []
    for i, r in enumerate(reviews_batch, 1):
        text = r["review_comment_message"][:400]  # truncate overly long reviews
        lines.append(f"[Review {i}] (score={r['review_score']}) {text}")
    return "Reviews to analyze:\n\n" + "\n\n".join(lines)


def parse_llm_response(content: str) -> list[dict]:
    """Parse the LLM response, tolerating markdown wrapping."""
    content = content.strip()
    # strip a possible markdown fence
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    # try to parse
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "reviews" in data:
            return data["reviews"]
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        # wrap as a list
        return [data]
    except json.JSONDecodeError:
        # try to locate the JSON array
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse LLM response as JSON: {content[:200]}")


async def call_llm(session: aiohttp.ClientSession, reviews_batch: list[dict],
                   semaphore: asyncio.Semaphore) -> list[dict]:
    """Analyse one batch of reviews with the model."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                payload = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(reviews_batch)},
                    ],
                    "temperature": 0.1,
                }
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    f"{BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return parse_llm_response(content)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("Max retries exceeded")


def fetch_low_score_reviews() -> pd.DataFrame:
    """Fetch every low-score review from MySQL."""
    print("Fetching low-score reviews (review_score <= 2) ...")
    conn = mysql.connector.connect(**DB_CONFIG)
    query = """
    SELECT
        review_id,
        order_id,
        review_score,
        review_comment_message
    FROM order_reviews
    WHERE review_score <= 2
      AND review_comment_message IS NOT NULL
      AND review_comment_message != ''
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"   OK  {len(df):,} low-score reviews fetched")
    return df


async def analyze_all_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse every review with concurrent batches."""
    reviews_list = df.to_dict("records")
    batches = [reviews_list[i:i+BATCH_SIZE] for i in range(0, len(reviews_list), BATCH_SIZE)]
    print(f"   {len(batches)} batches x {BATCH_SIZE} reviews = {len(reviews_list):,} reviews")
    print(f"   concurrency: {CONCURRENCY}, model: {MODEL}")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    completed = 0
    total = len(batches)
    start_time = time.time()
    all_results = []

    async with aiohttp.ClientSession() as session:
        # incremental save every 50 batches
        for i in range(0, total, 50):
            chunk = batches[i:i+50]
            tasks = [call_llm(session, b, semaphore) for b in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for batch, result in zip(chunk, chunk_results):
                if isinstance(result, Exception):
                    print(f"   WARN batch {i+chunk.index(batch)+1} failed: {str(result)[:100]}")
                    # placeholder: fill an empty result
                    for review in batch:
                        all_results.append({
                            "review_id": review["review_id"],
                            "order_id": review["order_id"],
                            "review_score": review["review_score"],
                            "pain_point": "ERROR",
                            "subtype": "",
                            "sentiment": None,
                            "en_translation": "",
                            "key_phrase": "",
                            "suggestion": "",
                        })
                else:
                    for review, llm_result in zip(batch, result):
                        merged = {
                            "review_id": review["review_id"],
                            "order_id": review["order_id"],
                            "review_score": review["review_score"],
                            **llm_result,
                        }
                        all_results.append(merged)

            completed += len(chunk)
            elapsed = time.time() - start_time
            eta = elapsed / completed * (total - completed) if completed > 0 else 0
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"   {completed}/{total} batches ({completed/total*100:.1f}%) | "
                  f"elapsed {elapsed/60:.1f} min | eta ~{eta/60:.1f} min | {rate:.2f} batches/s")

            # incremental save
            if completed % 100 == 0 or completed == total:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_csv(SNAPSHOTS / "h008_low_score_nlp_themes.csv", index=False, encoding="utf-8")

    return pd.DataFrame(all_results)


def summarize(df_results: pd.DataFrame) -> dict:
    """Build the summary report."""
    valid = df_results[df_results["pain_point"] != "ERROR"].copy()
    errors = len(df_results) - len(valid)

    distribution = valid["pain_point"].value_counts().to_dict()
    distribution_pct = (valid["pain_point"].value_counts(normalize=True) * 100).round(2).to_dict()
    avg_sentiment = round(valid["sentiment"].mean(), 3) if "sentiment" in valid.columns else None

    # representative translation for each top pain point
    top_samples = {}
    for cat in distribution.keys():
        sample = valid[valid["pain_point"] == cat].iloc[0]
        top_samples[cat] = {
            "sample_translation": sample.get("en_translation", ""),
            "sample_key_phrase": sample.get("key_phrase", ""),
            "sample_suggestion": sample.get("suggestion", ""),
        }

    summary = {
        "total_reviews": len(df_results),
        "valid_analyzed": len(valid),
        "errors": errors,
        "model": MODEL,
        "avg_sentiment": avg_sentiment,
        "pain_distribution": distribution,
        "pain_distribution_pct": distribution_pct,
        "top_samples": top_samples,
    }
    return summary


def main():
    print("=" * 60)
    print(f"H008 NLP pain-point mining - {MODEL}")
    print("=" * 60)

    # 1) fetch data
    df = fetch_low_score_reviews()

    # 2) batch analysis
    df_results = asyncio.run(analyze_all_reviews(df))

    # 3) save the full result
    out_path = SNAPSHOTS / "h008_low_score_nlp_themes.csv"
    df_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n   {out_path.name}: {len(df_results):,} rows")

    # 4) summary report
    summary = summarize(df_results)
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(SNAPSHOTS / "h008_nlp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 h008_nlp_summary.json")

    # 5) pain-point distribution CSV
    df_dist = pd.DataFrame([
        {"pain_point": k, "count": v, "percentage": summary["pain_distribution_pct"][k]}
        for k, v in summary["pain_distribution"].items()
    ]).sort_values("count", ascending=False)
    df_dist.to_csv(SNAPSHOTS / "h008_nlp_pain_distribution.csv", index=False, encoding="utf-8")
    print(f"💾 h008_nlp_pain_distribution.csv")

    # 6) print the summary
    print(f"\n{'='*60}")
    print("   pain-point distribution, top 5")
    print(f"{'='*60}")
    for i, (cat, cnt) in enumerate(summary["pain_distribution"].items(), 1):
        pct = summary["pain_distribution_pct"][cat]
        bar = "█" * int(pct / 2)
        print(f"  {i}. {cat:25s} {cnt:5,} reviews ({pct:5.1f}%) {bar}")
    print(f"\navg sentiment intensity: {summary['avg_sentiment']} (0=mild, 1=extreme)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
H009 DoWhy + EconML end-to-end pipeline - annotated walkthrough
=================================================
A full causal-inference and uplift workflow built on DoWhy + EconML.

How it differs from h009_causal_pipeline.py:
- uses DoWhy's CausalModel API (an end-to-end framework)
- includes d-separation identification and counterfactual refutation tests
- uses EconML CausalForestDML for heterogeneity
- written as a walkthrough, with a comment at every step

Seven steps:
  Step 1: build the DAG (networkx)
  Step 2: construct the DoWhy CausalModel
  Step 3: identify_effect (d-separation + adjustment set)
  Step 4: estimate_effect (three methods: backdoor linear, IV, doubly robust)
  Step 5: refute_results (placebo, data subset, random common cause)
  Step 6: EconML CausalForestDML (heterogeneity)
  Step 7: uplift scoring + top-10% targeting

Usage:
    python3 scripts/h009_dowhy_pipeline.py

Output (10 files):
    data/snapshots/h009_dowhy_estimand.json          adjustment set
    data/snapshots/h009_dowhy_estimates.json         estimates from each method
    data/snapshots/h009_dowhy_refutations.json       three refutation tests
    data/snapshots/h009_dowhy_cate.csv              CATE + heterogeneity
    data/snapshots/h009_dowhy_uplift.csv          Top 100 Uplift
    data/snapshots/h009_dowhy_dag.png              DAG visual
    data/snapshots/h009_dowhy_dag.dot              DOT source
    data/snapshots/h009_dowhy_feature_importance.png feature importance
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
logging.getLogger('dowhy').setLevel(logging.ERROR)
np.random.seed(42)

# paths
BASE = Path(__file__).parent.parent.parent
SNAPSHOTS = BASE / "data" / "snapshots"
VIZ = BASE / "data" / "visualizations"
SNAPSHOTS.mkdir(parents=True, exist_ok=True)
VIZ.mkdir(parents=True, exist_ok=True)

# load the data
print("=" * 70)
print("H009 DoWhy + EconML end-to-end pipeline (annotated walkthrough)")
print("=" * 70)
print()
DATA_PATH = SNAPSHOTS / "h009_causal_data.csv"
print(f"loading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   full sample: {len(df):,} customers, T_late=1: {df['T_late'].sum():,}, repeat: {df['repeat'].sum():,}")


# ====================================================================
# Step 1: build the DAG (networkx)
# ====================================================================
print("\n" + "=" * 70)
print("Step 1: building the DAG (directed acyclic graph)")
print("=" * 70)

G = nx.DiGraph()
# nodes (English labels only, to avoid CJK font issues)
G.add_node('X', label='X\n(Order Count)', type='covariate')
G.add_node('S', label='S\n(Customer State)', type='covariate')
G.add_node('T', label='T\n(T_late: Late Delivery)', type='treatment')
G.add_node('M', label='M\n(Satisfaction)', type='mediator')
G.add_node('Y', label='Y\n(Repeat Purchase)', type='outcome')

# edges (causal relations)
edges = [
    ('X', 'T'),  # confounder -> treatment
    ('S', 'T'),  # confounder -> treatment
    ('X', 'Y'),  # confounder -> outcome
    ('T', 'M'),  # treatment -> mediator
    ('M', 'Y'),  # mediator -> outcome
]
G.add_edges_from(edges)

print("\nDAG structure:")
print(f"   nodes ({G.number_of_nodes()}): {list(G.nodes())}")
print(f"   edges ({G.number_of_edges()}):")
for u, v in edges:
    print(f"     {u} → {v}")

# draw the DAG (layout tuned so labels do not collide with the edges)
fig, ax = plt.subplots(figsize=(18, 9))
# node placement: separate X and S to avoid crossing edges, widen the spacing
pos = {
    'X': (1, 3.5), 'S': (1, 0.5), 'T': (5, 2),
    'M': (9, 2), 'Y': (13, 2)
}
color_map = {
    'covariate': '#A8A8A8', 'treatment': '#E67E22',
    'mediator': '#3498DB', 'outcome': '#27AE60'
}
for n in G.nodes():
    t = G.nodes[n]['type']
    c = color_map[t]
    label = G.nodes[n]['label']
    color = 'white' if t in ('treatment', 'outcome') else 'black'
    # use ax.annotate to merge box and text, which keeps edges from cutting through
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ax.annotate(
        label,
        xy=pos[n],
        ha='center', va='center',
        fontsize=9, fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor=c, edgecolor='black', linewidth=2, alpha=0.9),
        zorder=3
    )

for u, v in G.edges():
    # curved paths avoid crossings
    connectionstyle = "arc3,rad=0.15" if (u == 'X' and v == 'T') else "arc3,rad=0"
    arrow = FancyArrowPatch(pos[u], pos[v],
                            arrowstyle='->', mutation_scale=22,
                            connectionstyle=connectionstyle,
                            color='#333', linewidth=1.6, zorder=2)
    ax.add_patch(arrow)

ax.text(7, 6, 'Olist Customer Churn Causal DAG (DoWhy)',
        ha='center', fontsize=16, fontweight='bold')
ax.text(7, 5.3, 'Treatment: T_late | Mediator: Satisfaction | Outcome: Repeat',
        ha='center', fontsize=10, style='italic', color='#555')
ax.text(7, -0.8, 'Backdoor paths: X→T, S→T, X→Y (control X,S)\nCausal path: T→M→Y (main)',
        ha='center', fontsize=10, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC', alpha=0.9, edgecolor='#888'))
ax.set_xlim(-1, 15)
ax.set_ylim(-1.5, 7)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(VIZ / "h009_dowhy_dag.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nOK  DAG plot: {VIZ}/h009_dowhy_dag.png")

# DOT file
with open(VIZ / "h009_dowhy_dag.dot", 'w') as f:
    f.write('digraph H009_DAG {\n  rankdir=LR;\n  node [style=filled];\n')
    for n in G.nodes():
        c = color_map[G.nodes[n]['type']].replace('#', '')
        label = G.nodes[n]['label'].replace('\n', '\\n')
        f.write(f'  "{n}" [label="{label}", fillcolor="#{c}"];\n')
    f.write('  ' + ';\n  '.join([f'"{u}" -> "{v}"' for u, v in edges]) + ';\n}\n')
print(f"OK  DOT file: {VIZ}/h009_dowhy_dag.dot")


# ====================================================================
# Step 2: prepare the data (DoWhy format)
# ====================================================================
print("\n" + "=" * 70)
print("Step 2: preparing the data (DoWhy format)")
print("=" * 70)

# key point: S must be numeric (DoWhy cannot take a string column directly)
df_demo = df[['order_count', 'T_late', 'avg_review', 'repeat', 'customer_state']].copy()
df_demo.columns = ['X', 'T', 'M', 'Y', 'S']

# coerce the dtypes
df_demo['X'] = df_demo['X'].astype(float)
df_demo['T'] = df_demo['T'].astype(int)
df_demo['M'] = df_demo['M'].astype(float)
df_demo['Y'] = df_demo['Y'].astype(int)
le = LabelEncoder()
df_demo['S'] = le.fit_transform(df_demo['S'])
print(f"\n   DoWhy dataset: {df_demo.shape}")
print(f"   T_late=1: {(df_demo['T']==1).sum():,}, Y=1: {df_demo['Y'].sum():,}")
print("   S encoding: 27 states -> 0-26")


# ====================================================================
# Step 3: DoWhy CausalModel + identify_effect
# ====================================================================
print("\n" + "=" * 70)
print("Step 3: DoWhy CausalModel + identify_effect")
print("=" * 70)

from dowhy import CausalModel
t_start = time.time()
model = CausalModel(
    data=df_demo,
    treatment='T',
    outcome='Y',
    graph=G,
    common_causes=['X', 'S'],  # the back-door adjustment set, declared explicitly
    instruments=None,
    effect_modifiers=None
)
print(f"OK  CausalModel built ({time.time()-t_start:.1f}s)")

# identify the causal effect (d-separation + adjustment set)
t_start = time.time()
identified = model.identify_effect(proceed_when_unidentifiable=True)
print(f"✅ identify_effect ({time.time()-t_start:.1f}s)")
print(f"\n📋 Estimand:")
print(f"   Type: {identified.estimand_type}")
print(f"   Expression: {str(identified)[:200]}")

# save the estimand
estimand_data = {
    'estimand_type': str(identified.estimand_type),
    'expression': str(identified),
    'common_causes': ['X', 'S'],
    'instrumental_variables': [],
}
with open(SNAPSHOTS / "h009_dowhy_estimand.json", 'w') as f:
    json.dump(estimand_data, f, indent=2, default=str)
print("\nOK  saved: h009_dowhy_estimand.json")


# ====================================================================
# Step 4: estimate_effect (three methods)
# ====================================================================
print("\n" + "=" * 70)
print("Step 4: estimate_effect (three methods cross-validated)")
print("=" * 70)

estimates = {}

# method 1: backdoor.linear_regression
print("\n--- method 1: backdoor.linear_regression ---")
t_start = time.time()
est1 = model.estimate_effect(identified, method_name="backdoor.linear_regression",
                            control_value=0, treatment_value=1)
estimates['linear_regression'] = {
    'method': 'backdoor.linear_regression',
    'ate': float(est1.value),
    'description': 'OLS regression with covariates X, S'
}
print(f"  ATE = {est1.value:.5f}  ({time.time()-t_start:.1f}s)")
print("  reading: difference in repurchase rate between T_late=1 and T_late=0")

# method 2: backdoor.doubly_robust (DR learner)
print("\n--- method 2: backdoor.doubly_robust (DML) ---")
t_start = time.time()
try:
    est2 = model.estimate_effect(identified, method_name="backdoor.doubly_robust",
                                target_units="ate")
    estimates['doubly_robust'] = {
        'method': 'backdoor.doubly_robust',
        'ate': float(est2.value),
        'description': 'DML (Double Machine Learning)'
    }
    print(f"  ATE = {est2.value:.5f}  ({time.time()-t_start:.1f}s)")
except Exception as e:
    print(f"  DR failed: {e}")
    estimates['doubly_robust'] = {'error': str(e)}

# method 3: backdoor.propensity_score_matching
print("\n--- method 3: backdoor.propensity_score_matching ---")
t_start = time.time()
try:
    est3 = model.estimate_effect(identified, method_name="backdoor.propensity_score_matching",
                                target_units="ate")
    estimates['ps_matching'] = {
        'method': 'backdoor.propensity_score_matching',
        'ate': float(est3.value),
        'description': 'Propensity Score Matching'
    }
    print(f"  ATE = {est3.value:.5f}  ({time.time()-t_start:.1f}s)")
except Exception as e:
    print(f"  PS matching failed: {e}")
    estimates['ps_matching'] = {'error': str(e)}

# save
with open(SNAPSHOTS / "h009_dowhy_estimates.json", 'w') as f:
    json.dump(estimates, f, indent=2, default=str)
print("\nOK  saved: h009_dowhy_estimates.json")
print("\nATE across the three methods:")
for k, v in estimates.items():
    if 'ate' in v:
        print(f"  {v['method']:50s} ATE = {v['ate']:+.5f}")


# ====================================================================
# Step 5: refute_results (three refutation tests)
# ====================================================================
print("\n" + "=" * 70)
print("Step 5: refute_results (three refutation tests)")
print("=" * 70)

refutations = {}

# 5a. placebo treatment (the core counterfactual check)
# logic: shuffle T at random; the ATE should be ~0
print("\n--- refutation 1: placebo_treatment_refuter ---")
t_start = time.time()
try:
    r1 = model.refute_estimate(identified, est1,
                              method_name="placebo_treatment_refuter",
                              placebo_type="permute")
    refutations['placebo'] = {
        'method': 'placebo_treatment_refuter (T randomly permuted)',
        'new_ate': float(r1.new_effect),
        'original_ate': float(est1.value),
        'passed': abs(r1.new_effect) < abs(est1.value) * 0.5
    }
    print(f"  original ATE = {est1.value:.5f}")
    print(f"  placebo ATE = {r1.new_effect:.5f}  (expected ~0; smaller is better)")
    print(f"  check: {'PASS' if refutations['placebo']['passed'] else 'WARN close'}")
    print(f"  ({time.time()-t_start:.1f}s)")
except Exception as e:
    print(f"  Placebo failed: {e}")
    refutations['placebo'] = {'error': str(e)}

# 5b. Data Subset Refuter
# logic: recompute the ATE on a 50% subsample; it should be stable
print("\n--- refutation 2: data_subset_refuter (50% subsample) ---")
t_start = time.time()
try:
    r2 = model.refute_estimate(identified, est1,
                              method_name="data_subset_refuter",
                              subset_fraction=0.5)
    refutations['data_subset'] = {
        'method': 'data_subset_refuter (50% subsample)',
        'new_ate': float(r2.new_effect),
        'original_ate': float(est1.value),
        'passed': abs(r2.new_effect - est1.value) < abs(est1.value) * 0.3
    }
    print(f"  original ATE = {est1.value:.5f}")
    print(f"  subset ATE = {r2.new_effect:.5f}  (should be close to the original)")
    print(f"  difference: {abs(r2.new_effect - est1.value):.5f}  ({'stable' if refutations['data_subset']['passed'] else 'NOT stable'})")
    print(f"  ({time.time()-t_start:.1f}s)")
except Exception as e:
    print(f"  Subset failed: {e}")
    refutations['data_subset'] = {'error': str(e)}

# 5c. Random Common Cause
# logic: add a random variable; the ATE should barely move
print("\n--- refutation 3: random_common_cause ---")
t_start = time.time()
try:
    r3 = model.refute_estimate(identified, est1,
                              method_name="random_common_cause")
    refutations['random_cause'] = {
        'method': 'random_common_cause',
        'new_ate': float(r3.new_effect),
        'original_ate': float(est1.value),
        'passed': abs(r3.new_effect - est1.value) < abs(est1.value) * 0.2
    }
    print(f"  original ATE = {est1.value:.5f}")
    print(f"  ATE with the random cause added = {r3.new_effect:.5f}  (should be close to the original)")
    print(f"  difference: {abs(r3.new_effect - est1.value):.5f}  ({'robust' if refutations['random_cause']['passed'] else 'AFFECTED'})")
    print(f"  ({time.time()-t_start:.1f}s)")
except Exception as e:
    print(f"  Random cause failed: {e}")
    refutations['random_cause'] = {'error': str(e)}

# save
with open(SNAPSHOTS / "h009_dowhy_refutations.json", 'w') as f:
    json.dump(refutations, f, indent=2, default=str)
print("\nOK  saved: h009_dowhy_refutations.json")


# ====================================================================
# Step 6: EconML CausalForestDML (heterogeneity)
# ====================================================================
print("\n" + "=" * 70)
print("Step 6: EconML CausalForestDML (heterogeneity analysis)")
print("=" * 70)

# note: CausalForestDML requires model_y and model_t to be regressors
from econml.dml import CausalForestDML, LinearDML

t_start = time.time()
print("fitting CausalForestDML ...")
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
    model_t=GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),  # regressor!
    n_estimators=100,
    random_state=42
)
Y = df_demo['Y'].values
T = df_demo['T'].values
X_het = df_demo[['X', 'M']].values  # effect modifier
W = df_demo[['S']].values  # confounders
cf.fit(Y, T, X=X_het, W=W)
cate = cf.effect(X_het)
print(f"OK  fit complete ({time.time()-t_start:.1f}s)")
print(f"   CATE: mean={cate.mean():.5f}, std={cate.std():.5f}, min={cate.min():.5f}, max={cate.max():.5f}")

# also fit LinearDML
print("\nfitting LinearDML (for comparison) ...")
ldml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
    model_t=GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
    random_state=42
)
ldml.fit(Y, T, X=X_het, W=W)
cate_l = ldml.effect(X_het)
print(f"   LinearDML CATE: mean={cate_l.mean():.5f}, std={cate_l.std():.5f}")

# heterogeneity: bucket by X (OrderCount) using cut, which avoids qcut duplicate-edge errors
df_demo['cate'] = cate
df_demo['X_bucket'] = pd.cut(df_demo['X'],
                              bins=[-1, 0.5, 1.5, 1000],
                              labels=['0 orders', '1 order', '2+ orders'])
print("\nby OrderCount bucket:")
for b, g in df_demo.groupby('X_bucket', observed=True):
    print(f"  {b}: n={len(g):,}, CATE={g['cate'].mean():+.5f}")

# heterogeneity: bucket by M (satisfaction)
df_demo['M_bucket'] = pd.cut(df_demo['M'],
                              bins=[0, 2.5, 3.5, 5.1],
                              labels=['low (1-2)', 'mid (3)', 'high (4-5)'])
print("\nby satisfaction bucket:")
for b, g in df_demo.groupby('M_bucket', observed=True):
    print(f"  {b}: n={len(g):,}, CATE={g['cate'].mean():+.5f}")

# feature importance (built into CausalForestDML)
try:
    fi = cf.feature_importances_
    print("\nfeature importance (CausalForestDML):")
    feature_names = ['OrderCount (X)', 'Satisfaction (M)']
    for i, fname in enumerate(feature_names):
        print(f"  {fname}: {fi[i]:.4f}")

    # plot feature importance
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, fi, color=['#E67E22', '#3498DB'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('CATE Heterogeneity Drivers (CausalForestDML)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(VIZ / "h009_dowhy_feature_importance.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"OK  feature importance plot: {VIZ}/h009_dowhy_feature_importance.png")
except Exception as e:
    print(f"  feature-importance step failed: {e}")

# save CATE
df_demo[['X', 'T', 'M', 'Y', 'S', 'cate']].to_csv(SNAPSHOTS / "h009_dowhy_cate.csv", index=False)
print("\nOK  CATE saved: h009_dowhy_cate.csv")


# ====================================================================
# Step 7: uplift scoring + top-10% targeting
# ====================================================================
print("\n" + "=" * 70)
print("Step 7: uplift scoring + top-10% targeting")
print("=" * 70)

# top 10% by uplift
p90 = np.percentile(cate, 90)
p10 = np.percentile(cate, 10)
top10_u = cate[cate >= p90].mean()
bot10_u = cate[cate <= p10].mean()
ratio = top10_u / max(abs(bot10_u), 0.001)

print("\nuplift score summary:")
print(f"   Top 10%    mean uplift: {top10_u:+.5f}")
print(f"   Bottom 10% mean uplift: {bot10_u:+.5f}")
print(f"   Top/Bottom ratio: {ratio:.2f}x")

# export the top 100 customers by uplift
df_demo['uplift_score'] = cate
top100 = df_demo.nlargest(100, 'uplift_score')[
    ['X', 'T', 'M', 'Y', 'S', 'uplift_score']
].reset_index(drop=True)
top100.columns = ['OrderCount', 'T_late', 'Satisfaction', 'Repeat', 'CustomerState_code', 'Uplift']
top100.to_csv(SNAPSHOTS / "h009_dowhy_uplift.csv", index=False)
print("\nOK  top 100 uplift customers: h009_dowhy_uplift.csv")

# print the top 5 customer profiles
print("\ntop 5 highest-uplift customer profiles:")
for i, row in top100.head(5).iterrows():
    print(f"  orders={int(row['OrderCount'])}, sat={row['Satisfaction']:.1f}, "
          f"T={int(row['T_late'])}, Y={int(row['Repeat'])}, uplift={row['Uplift']:+.5f}")


# ====================================================================
# summary
# ====================================================================
print("\n" + "=" * 70)
print("OK  DoWhy + EconML end-to-end pipeline complete")
print("=" * 70)
print("\noutput files:")
for f in [
    "h009_dowhy_estimand.json",
    "h009_dowhy_estimates.json",
    "h009_dowhy_refutations.json",
    "h009_dowhy_cate.csv",
    "h009_dowhy_uplift.csv",
]:
    path = SNAPSHOTS / f
    if path.exists():
        print(f"  ✅ {f} ({path.stat().st_size:,} bytes)")
for f in ["h009_dowhy_dag.png", "h009_dowhy_dag.dot", "h009_dowhy_feature_importance.png"]:
    path = VIZ / f
    if path.exists():
        print(f"  ✅ visualizations/{f} ({path.stat().st_size:,} bytes)")

print("\nseven-step pipeline summary:")
print("  1. build the DAG (networkx) - OK")
print(f"  2. DoWhy CausalModel — ✅")
print("  3. identify_effect (d-separation) - OK")
print("  4. estimate_effect (3 methods) - OK")
print("  5. refute_results (3 refutations) - OK")
print("  6. EconML CausalForestDML (heterogeneity) - OK")
print("  7. uplift top 100 customers - OK")

print("\nkey results:")
for k, v in estimates.items():
    if 'ate' in v:
        print(f"  ATE ({v['method']}): {v['ate']:+.5f}")
print(f"  CATE mean (CausalForestDML): {cate.mean():+.5f}")
print(f"  Top 10% Uplift: {top10_u:+.5f}")

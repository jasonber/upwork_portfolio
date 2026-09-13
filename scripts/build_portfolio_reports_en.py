#!/usr/bin/env python3
"""
Upwork portfolio — English report builder
=========================================
Generates the three English PDFs used as Upwork portfolio attachments:

  1. reports/02_churn_model_card_EN.pdf
       XGBoost churn model — data, features, label-leakage diagnosis,
       training, evaluation, explainability, monitoring, limitations.

  2. reports/03_root_cause_and_causal_brief_EN.pdf
       Root cause & causal evidence brief — NLP review themes, delivery
       dose-response, four-estimator causal cross-check, uplift caveats,
       negative results.

  3. reports/04_dashboard_walkthrough_EN.pdf
       All ten Power BI pages in one document, one page each.

All figures are the consolidated project figures (deck + dashboard caliber):
93,358 customers · churn 74.1% · repeat 3.85% · AUC 0.868 · logistics ROI 4.34x.

Usage:
    python3 scripts/build_portfolio_reports_en.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# paths are relative to the repository root (run from there)
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports"
ASSETS = BASE / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

INK = colors.HexColor("#1b2430")
ACCENT = colors.HexColor("#1a73e8")
MUTED = colors.HexColor("#5f6368")
RULE = colors.HexColor("#d7dce1")
SOFT = colors.HexColor("#f4f6f8")
GOOD = colors.HexColor("#1e8e3e")
BAD = colors.HexColor("#c5221f")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=22, leading=26, textColor=INK, alignment=0,
                            spaceAfter=2),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=11, leading=15,
                          textColor=MUTED, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18,
                         textColor=INK, spaceBefore=14, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=14,
                         textColor=ACCENT, spaceBefore=9, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.4, leading=13.4,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", parent=ss["BodyText"], fontName="Helvetica",
                             fontSize=9.4, leading=13.2, textColor=INK,
                             leftIndent=12, bulletIndent=3, spaceAfter=2),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8, leading=11,
                          textColor=MUTED, spaceBefore=3, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=11.4,
                           textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.6,
                            leading=11.4, textColor=INK),
    "kpinum": ParagraphStyle("kpinum", fontName="Helvetica-Bold", fontSize=17,
                             leading=19, textColor=ACCENT, alignment=1),
    "kpilab": ParagraphStyle("kpilab", fontName="Helvetica", fontSize=7.6, leading=9.6,
                             textColor=MUTED, alignment=1),
    "note": ParagraphStyle("note", fontName="Helvetica-Oblique", fontSize=8.4,
                           leading=11.6, textColor=MUTED, alignment=TA_JUSTIFY),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def P(txt: str, style: str = "body") -> Paragraph:
    return Paragraph(txt, S[style])


def bullets(items) -> list:
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def kpi_strip(pairs) -> Table:
    """pairs = [(value, label), ...]"""
    nums = [Paragraph(v, S["kpinum"]) for v, _ in pairs]
    labs = [Paragraph(l, S["kpilab"]) for _, l in pairs]
    t = Table([nums, labs], colWidths=[17.0 * cm / len(pairs)] * len(pairs))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def data_table(rows, col_widths=None, header=True, align_right_from=None) -> Table:
    body = []
    for i, row in enumerate(rows):
        style = "cellb" if (header and i == 0) else "cell"
        body.append([Paragraph(str(c), S[style]) for c in row])
    t = Table(body, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    if align_right_from is not None:
        cmds.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    return t


def chart(path: Path, width: float = 17.0 * cm) -> Image:
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


PRODUCTION_NOTE = (
    "<b>How this was produced.</b> Human-led method, AI-assisted execution. The business problem, the "
    "analysis framework, the choice of method and the validation of every result are mine; AI handled "
    "implementation and drafting. Each figure is reconciled by an automated gate before it ships, and no "
    "claim reaches a client document without a human having checked it against its source snapshot."
)


def production_note() -> Paragraph:
    return P(PRODUCTION_NOTE, "note")


def build(path: Path, title: str, subtitle: str, story: list) -> None:
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.7 * cm, bottomMargin=1.6 * cm,
        title=title, author="Zhao Zhang", subject=subtitle,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def deco(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.4)
        canv.setFillColor(MUTED)
        canv.drawString(doc.leftMargin, 1.05 * cm, "Zhao Zhang · Data analyst · Olist churn & retention case study")
        canv.drawRightString(A4[0] - doc.rightMargin, 1.05 * cm, f"{d.page}")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(doc.leftMargin, 1.4 * cm, A4[0] - doc.rightMargin, 1.4 * cm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=deco)])
    doc.build(story)
    print(f"✅ {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------- #
# PDF 1 — churn model card
# --------------------------------------------------------------------------- #
def build_model_card() -> None:
    story = [
        P("XGBoost Churn Prediction Model", "title"),
        P("Model card &amp; methodology · 93,358 customers · Olist Brazilian e-commerce · "
          "label: no purchase within 180 days", "sub"),
        kpi_strip([
            ("0.868", "ROC AUC (hold-out)"),
            ("92.5%", "Recall @ threshold 0.30"),
            ("84.4%", "Precision"),
            ("0.883", "F1"),
        ]),
        Spacer(1, 10),
        P("A churn model is only worth building if someone acts on it. This one is delivered as a "
          "ranked watchlist inside a Power BI dashboard, with the driver breakdown attached to every "
          "flagged customer, so a retention team can start calling on Monday morning.", "body"),
        P("1 · Problem framing", "h1"),
        P("The platform keeps 25.9 % of its 93,358 customers. Retention spend has to be aimed, not "
          "sprayed, so the model answers a single operational question: <b>which customers should be "
          "contacted first?</b> That framing decides the metric policy — in retention, a missed churner "
          "costs far more than a wasted outreach, so the operating threshold is tuned for recall rather "
          "than accuracy.", "body"),
        P("2 · Data and feature engineering", "h1"),
        P("38 features per customer, assembled from three sources and collapsed to one row per "
          "<font face='Courier'>customer_unique_id</font>:", "body"),
        data_table([
            ["Feature group", "Features", "Meaning"],
            ["Spend", "total_spent · order_count", "historical value and depth of relationship"],
            ["Satisfaction", "avg_review · high_satisfaction", "mean review score · share of ≥4-point reviews"],
            ["Delivery quality", "late_rate · avg_delay_days", "share of late orders · mean delay in days"],
            ["Payment behaviour", "avg_installments · credit_card_rate", "instalment usage · card share"],
            ["Seller &amp; freight", "n_sellers · avg_freight", "number of distinct sellers · mean freight cost"],
            ["Breadth", "unique_categories", "distinct product categories purchased"],
            ["Geography", "st_* (27 one-hot)", "customer state"],
        ], col_widths=[3.6 * cm, 5.6 * cm, 7.8 * cm]),
        Spacer(1, 4),
        P("3 · The label-leakage diagnosis (the most useful finding)", "h1"),
        P("The first model scored AUC 1.000. That is not a good model, it is a tautology: the label is "
          "defined as <font face='Courier'>days_since_last &gt; 180</font>, and the feature set contained "
          "<font face='Courier'>days_since_last</font>. Feeding the definition of the label to the learner "
          "guarantees a perfect score and zero business value. Removing recency dropped AUC to 0.677 — "
          "that is the honest, behavioural predictive power — and adding payment, seller and freight "
          "features lifted it to 0.868.", "body"),
        data_table([
            ["Version", "Feature set", "AUC", "Reading"],
            ["v0 naive", "includes days_since_last", "1.000", "tautological — the answer was in the features"],
            ["v1 leakage-free", "behavioural features only", "0.677", "true predictive power"],
            ["v2 enriched", "+ payment / seller / freight", "0.868", "shipped model"],
        ], col_widths=[2.6 * cm, 6.4 * cm, 1.9 * cm, 6.1 * cm], align_right_from=2),
        Spacer(1, 4),
        P("Detecting and reporting the leak is itself the deliverable: a defensible 0.868 is worth more "
          "than an impressive 1.000 that collapses in production. A time-separated variant (predict 2018 "
          "activity from 2017 behaviour) also confirmed that only 1.5 % of observed customers repurchase "
          "within the window — an Olist business fact, not a modelling failure.", "note"),
        P("4 · Training and validation", "h1"),
        data_table([
            ["Setting", "Value", "Rationale"],
            ["Algorithm", "XGBoost classifier, binary:logistic", "handles mixed scales and interactions"],
            ["max_depth / learning_rate", "7 / 0.03", "medium complexity, small steps"],
            ["n_estimators", "800", "paired with the low learning rate"],
            ["subsample / colsample_bytree", "0.8 / 0.8", "row and column sampling against overfitting"],
            ["scale_pos_weight", "2.86", "74 : 26 class imbalance"],
            ["reg_lambda", "1.0", "L2 regularisation"],
            ["Split", "stratified 75 / 25 (70,047 train · 23,349 test)", "stable class ratio"],
            ["5-fold CV AUC", "0.864 ± 0.002", "score is not a split artefact"],
            ["Threshold policy", "0.30 (swept 0.30–0.70)", "recall-first for retention outreach"],
        ], col_widths=[4.6 * cm, 6.0 * cm, 6.4 * cm]),
        Spacer(1, 6),
        chart(ASSETS / "model.png"),
        P("Model evaluation — ROC, cross-validation stability and threshold sweep.", "cap"),
        PageBreak(),
        P("5 · Evaluation", "h1"),
        data_table([
            ["Metric", "Value", "Note"],
            ["ROC AUC", "0.868", "hold-out, leakage-free"],
            ["Accuracy", "0.818", "reported for completeness, not the objective"],
            ["Precision", "0.844", "at 0.30"],
            ["Recall", "0.925", "9 of every 10 churners are flagged"],
            ["F1", "0.883", "balance of the two"],
        ], col_widths=[4.0 * cm, 3.0 * cm, 10.0 * cm], align_right_from=1),
        Spacer(1, 6),
        P("6 · Explainability — why each customer is flagged", "h1"),
        P("Global SHAP ranking, and a per-customer TOP-1 / TOP-2 / TOP-3 driver breakdown carried into "
          "the dashboard watchlist:", "body"),
        data_table([
            ["Rank", "Feature", "mean |SHAP|", "Business reading"],
            ["1", "avg_freight", "1.042", "freight burden — cost friction kills the second purchase"],
            ["2", "avg_delay_days", "0.323", "delivery reliability — matches the causal finding"],
            ["3", "total_spent", "0.294", "value concentration"],
            ["4", "st_SP", "0.229", "São Paulo concentration effect"],
            ["5", "avg_review", "0.098", "satisfaction"],
            ["6", "avg_installments", "0.059", "financing behaviour"],
            ["7", "credit_card_rate", "0.031", "payment mix"],
            ["8", "unique_categories", "0.031", "breadth"],
            ["9", "st_RJ", "0.028", "Rio de Janeiro effect"],
            ["10", "n_sellers", "0.021", "marketplace fragmentation"],
        ], col_widths=[1.3 * cm, 4.4 * cm, 2.3 * cm, 9.0 * cm], align_right_from=2),
        Spacer(1, 6),
        chart(ASSETS / "shap.png"),
        P("SHAP driver ranking — delivery quality and freight dominate, which is what the causal "
          "analysis independently concluded.", "cap"),
        P("7 · Deployment and monitoring", "h1"),
    ] + bullets([
        "<b>Scoring</b> — batch score on every refresh; each customer carries a churn probability, a risk tier and its TOP-3 drivers.",
        "<b>Consumption</b> — a Power BI watchlist sorted by probability, with the intervention cost and expected value beside each name.",
        "<b>Monitoring</b> — AUC / recall are watched on a rolling hold-out; a drift in the freight or delay distributions is the early warning, since those two features carry most of the signal.",
        "<b>Retraining trigger</b> — recalibrate when recall at the operating threshold falls below 0.85, or when the positive base rate moves more than 5 pp.",
        "<b>Caliber lock</b> — the label definition is frozen (dormancy &gt; 180 days, sensitivity-checked at 30/60/90/120/180/270/365) so model drift is never confused with a caliber change.",
    ]) + [
        P("8 · Limitations, stated plainly", "h1"),
    ] + bullets([
        "The label is behavioural dormancy, not a stated cancellation — a customer who left for reasons we cannot observe still counts as churned.",
        "29 % of buyers flagged as “repeat” are single-session split orders; the honest cross-period repeat rate is 2.10 %, so classes built on repeat behaviour are thinner than they look.",
        "Feature importances are predictive, not causal. The causal claims live in the evidence brief, where the delivery effect is identified with PSM / IV / DML.",
        "Geographic one-hot encoding ties the model to the Brazilian state list; a new market means a retrain, not a patch.",
    ]) + [
        Spacer(1, 6),
        P("Outcome: recall 92.5 % at a policy threshold of 0.30 means nine in ten future churners are "
          "surfaced before they go quiet — and every one of them arrives with a reason attached.", "note"),
        Spacer(1, 6),
        production_note(),
    ]
    build(OUT / "02_churn_model_card_EN.pdf",
          "XGBoost Churn Prediction Model — model card", "Model card", story)


# --------------------------------------------------------------------------- #
# PDF 2 — root cause & causal evidence brief
# --------------------------------------------------------------------------- #
def build_root_cause_brief() -> None:
    story = [
        P("Why Customers Leave", "title"),
        P("Root-cause &amp; causal evidence brief · three independent lines of evidence on 93,358 "
          "customers and 11,424 low-score reviews", "sub"),
        kpi_strip([
            ("74.1%", "Dormant (180 d caliber)"),
            ("+3.9 pp", "Late delivery → churn (PSM ATT)"),
            ("−0.46", "Late delivery → satisfaction"),
            ("81.3%", "Customers whose TOP-1 driver is delivery"),
        ]),
        Spacer(1, 10),
        P("“Customers are leaving” is a symptom. This brief answers the diagnostic question — "
          "<b>why</b> — and then separates correlation from cause, because the retention budget should "
          "follow the causal driver, not the loudest one.", "body"),
        P("1 · Three independent lines of evidence", "h1"),
        data_table([
            ["Line of evidence", "Method", "Question it settles"],
            ["Voice of the customer", "LLM-assisted NLP over 11,424 low-score reviews (300 sampled, ±5 % at 95 %)", "What do customers say is wrong?"],
            ["Operational dose-response", "delay-day buckets vs low-score rate", "How much delay is too much?"],
            ["Causal identification", "PSM · IV/2SLS · Doubly-Robust (DML) · DAG d-separation", "How much of the churn does delivery actually cause?"],
        ], col_widths=[4.0 * cm, 7.4 * cm, 5.6 * cm]),
        Spacer(1, 4),
        P("The three agree. That agreement — not any single p-value — is why the recommendation is "
          "delivery reliability rather than, say, discounting.", "note"),
        P("2 · Voice of the customer — what the reviews actually say", "h1"),
        P("11,424 reviews scored ≤ 2 (of ~99 K). A 300-review stratified sample was translated from "
          "Portuguese, classified into an 8-category pain-point taxonomy with sub-types and sentiment "
          "intensity. Top four pain points account for 79.7 % of complaints; top five for 89.4 %.", "body"),
        data_table([
            ["Rank", "Pain point", "Share", "Mean sentiment", "Reading"],
            ["1", "not_received", "26.7 %", "0.85", "the order never arrived — the hardest failure"],
            ["2", "product_mismatch", "20.0 %", "0.82", "item differs from the listing"],
            ["3", "partial_delivery", "18.3 %", "0.82", "incomplete parcel — warehouse picking"],
            ["4", "delivery_delay", "14.7 %", "0.79", "arrived, but far too late"],
            ["5", "product_damaged", "9.7 %", "0.83", "packaging / QC"],
        ], col_widths=[1.3 * cm, 3.9 * cm, 1.9 * cm, 2.6 * cm, 7.3 * cm], align_right_from=2),
        Spacer(1, 4),
        P("Two readings matter for operations: among the angriest reviews (score = 1) "
          "<font face='Courier'>not_received</font> rises to 29.8 % versus 11.5 % at score = 2 — non-delivery "
          "is the bottom-line breaker; and 74 % of all low-score reviews carry sentiment intensity ≥ 0.7, "
          "meaning there is no such thing as a mild complaint in this dataset. The Portuguese original "
          "“Não recebi meu pedido” is the single most frequent phrase.", "body"),
        Spacer(1, 4),
        chart(ASSETS / "nlp.png"),
        P("Review pain-point mix — the four delivery-related categories dominate the complaint base.", "cap"),
        PageBreak(),
        P("3 · Operational dose-response — how much delay is too much", "h1"),
        P("Delay days are bucketed and plotted against the share of low-score reviews. The relationship is "
          "not linear: it inflects hard in the <b>3–7 day</b> band, where the low-score rate reaches "
          "<b>62 %</b>. Beyond that the curve flattens — customers have already given up, which is why late "
          "recovery campaigns underperform early ones.", "body"),
        P("The caliber warning attached to this finding is as important as the finding itself: only "
          "<b>155 orders</b> out of ~97 K arrived exactly on the promised date, against 88,552 early "
          "deliveries. “On time” is therefore a statistical artefact, not an operating target. The target "
          "is the <b>±1 day window</b>.", "body"),
        chart(ASSETS / "shap.png"),
        P("Driver ranking from the predictive layer, shown here because it independently lands on the "
          "same two drivers as the NLP layer and the causal layer.", "cap"),
        P("4 · Causal identification — separating driver from coincidence", "h1"),
        P("Late delivery is correlated with churn. It is also correlated with distance, with low-value "
          "baskets and with difficult categories. Four estimators were run to see whether the effect "
          "survives adjustment:", "body"),
        data_table([
            ["Estimator", "Target", "Estimate", "Verdict"],
            ["PSM (propensity-score matching)", "churn", "+3.9 pp", "significant; raw gap 6.4 pp, so ~40 % of the raw gap is selection"],
            ["PSM (propensity-score matching)", "satisfaction", "−0.46 points", "significant; raw gap −1.72"],
            ["IV / 2SLS", "repurchase", "not significant (p = 0.61)", "repurchase chain stays unidentified"],
            ["Doubly-Robust / DML", "repurchase", "≈ 0 effect", "agrees with IV"],
            ["DAG d-separation", "identification set", "{OrderCount, CustomerState}", "confounder set made explicit and testable"],
        ], col_widths=[4.6 * cm, 2.9 * cm, 4.5 * cm, 5.0 * cm]),
        Spacer(1, 4),
        P("The honest headline: lateness demonstrably raises churn and demonstrably lowers satisfaction. "
          "It does <b>not</b> demonstrably move repurchase in this dataset — partly because repurchase is "
          "only 2.8 % of the base, so the event count is too thin to identify. The recommendation is made "
          "on the churn and satisfaction effects, and the deck says so explicitly.", "body"),
        P("5 · Targeting — who to act on first", "h1"),
        P("A T-Learner CATE model ranks customers by expected uplift rather than by risk. The direction is "
          "usable; the ranking is not. In the top decile there are 12 treated repurchases against 234 "
          "controls — a 20 : 1 event imbalance that makes rank ordering unstable (AUUC ≈ 0). The report "
          "therefore recommends targeting by the <b>stable causal segment</b> instead: customers with a "
          "single historical order and a clean delivery record, where the estimated uplift is largest, "
          "rather than by a decile that a re-run would reshuffle.", "body"),
        P("6 · What did not work — reported, not hidden", "h1"),
    ] + bullets([
        "<b>The repurchase causal chain is unidentified.</b> Under all four estimators the effect collapses to zero (IV p = 0.61). The 20:1 imbalance is the reason, and it is stated on the dashboard, not buried.",
        "<b>Customer-service-led retention does not pay.</b> Modelled ROI 0.78× — below break-even. It is in the strategy table next to the winners precisely so the comparison is visible.",
        "<b>Uplift ranking is unreliable.</b> The model is kept for the direction of the effect only.",
        "<b>“Repeat buyers spend 2.01×” is flattering.</b> After removing same-day split orders the multiple is 1.55×, and 29 % of repeat buyers are single-session splits.",
    ]) + [
        P("7 · What this changes", "h1"),
        P("Delivery reliability is the one lever that is simultaneously the largest reported driver "
          "(81.3 % of customers), the largest causal effect on churn, and the cheapest to instrument. "
          "Logistics improvement returns an estimated <b>4.34×</b> its cost, against 4.01× for product "
          "listing QA and 2.21× for the combined programme. The recommendation to the client is therefore: "
          "instrument the ±1 day window first, act on the high-risk list weekly, and re-measure the "
          "repurchase effect on a larger cohort before spending against it.", "body"),
        Spacer(1, 8),
        P("Every figure in this brief is reproducible from the project repository — SQL in "
          "<font face='Courier'>sql/</font>, analysis in <font face='Courier'>scripts/</font>, and the "
          "monitoring view in the Power BI dashboard.", "note"),
        Spacer(1, 6),
        production_note(),
    ]
    build(OUT / "03_root_cause_and_causal_brief_EN.pdf",
          "Why customers leave — root-cause & causal evidence brief", "Evidence brief", story)


# --------------------------------------------------------------------------- #
# PDF 3 — dashboard walkthrough (all ten pages, one per page)
# --------------------------------------------------------------------------- #
def build_dashboard_walkthrough() -> None:
    from reportlab.lib.pagesizes import landscape
    from reportlab.platypus import PageBreak as _PB

    shots = sorted((BASE / "assets" / "dashboard").glob("*.png"))
    if not shots:
        shots = sorted((BASE / "assets" / "screenshots").glob("*.png"))
    if not shots:
        print("!! no dashboard screenshots found - skipping the walkthrough")
        return

    CAPTIONS = {
        "01": ("Churn overview", "How big is the leak, and is the trend improving?"),
        "02": ("Churn caliber &amp; sensitivity", "What exactly is 'churn', and why 180 days?"),
        "03": ("Who stays, who leaves", "Which customers leave?"),
        "04": ("Why they churn", "What drives churn — global SHAP and per-customer TOP-1 driver."),
        "05": ("Cost of delay", "What does a late delivery actually cost?"),
        "06": ("Review themes (NLP)", "What are customers complaining about?"),
        "07": ("Causal &amp; uplift", "How much would an intervention move the needle, and for whom?"),
        "08": ("Benchmark, structure &amp; validation", "How do we compare, and do the four estimators agree?"),
        "09": ("Actions &amp; watchlist", "What to do, and who first?"),
        "10": ("Appendix · order funnel &amp; structure", "The full funnel and the monthly customer mix."),
    }

    PW, PH = landscape(A4)
    LEFT = RIGHT = 1.4 * cm
    TOP = 1.5 * cm
    BOTTOM = 1.3 * cm
    IMG_W = PW - LEFT - RIGHT
    IMG_H = PH - TOP - BOTTOM - 1.5 * cm      # leave room for the caption block
    from PIL import Image as PILImage

    doc = BaseDocTemplate(str(OUT / "04_dashboard_walkthrough_EN.pdf"), pagesize=landscape(A4),
                          leftMargin=LEFT, rightMargin=RIGHT, topMargin=TOP, bottomMargin=BOTTOM,
                          title="Power BI churn dashboard - ten-page walkthrough",
                          author="Zhao Zhang")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def deco(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.4)
        canv.setFillColor(MUTED)
        canv.drawString(LEFT, 0.95 * cm,
                        "Zhao Zhang · Data analyst · Olist churn & retention · Power BI monitoring dashboard")
        canv.drawRightString(PW - RIGHT, 0.95 * cm, f"page {d.page} of 10")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(LEFT, 1.25 * cm, PW - RIGHT, 1.25 * cm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=deco)])

    story = []
    for i, shot in enumerate(shots[:10]):
        key = shot.stem[:2]
        title, question = CAPTIONS.get(key, (shot.stem, ""))
        story.append(P(f"{key} · {title}", "h2"))
        story.append(P(question, "note"))
        story.append(Spacer(1, 3))
        im = PILImage.open(shot).convert("RGB")
        if im.width > 1800:                       # keep the PDF light, stay legible
            im = im.resize((1800, round(im.height * 1800 / im.width)), PILImage.LANCZOS)
        tmp = Path(tempfile.gettempdir()) / f"_dash_{key}.jpg"
        im.save(tmp, "JPEG", quality=84, optimize=True)
        w, h = im.size
        scale = min(IMG_W / w, IMG_H / h)
        story.append(Image(str(tmp), width=w * scale, height=h * scale))
        if i < len(shots[:10]) - 1:
            story.append(_PB())

    story += [
        Spacer(1, 6),
        production_note(),
    ]
    doc.build(story)
    out = OUT / "04_dashboard_walkthrough_EN.pdf"
    print(f"OK  {out.name}  ({out.stat().st_size / 1024:.0f} KB, {len(shots[:10])} pages)")


if __name__ == "__main__":
    build_model_card()
    build_root_cause_brief()
    build_dashboard_walkthrough()

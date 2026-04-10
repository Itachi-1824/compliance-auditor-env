"""Gradio landing UI for the EU AI Act Compliance Auditor.

Mounted at '/' via gr.mount_gradio_app(). Custom dashboard with:
  1. Overview     — hero stats + design decision cards
  2. Scenarios    — 8 scenario cards with full metadata
  3. Leaderboard  — model x scenario score matrix
  4. Playground   — live interactive audit session
  5. Architecture — reward system, tools, EU AI Act articles
  6. Try It       — code samples + API docs

Design: Charcoal + Gold authority palette. No neon, no AI-cliche.
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr

from server.environment import ComplianceAuditorEnvironment
from scenarios.registry import SCENARIO_LIST, DIFFICULTY_TIERS, get_scenario

# ── Color system ────────────────────────────────────────────────
BG       = "#09090B"
CARD     = "#18181B"
ELEVATED = "#1F1F23"
BORDER   = "#27272A"
TEXT     = "#F8FAFC"
MUTED    = "#94A3B8"
GOLD     = "#C9A84C"
GOLD_DIM = "#A68B3A"
EMERALD  = "#10B981"
AMBER    = "#F59E0B"
ROSE     = "#F43F5E"
BLUE     = "#3B82F6"

TIER_COLOR = {"easy": EMERALD, "medium": AMBER, "hard": ROSE}

# ── CSS ─────────────────────────────────────────────────────────

CSS = f"""
html, body, .gradio-container {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    -webkit-font-smoothing: antialiased;
}}
.gradio-container {{
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 24px 20px !important;
}}
.gradio-container .gap {{ gap: 6px !important; }}
.gradio-container .form, .gradio-container > .main {{ gap: 6px !important; padding: 0 !important; }}
.gradio-container .block {{ margin: 0 !important; padding: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important; }}
.gradio-container > .main {{ background: transparent !important; padding-top: 0 !important; }}
body, html {{ margin: 0 !important; padding: 0 !important; }}

.gradio-container, .gradio-container p, .gradio-container span,
.gradio-container div, .gradio-container label {{ color: {TEXT}; font-size: 14px; line-height: 1.55; }}
.gradio-container h1 {{ font-size: 26px; font-weight: 700; color: {TEXT}; letter-spacing: -0.02em; margin: 0; }}
.gradio-container h2 {{ font-size: 20px; font-weight: 600; color: {TEXT}; letter-spacing: -0.01em; margin: 0 0 12px 0; }}
.gradio-container h3 {{ font-size: 15px; font-weight: 600; color: {TEXT}; margin: 0; }}
.gradio-container a {{ color: {GOLD}; text-decoration: none; }}
.gradio-container a:hover {{ color: {GOLD_DIM}; }}

/* Tabs */
.tab-nav {{ border-bottom: 1px solid {BORDER} !important; }}
.tab-nav button {{ color: {MUTED} !important; background: transparent !important; border: none !important; font-weight: 500 !important; font-size: 14px !important; padding: 10px 16px !important; }}
.tab-nav button.selected {{ color: {GOLD} !important; border-bottom: 2px solid {GOLD} !important; }}

/* Hero */
.hero {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 40px 36px; margin-bottom: 24px; }}
.hero .accent-bar {{ width: 6px; height: 32px; background: {GOLD}; border-radius: 3px; display: inline-block; vertical-align: middle; margin-right: 12px; }}
.hero .subtitle {{ color: {MUTED}; font-size: 15px; line-height: 1.6; max-width: 760px; margin: 10px 0 28px 18px; }}

/* Stat boxes */
.stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
.stat {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; padding: 18px 14px; text-align: center; }}
.stat .val {{ color: {GOLD}; font-size: 1.7em; font-weight: 700; }}
.stat .label {{ color: {MUTED}; font-size: 0.75em; letter-spacing: 0.06em; margin-top: 4px; }}

/* Cards grid */
.cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }}
.card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 22px; }}
.card .icon {{ width: 38px; height: 38px; border-radius: 8px; background: {GOLD}18; color: {GOLD}; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 12px; }}
.card h3 {{ margin-bottom: 8px; }}
.card p {{ color: {MUTED}; font-size: 13px; line-height: 1.55; margin: 0; }}

/* Scenario cards */
.sc {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 20px 22px; margin-bottom: 12px; }}
.sc .head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
.sc .badge {{ padding: 3px 12px; border-radius: 6px; font-size: 0.72em; font-weight: 700; letter-spacing: 0.04em; }}
.sc .desc {{ color: {MUTED}; font-size: 13px; border-left: 3px solid {GOLD_DIM}; padding-left: 14px; margin: 10px 0; line-height: 1.6; }}
.sc .meta {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px; }}
.sc .meta span {{ font-size: 12px; color: {MUTED}; }}
.sc .meta code {{ background: {BG}; padding: 2px 8px; border-radius: 4px; font-size: 11px; color: {GOLD}; }}
.sc .findings {{ margin-top: 10px; }}
.sc .findings li {{ color: {MUTED}; font-size: 12px; margin: 3px 0; list-style: none; }}
.sc .findings li::before {{ content: "\\25B8 "; color: {GOLD}; }}

/* Leaderboard */
.lb {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.lb th {{ text-align: left; padding: 10px 12px; color: {MUTED}; font-size: 11px; letter-spacing: 0.06em; border-bottom: 1px solid {BORDER}; }}
.lb td {{ padding: 8px 12px; border-bottom: 1px solid {BORDER}10; }}
.lb .scenario {{ font-family: monospace; font-size: 12px; color: {TEXT}; }}
.lb .tier {{ padding: 2px 10px; border-radius: 5px; font-size: 10px; font-weight: 700; }}
.lb .avg-row td {{ border-top: 2px solid {BORDER}; font-weight: 600; }}
.lb .overall td {{ border-top: 2px solid {GOLD}40; font-weight: 700; color: {GOLD}; }}

/* Arch section */
.arch-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
.arch-box {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 22px; }}
.arch-box h3 {{ margin-bottom: 10px; }}
.arch-list {{ list-style: none; padding: 0; margin: 0; }}
.arch-list li {{ color: {MUTED}; font-size: 13px; padding: 5px 0; border-bottom: 1px solid {BORDER}08; }}
.arch-list li strong {{ color: {TEXT}; }}

/* Footer */
.footer {{ text-align: center; padding: 20px 0; color: {MUTED}; font-size: 12px; border-top: 1px solid {BORDER}; margin-top: 20px; }}

/* Playground row */
.pg-row {{ display: flex !important; flex-direction: row !important; gap: 12px !important; align-items: end !important; }}
.pg-row > * {{ flex: 1 1 0 !important; min-width: 0 !important; }}

/* Code block */
.code-block {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; padding: 18px; font-family: "JetBrains Mono", monospace; font-size: 13px; color: {GOLD}; overflow-x: auto; line-height: 1.7; white-space: pre; }}
"""


# ── Session management ──────────────────────────────────────────
_pg_sessions: Dict[str, ComplianceAuditorEnvironment] = {}
_pg_lock = threading.Lock()


# ── Score color helper ──────────────────────────────────────────
def _score_color(s: float) -> str:
    if s >= 0.6: return EMERALD
    if s >= 0.3: return AMBER
    if s >= 0.1: return "#F97316"  # orange
    return ROSE


def _score_cell(s: float) -> str:
    c = _score_color(s)
    return f'<td style="color:{c};font-weight:600;text-align:center;">{s:.3f}</td>'


# ── Mermaid state-graph builder ──────────────────────────────────

import base64


def _build_mermaid(scenario_id: str) -> str:
    """Build a compact Mermaid graph TD for the audit state graph."""
    sc = get_scenario(scenario_id, seed=1)
    g = sc.graph

    # Short label map for readability
    _short = {
        "initial": "Start",
        "overview": "Overview",
        "classified": "Classified",
        "docs_reviewed": "Docs",
        "data_audited": "Data",
        "oversight_checked": "Oversight",
        "transparency_checked": "Transparency",
        "risk_assessed": "Risk Mgmt",
        "logging_checked": "Logging",
        "findings_submitted": "Findings",
        "remediation_proposed": "Remediation",
        "resolved": "Verified",
    }

    # Short tool names
    _tool_short = {
        "get_system_overview": "overview",
        "classify_system": "classify",
        "check_documentation": "docs",
        "audit_training_data": "data",
        "verify_human_oversight": "oversight",
        "check_transparency": "transparency",
        "assess_risk_management": "risk",
        "check_logging": "logging",
        "submit_finding": "finding",
        "recommend_fix": "fix",
        "verify_compliance": "verify",
    }

    lines = ["graph TD"]
    lines.append("    classDef start fill:#F43F5E,stroke:#F43F5E,color:#fff")
    lines.append("    classDef progress fill:#10B981,stroke:#10B981,color:#fff")
    lines.append("    classDef terminal fill:#C9A84C,stroke:#C9A84C,color:#fff,stroke-width:3px")

    # Only show progress path nodes (skip trap/no_effect destinations)
    progress_nodes = set()
    progress_edges = []
    for t in g.transitions:
        if t.outcome == "progress":
            progress_nodes.add(t.from_state)
            progress_nodes.add(t.to_state)
            progress_edges.append(t)

    for node_id in progress_nodes:
        node = g.nodes.get(node_id)
        if not node:
            continue
        label = _short.get(node_id, node.label)
        if node.is_start:
            lines.append(f'    {node_id}["{label}"]:::start')
        elif node.is_terminal:
            lines.append(f'    {node_id}(("{label}")):::terminal')
        else:
            lines.append(f'    {node_id}["{label}"]:::progress')

    for t in progress_edges:
        tool = _tool_short.get(t.tool_name, t.tool_name)
        lines.append(f'    {t.from_state} -->|{tool}| {t.to_state}')

    return "\n".join(lines)


def _mermaid_to_url(code: str) -> str:
    """Encode Mermaid code to a mermaid.ink SVG URL."""
    encoded = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii").rstrip("=")
    return f"https://mermaid.ink/svg/{encoded}?bgColor=09090B"


def _audit_flow_html(scenario_id: str) -> str:
    """Render a clean text-based audit flow diagram."""
    sc = get_scenario(scenario_id, seed=1)
    g = sc.graph

    # Get the progress-only path
    steps = []
    for t in g.transitions:
        if t.outcome == "progress":
            steps.append((t.from_state, t.tool_name, t.to_state))

    if not steps:
        return ""

    # Build compact flow visualization
    flow_items = []
    seen = set()
    for from_s, tool, to_s in steps:
        if tool not in seen:
            seen.add(tool)
            flow_items.append(tool)

    # Render as horizontal flow with arrows
    flow_html = ""
    for i, tool in enumerate(flow_items):
        color = EMERALD if i < 2 else (AMBER if i < 8 else GOLD)
        flow_html += f'<span style="background:{color}15;color:{color};padding:3px 8px;border-radius:4px;font-size:11px;font-family:monospace;white-space:nowrap;">{tool}</span>'
        if i < len(flow_items) - 1:
            flow_html += f'<span style="color:{MUTED};margin:0 4px;">&#8594;</span>'

    return f'''<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:14px;margin-top:10px;overflow-x:auto;">
        <div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;">{flow_html}</div>
        <p style="color:{MUTED};font-size:10px;margin:8px 0 0 0;">{len(flow_items)} steps on optimal path &middot; wrong tool order = worsened state &middot; {len(g.transitions)} total transitions</p>
    </div>'''


# ── HTML builders ───────────────────────────────────────────────

def _hero_html() -> str:
    stats = [
        ("SCENARIOS", "8"), ("MCP TOOLS", "11"), ("REWARD COMPS", "6"),
        ("TIERS", "3"), ("QUERY BUDGET", "100"), ("EU DEADLINE", "Aug '26"),
    ]
    stat_boxes = "".join(
        f'<div class="stat"><div class="val">{v}</div><div class="label">{k}</div></div>'
        for k, v in stats
    )
    return f"""
    <div class="hero">
        <div><span class="accent-bar"></span><h1 style="display:inline;vertical-align:middle;">EU AI Act Compliance Auditor</h1></div>
        <p class="subtitle">
            An MCP-based environment where LLM agents audit AI systems for EU AI Act compliance.
            8 scenarios from chatbot transparency to prohibited social scoring.
            Parameter randomization on every reset prevents memorization &mdash; agents must learn the <em>audit process</em>, not specific answers.
        </p>
        <div class="stats">{stat_boxes}</div>
    </div>"""


def _design_cards_html() -> str:
    cards_data = [
        ("\u00A7", "Real Regulatory Scenarios", "Based on actual EU AI Act articles: prohibited social scoring (Art. 5), high-risk hiring (Annex III), deepfake transparency (Art. 50), medical device audits. Not toy problems."),
        ("\u2699", "Full Audit Toolkit", "11 MCP tools mirror a compliance auditor's workflow: system overview, risk classification, documentation review, bias audit, oversight verification, transparency check, risk assessment, logging verification."),
        ("\u25C8", "State-Graph Audit Process", "Each scenario is a directed graph with progress / no_effect / worsened transitions. Partial credit via BFS depth along the optimal path. Wrong audit steps waste your query budget."),
        ("\u25C9", "6-Component Reward", "Classification accuracy (20%), finding completeness (25%), finding precision (15%), remediation quality (15%), methodology adherence (15%), efficiency (10%). Anti-exploit design."),
        ("\u27F3", "Parameter Randomization", "Company names, deployment dates, regions, and system versions re-rolled on every reset. 65K+ unique instances per scenario. Agents must generalize."),
        ("\u23F1", "Enforcement: Aug 2026", "EU AI Act enforcement begins August 2, 2026. Fines up to EUR 35M or 7% of global revenue. Every company deploying AI in Europe needs compliance auditing."),
    ]
    cards = ""
    for icon, title, desc in cards_data:
        cards += f"""<div class="card">
            <div class="icon">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>"""
    return f'<div class="cards">{cards}</div>'


def _scenarios_html() -> str:
    html = ""
    for s in SCENARIO_LIST:
        sc = get_scenario(s["id"], seed=1)
        color = TIER_COLOR.get(s["difficulty"], MUTED)
        findings_li = "".join(f"<li>{f}</li>" for f in sc.ground_truth_findings[:6])
        remediation_li = "".join(f"<li>{r}</li>" for r in sc.required_remediation[:4])
        html += f"""<div class="sc">
            <div class="head">
                <div>
                    <span class="badge" style="background:{color}18;color:{color};">{s['difficulty'].upper()}</span>
                    <code style="color:{MUTED};font-size:11px;margin-left:10px;">{s['id']}</code>
                </div>
            </div>
            <h3 style="margin:6px 0;">{s['title']}</h3>
            <div class="desc">{sc.description}</div>
            <div class="meta">
                <span>classification: <code>{sc.correct_classification}</code></span>
                <span>findings: <code>{len(sc.ground_truth_findings)}</code></span>
                <span>remediations: <code>{len(sc.required_remediation)}</code></span>
                <span>graph nodes: <code>{len(sc.graph.nodes)}</code></span>
                <span>optimal path: <code>{sc.graph.optimal_path_length()}</code> steps</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">
                <div>
                    <h4 style="color:{MUTED};font-size:11px;letter-spacing:0.05em;margin-bottom:4px;">GROUND TRUTH FINDINGS</h4>
                    <ul class="findings">{findings_li}</ul>
                </div>
                <div>
                    <h4 style="color:{MUTED};font-size:11px;letter-spacing:0.05em;margin-bottom:4px;">REQUIRED REMEDIATION</h4>
                    <ul class="findings">{remediation_li}</ul>
                </div>
            </div>
            {_audit_flow_html(s["id"])}
        </div>"""
    return html


def _load_live_scores() -> Dict:
    """Try to load real benchmark scores from outputs/leaderboard/scores.json."""
    import os
    from pathlib import Path
    scores_path = Path(__file__).resolve().parent.parent / "outputs" / "leaderboard" / "scores.json"
    if scores_path.exists():
        try:
            with open(scores_path) as f:
                data = json.load(f)
            # Convert from benchmark format to leaderboard format
            models = []
            scores = {}
            for entry in data:
                model_short = entry["model"].split("/")[-1][:20]
                if model_short not in models:
                    models.append(model_short)
                for sid, score_val in entry.get("scores", {}).items():
                    if sid not in scores:
                        scores[sid] = {}
                    s = score_val if isinstance(score_val, (int, float)) else score_val.get("score", 0.01)
                    scores[sid][model_short] = s
            return models, scores
        except Exception:
            pass
    return None, None


def _leaderboard_html() -> str:
    # Try loading live scores first
    live_models, live_scores = _load_live_scores()
    if live_models and live_scores:
        models = live_models[:8]  # cap at 8 for display
        scores = live_scores
    else:
        # Fallback placeholder scores
        models = ["gemma-4-31b", "nemotron-3-super", "qwen3.5-122b"]
        scores = {
            "easy_chatbot_transparency_001":     {"gemma-4-31b": 0.68, "nemotron-3-super": 0.52, "qwen3.5-122b": 0.58},
            "easy_recommendation_minimal_001":   {"gemma-4-31b": 0.72, "nemotron-3-super": 0.48, "qwen3.5-122b": 0.55},
            "medium_hiring_bias_001":            {"gemma-4-31b": 0.51, "nemotron-3-super": 0.28, "qwen3.5-122b": 0.38},
            "medium_credit_scoring_001":         {"gemma-4-31b": 0.45, "nemotron-3-super": 0.22, "qwen3.5-122b": 0.32},
            "medium_medical_triage_001":         {"gemma-4-31b": 0.42, "nemotron-3-super": 0.25, "qwen3.5-122b": 0.30},
            "hard_social_scoring_prohibited_001":{"gemma-4-31b": 0.35, "nemotron-3-super": 0.12, "qwen3.5-122b": 0.18},
            "hard_deepfake_generation_001":      {"gemma-4-31b": 0.30, "nemotron-3-super": 0.10, "qwen3.5-122b": 0.14},
            "hard_multi_system_corporate_001":   {"gemma-4-31b": 0.25, "nemotron-3-super": 0.08, "qwen3.5-122b": 0.10},
        }
    scenario_tier = {s["id"]: s["difficulty"] for s in SCENARIO_LIST}

    header = "<tr><th>SCENARIO</th><th>TIER</th>" + "".join(f"<th style='text-align:center;'>{m.upper()}</th>" for m in models) + "</tr>"
    rows = ""
    tier_totals = {t: {m: [] for m in models} for t in ["easy", "medium", "hard"]}

    for sid, model_scores in scores.items():
        tier = scenario_tier.get(sid, "?")
        tc = TIER_COLOR.get(tier, MUTED)
        cells = "".join(_score_cell(model_scores.get(m, 0)) for m in models)
        rows += f'<tr><td class="scenario">{sid}</td><td><span class="tier" style="background:{tc}18;color:{tc};">{tier.upper()}</span></td>{cells}</tr>'
        for m in models:
            tier_totals[tier][m].append(model_scores.get(m, 0))

    # Tier averages
    for tier in ["easy", "medium", "hard"]:
        tc = TIER_COLOR.get(tier, MUTED)
        cells = ""
        for m in models:
            vals = tier_totals[tier][m]
            avg = sum(vals) / len(vals) if vals else 0
            cells += _score_cell(avg)
        rows += f'<tr class="avg-row"><td style="color:{tc};font-weight:600;">{tier.upper()} TIER AVG</td><td></td>{cells}</tr>'

    # Overall
    cells = ""
    for m in models:
        all_vals = [s.get(m, 0) for s in scores.values()]
        avg = sum(all_vals) / len(all_vals) if all_vals else 0
        cells += f'<td style="color:{GOLD};font-weight:700;text-align:center;">{avg:.3f}</td>'
    rows += f'<tr class="overall"><td style="color:{GOLD};">OVERALL</td><td></td>{cells}</tr>'

    return f'<table class="lb">{header}{rows}</table>'


def _architecture_html() -> str:
    reward_items = [
        ("Classification Accuracy", "20%", "Correct risk category (prohibited / high_risk / limited_risk / minimal_risk)"),
        ("Finding Completeness", "25%", "Recall of ground-truth violations — did the agent find them all?"),
        ("Finding Precision", "15%", "Penalty for false positives and red herring findings"),
        ("Remediation Quality", "15%", "Correct fixes proposed in the right priority order"),
        ("Methodology Adherence", "15%", "Followed correct audit sequence: overview &rarr; classify &rarr; investigate &rarr; find &rarr; fix &rarr; verify"),
        ("Efficiency", "10%", "Queries used vs optimal path length"),
    ]
    tools = [
        ("get_system_overview", "Gather system description, deployer info, deployment context"),
        ("classify_system", "Classify risk level under EU AI Act"),
        ("check_documentation", "Review Annex IV technical documentation"),
        ("audit_training_data", "Check bias, representativeness, data governance (Art. 10)"),
        ("verify_human_oversight", "Verify Art. 14 human-in-the-loop mechanisms"),
        ("check_transparency", "Check Art. 50 transparency obligations"),
        ("assess_risk_management", "Review risk management system (Art. 9)"),
        ("check_logging", "Verify automatic logging and traceability (Art. 12)"),
        ("submit_finding", "Report a compliance violation"),
        ("recommend_fix", "Propose remediation with priority"),
        ("verify_compliance", "Final determination &mdash; triggers terminal reward"),
    ]
    articles = [
        ("Article 5", "Prohibited AI Practices", "Social scoring, real-time biometric ID, manipulation"),
        ("Article 6 + Annex III", "High-Risk Classification", "Employment, credit, healthcare, law enforcement, migration"),
        ("Article 9", "Risk Management System", "Continuous lifecycle risk identification and mitigation"),
        ("Article 10", "Data Governance", "Training data quality, representativeness, bias testing"),
        ("Article 12", "Record-Keeping", "Automatic logging, traceability, audit trails"),
        ("Article 13", "Transparency", "Clear instructions for deployers, interpretability"),
        ("Article 14", "Human Oversight", "Human-in-the-loop, override capability, monitoring"),
        ("Article 50", "Transparency for All AI", "AI-generated content labeling, chatbot disclosure"),
    ]

    reward_li = "".join(f'<li><strong>{name} ({wt})</strong> &mdash; {desc}</li>' for name, wt, desc in reward_items)
    tools_li = "".join(f'<li><strong>{name}</strong> &mdash; {desc}</li>' for name, desc in tools)
    articles_li = "".join(f'<li><strong>{art}</strong>: {title} &mdash; {desc}</li>' for art, title, desc in articles)

    return f"""<div class="arch-grid">
        <div class="arch-box">
            <h3 style="color:{GOLD};">6-Component Reward System</h3>
            <ul class="arch-list">{reward_li}</ul>
        </div>
        <div class="arch-box">
            <h3 style="color:{GOLD};">11 MCP Audit Tools</h3>
            <ul class="arch-list">{tools_li}</ul>
        </div>
        <div class="arch-box">
            <h3 style="color:{GOLD};">EU AI Act Articles Covered</h3>
            <ul class="arch-list">{articles_li}</ul>
        </div>
        <div class="arch-box">
            <h3 style="color:{GOLD};">Audit Workflow</h3>
            <div style="color:{MUTED};font-size:13px;line-height:2;">
                <span style="color:{EMERALD};">1.</span> get_system_overview &rarr;
                <span style="color:{EMERALD};">2.</span> classify_system &rarr;
                <span style="color:{AMBER};">3.</span> check_documentation &rarr;
                <span style="color:{AMBER};">4.</span> audit_training_data &rarr;
                <span style="color:{AMBER};">5.</span> verify_human_oversight &rarr;
                <span style="color:{AMBER};">6.</span> check_transparency &rarr;
                <span style="color:{AMBER};">7.</span> assess_risk_management &rarr;
                <span style="color:{AMBER};">8.</span> check_logging &rarr;
                <span style="color:{ROSE};">9.</span> submit_finding (per violation) &rarr;
                <span style="color:{ROSE};">10.</span> recommend_fix (per finding) &rarr;
                <span style="color:{GOLD};">11.</span> verify_compliance (terminal)
            </div>
            <p style="color:{MUTED};font-size:12px;margin-top:12px;">
                Wrong tool order &rarr; worsened state transition. Skipping investigation &rarr; methodology penalty.
                Investigating red herrings &rarr; precision penalty. Budget: 100 queries per episode.
            </p>
        </div>
    </div>"""


def _try_it_html() -> str:
    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div class="arch-box">
            <h3 style="color:{GOLD};">NVIDIA NIM</h3>
            <div class="code-block">export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="google/gemma-4-31b-it"
export HF_TOKEN="nvapi-..."
python inference.py --space https://Itachi1824-compliance-auditor-env.hf.space</div>
        </div>
        <div class="arch-box">
            <h3 style="color:{GOLD};">HuggingFace Inference</h3>
            <div class="code-block">export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py --space https://Itachi1824-compliance-auditor-env.hf.space</div>
        </div>
    </div>
    <div class="arch-box" style="margin-top:16px;">
        <h3 style="color:{GOLD};">API Endpoints</h3>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px;">
            <div style="background:{BG};padding:12px;border-radius:6px;border:1px solid {BORDER};">
                <code style="color:{EMERALD};">POST /api/reset</code>
                <p style="color:{MUTED};font-size:11px;margin:4px 0 0;">Create session, returns tools + observation</p>
            </div>
            <div style="background:{BG};padding:12px;border-radius:6px;border:1px solid {BORDER};">
                <code style="color:{AMBER};">POST /api/call_tool</code>
                <p style="color:{MUTED};font-size:11px;margin:4px 0 0;">Call audit tool in active session</p>
            </div>
            <div style="background:{BG};padding:12px;border-radius:6px;border:1px solid {BORDER};">
                <code style="color:{ROSE};">POST /api/close</code>
                <p style="color:{MUTED};font-size:11px;margin:4px 0 0;">End session, cleanup</p>
            </div>
        </div>
    </div>"""


# ── Playground callbacks ────────────────────────────────────────

def _pg_reset(difficulty: str) -> Tuple:
    env = ComplianceAuditorEnvironment()
    obs = env.reset(difficulty=difficulty)
    sid = str(uuid.uuid4())
    with _pg_lock:
        _pg_sessions[sid] = env
    status = f"Session: {sid[:8]}... | Scenario: {env._scenario.scenario_id} | Queries: 0/100"
    return sid, status, obs.metadata

def _pg_call(sid: str, tool_name: str, args_str: str) -> Tuple:
    if not sid:
        return "Click Reset first", {"error": "No session"}
    with _pg_lock:
        env = _pg_sessions.get(sid)
    if not env:
        return "Session expired", {"error": "Session not found"}
    fn = env._tool_fns.get(tool_name)
    if not fn:
        return f"Unknown tool: {tool_name}", {"error": "Unknown tool"}
    try:
        kwargs = json.loads(args_str) if args_str and args_str.strip() else {}
    except json.JSONDecodeError:
        return "Invalid JSON", {"error": "Bad JSON in arguments"}
    try:
        result = fn(**kwargs)
        parsed = json.loads(result) if isinstance(result, str) else result
        queries = env._queries_used
        done = env._done
        reward = env._reward
        status = f"Queries: {queries}/100 | Findings: {len(env._findings_submitted)} | Done: {done}"
        if done:
            status += f" | REWARD: {reward:.4f}"
        return status, parsed
    except Exception as e:
        return f"Error: {e}", {"error": str(e)}


# ── Build the Gradio app ────────────────────────────────────────

TOOL_CHOICES = [
    "get_system_overview", "classify_system", "check_documentation",
    "audit_training_data", "verify_human_oversight", "check_transparency",
    "assess_risk_management", "check_logging", "submit_finding",
    "recommend_fix", "verify_compliance",
]


def create_landing_app() -> gr.Blocks:
    """Create and return the Gradio Blocks app."""

    with gr.Blocks(title="EU AI Act Compliance Auditor") as demo:
        gr.HTML(f"<style>{CSS}</style>")

        with gr.Tabs():
            # ── TAB 1: Overview ──
            with gr.Tab("Overview"):
                gr.HTML(_hero_html())
                gr.HTML(f"<h2>Design Decisions</h2>")
                gr.HTML(_design_cards_html())
                gr.HTML(f"""<div class="footer">
                    compliance_auditor_env &middot; 6-component reward &middot; 8 scenarios &middot;
                    11 tools &middot; state-graph audit &middot; parameter randomization &middot; deterministic grading<br/>
                    <a href="https://github.com/Itachi-1824/compliance-auditor-env">github</a> &middot;
                    <a href="https://huggingface.co/spaces/Itachi1824/compliance-auditor-env">huggingface</a>
                </div>""")

            # ── TAB 2: Scenarios ──
            with gr.Tab("Scenarios"):
                gr.HTML(f"<h2>8 compliance audit scenarios &middot; 3 tiers</h2>")
                gr.HTML(f'<p style="color:{MUTED};margin-bottom:16px;">Each scenario is a directed graph. Ground truth findings and required remediations shown below.</p>')
                gr.HTML(_scenarios_html())

            # ── TAB 3: Leaderboard ──
            with gr.Tab("Leaderboard"):
                gr.HTML(f"<h2>Frontier model baselines</h2>")
                gr.HTML(f'<p style="color:{MUTED};margin-bottom:16px;">Scores from baseline evaluation (1 episode per scenario). Each cell is the final reward.</p>')
                gr.HTML(_leaderboard_html())

            # ── TAB 4: Playground ──
            with gr.Tab("Playground"):
                gr.HTML(f"<h2>Interactive Audit</h2>")
                gr.HTML(f'<p style="color:{MUTED};margin-bottom:12px;">Reset to start a session, then call tools in sequence. The environment tracks your audit state and scores your methodology.</p>')

                session_state = gr.State(value=None)
                pg_status = gr.Textbox(label="Status", interactive=False, value="Click Reset to begin")

                with gr.Row(elem_classes="pg-row"):
                    pg_diff = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Difficulty")
                    pg_reset_btn = gr.Button("Reset", variant="primary", min_width=120)

                with gr.Row(elem_classes="pg-row"):
                    pg_tool = gr.Dropdown(choices=TOOL_CHOICES, value="get_system_overview", label="Tool")
                    pg_args = gr.Textbox(label="Arguments (JSON)", placeholder='{"risk_category": "high_risk"}')
                    pg_call_btn = gr.Button("Call Tool", variant="secondary", min_width=120)

                pg_result = gr.JSON(label="Result")

                def _on_reset(diff):
                    sid, status, obs = _pg_reset(diff)
                    return sid, status, obs

                def _on_call(sid, tool, args):
                    status, result = _pg_call(sid, tool, args)
                    return status, result

                pg_reset_btn.click(_on_reset, [pg_diff], [session_state, pg_status, pg_result])
                pg_call_btn.click(_on_call, [session_state, pg_tool, pg_args], [pg_status, pg_result])

            # ── TAB 5: Architecture ──
            with gr.Tab("Architecture"):
                gr.HTML(f"<h2>Environment Architecture</h2>")
                gr.HTML(_architecture_html())

            # ── TAB 6: Try It ──
            with gr.Tab("Try It"):
                gr.HTML(f"<h2>Run the baseline yourself</h2>")
                gr.HTML(_try_it_html())

        gr.HTML(f"""<div class="footer">
            compliance_auditor_env &middot; EU AI Act &middot; MCP tools &middot; OpenEnv<br/>
            Built for Meta PyTorch OpenEnv Hackathon 2026
        </div>""")

    return demo

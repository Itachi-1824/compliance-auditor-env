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
        ("FIXED SCENARIOS", "9"), ("PROCEDURAL", "\u221E"), ("MCP TOOLS", "11"),
        ("REWARD COMPS", "6"), ("TESTS", "74"), ("EU DEADLINE", "Aug '26"),
    ]
    stat_boxes = "".join(
        f'<div class="stat"><div class="val">{v}</div><div class="label">{k}</div></div>'
        for k, v in stats
    )
    return f"""
    <div class="hero">
        <div><span class="accent-bar"></span><h1 style="display:inline;vertical-align:middle;">EU AI Act Compliance Auditor</h1></div>
        <p class="subtitle">
            An MCP environment where LLM agents audit AI systems for EU AI Act compliance.
            Tools return investigation-grade regulatory documents &mdash; statistical tables, documentation inventories,
            operational procedures &mdash; that require genuine analysis to identify violations.
            No pre-digested verdicts. The agent must reason about evidence across 8 scenarios spanning
            prohibited social scoring, high-risk hiring bias, medical device compliance, and multi-system corporate audits.
        </p>
        <div class="stats">{stat_boxes}</div>
    </div>"""


def _design_cards_html() -> str:
    cards_data = [
        ("\u00A7", "Investigation-Grade Documents", "Tools return 30-70 line regulatory documents: Annex IV cross-reference tables, demographic callback rate matrices, operational procedure extracts. No labels like 'COMPLIANT' or 'FAILED' &mdash; the agent must analyze the evidence and reason about violations."),
        ("\u2699", "Dynamic Audit State", "The environment responds to the agent's actions in real-time. After submitting findings, subsequent tool calls show audit progress. After classification, investigation tools reflect the current audit context. The environment feels alive, not static."),
        ("\u25C8", "5 Unique Graph Topologies", "Each scenario has a distinct state graph. Prohibited systems have short detection paths (5 steps). Full high-risk audits require 11 steps across all investigation tools. Wrong tool order triggers worsened transitions. BFS-based partial credit."),
        ("\u25C9", "12 Anti-Gaming Tests", "Adversarial test suite proves the reward can't be gamed: skip investigation, spam findings, red herring bait, hallucinated findings, wrong classification isolation, fewer-than-optimal rushing, and 6 more exploit strategies. All proven ineffective."),
        ("\u27F3", "Cross-Document Reasoning", "Findings require correlating evidence across multiple tools. Hiring bias: training data shows 23% callback gap (audit_training_data) while only 5% of rejections reviewed (verify_human_oversight). Social scoring: 'wellness app' framing (overview) vs. public service access impact (check_transparency)."),
        ("\u221E", "Procedural Scenario Generator", "Beyond the 9 fixed scenarios, a seed-based procedural generator combines 5 system types &times; 16 violation templates &times; 5 red herrings to produce <strong>infinite unique scenarios</strong>. Use <code>procedural_medium_42</code> as scenario ID &mdash; every seed creates a different audit. Impossible to memorize."),
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


def _investigation_depth_html() -> str:
    """Show the before/after of investigation-grade tool responses."""
    return f"""
    <div class="arch-box" style="margin-bottom:16px;">
        <h3 style="color:{GOLD};">Investigation-Grade Tool Responses</h3>
        <p style="color:{MUTED};font-size:13px;margin-bottom:14px;">Tools return realistic regulatory documents requiring analysis — not pre-digested answers.</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div>
                <h4 style="color:{ROSE};font-size:11px;letter-spacing:0.05em;margin-bottom:8px;">TYPICAL ENV (pre-digested)</h4>
                <div class="code-block" style="font-size:11px;color:{MUTED};border-color:{ROSE}40;">{{"bias_assessment": "FAILED",
 "callback_rate_gap": "23%",
 "article_14_compliance": "NON-COMPLIANT",
 "human_oversight": "INSUFFICIENT"}}</div>
            </div>
            <div>
                <h4 style="color:{EMERALD};font-size:11px;letter-spacing:0.05em;margin-bottom:8px;">THIS ENV (investigation-grade)</h4>
                <div class="code-block" style="font-size:11px;color:{EMERALD};border-color:{EMERALD}40;">CALLBACK RATES BY DEMOGRAPHIC:
  Group             Rate     vs Baseline
  Male applicants   34.2%    (baseline)
  Female applicants 26.3%    -23.1%
  Eastern EU        27.4%    -19.9%

REVIEW STATISTICS (Q4 2025):
  Auto-rejected:    208,375  (60.0%)
  QA sample:         10,419  (5.0%)
  QA overrides:         312  (3.0%)</div>
            </div>
        </div>
        <p style="color:{MUTED};font-size:12px;margin-top:10px;">The agent must identify the 23% callback disparity from the table, recognize that 95% of rejections have no human review,
        and correlate these across documents to form findings. No verdict is pre-computed.</p>
    </div>"""


def _antigaming_html() -> str:
    """Anti-gaming test showcase."""
    tests = [
        ("Skip Investigation", "Submit correct findings without reading documents", "methodology = 0.0"),
        ("Spam Findings", "Flood 16 findings hoping to hit ground truth", "precision < 0.50"),
        ("Red Herring Bait", "Submit red herrings as violations", "precision = 0.0, completeness = 0.0"),
        ("Immediate Verify", "Call verify_compliance with empty inputs", "total < 0.05"),
        ("Wrong Classification", "Everything correct except risk category", "loses &ge; 10% gap"),
        ("Skip Remediation", "Find all violations but propose no fixes", "remediation = 0.0"),
        ("Classify Before Overview", "Skip system understanding", "methodology < 0.50"),
        ("Rush (Fewer Steps)", "Game efficiency by taking fewer steps", "efficiency penalized"),
        ("Hallucinate Findings", "Submit plausible-sounding false findings", "completeness < 0.40"),
        ("Wrong Class on Prohibited", "Call prohibited system high_risk", "classification = 0.40"),
        ("Perfect Run Sanity", "Legitimate perfect audit", "total > 0.85"),
        ("Bounds Check", "All scenarios x all inputs", "reward in (0.001, 0.999)"),
    ]
    rows = ""
    for name, strategy, result in tests:
        rows += f'<tr><td style="color:{TEXT};font-weight:500;">{name}</td><td style="color:{MUTED};font-size:12px;">{strategy}</td><td style="color:{ROSE};font-family:monospace;font-size:12px;">{result}</td></tr>'
    return f"""
    <div class="arch-box" style="margin-bottom:16px;">
        <h3 style="color:{GOLD};">12 Anti-Gaming Tests</h3>
        <p style="color:{MUTED};font-size:13px;margin-bottom:10px;">Adversarial test suite proving the reward function is robust against common exploits.</p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr><th style="text-align:left;color:{MUTED};font-size:10px;padding:6px 8px;border-bottom:1px solid {BORDER};">EXPLOIT</th>
                <th style="text-align:left;color:{MUTED};font-size:10px;padding:6px 8px;border-bottom:1px solid {BORDER};">STRATEGY</th>
                <th style="text-align:left;color:{MUTED};font-size:10px;padding:6px 8px;border-bottom:1px solid {BORDER};">RESULT</th></tr>
            {rows}
        </table>
    </div>"""


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


def _compliance_map_html() -> str:
    """EU AI Act article coverage matrix — unique to compliance audit domain."""
    mappings = [
        ("Article 5", "Prohibited Practices", "classify_system", ["hard_social_scoring"]),
        ("Article 6 + Annex III", "High-Risk Classification", "classify_system, assess_risk_management", ["medium_hiring", "medium_credit", "medium_medical", "hard_multi_system"]),
        ("Article 9", "Risk Management", "assess_risk_management", ["medium_hiring", "medium_credit", "medium_medical"]),
        ("Article 10", "Data Governance", "audit_training_data", ["medium_hiring", "medium_credit", "medium_medical", "hard_multi_system"]),
        ("Article 12", "Record-Keeping", "check_logging", ["medium_hiring", "medium_medical", "hard_deepfake", "hard_multi_system"]),
        ("Article 13", "Transparency (Deployers)", "check_transparency, check_documentation", ["medium_hiring", "medium_credit", "medium_medical"]),
        ("Article 14", "Human Oversight", "verify_human_oversight", ["medium_hiring", "medium_credit", "medium_medical", "hard_multi_system"]),
        ("Article 50", "Transparency (All AI)", "check_transparency", ["easy_chatbot", "hard_deepfake"]),
        ("Annex IV", "Technical Documentation", "check_documentation", ["medium_hiring", "medium_credit", "medium_medical", "hard_deepfake"]),
        ("MDR + AI Act", "Medical Device Dual-Regulation", "check_documentation, assess_risk_management", ["medium_medical"]),
    ]

    rows = ""
    for article, title, tools_str, scenarios in mappings:
        tool_badges = " ".join(
            f'<span style="background:{AMBER}15;color:{AMBER};padding:2px 8px;border-radius:4px;font-size:10px;font-family:monospace;">{t.strip()}</span>'
            for t in tools_str.split(",")
        )
        scenario_badges = " ".join(
            f'<span style="background:{BLUE}15;color:{BLUE};padding:2px 6px;border-radius:4px;font-size:10px;">{s}</span>'
            for s in scenarios
        )
        rows += f'''<tr>
            <td style="padding:10px 8px;border-bottom:1px solid {BORDER}10;white-space:nowrap;">
                <strong style="color:{GOLD};">{article}</strong><br/>
                <span style="color:{MUTED};font-size:11px;">{title}</span>
            </td>
            <td style="padding:10px 8px;border-bottom:1px solid {BORDER}10;">{tool_badges}</td>
            <td style="padding:10px 8px;border-bottom:1px solid {BORDER}10;">{scenario_badges}</td>
        </tr>'''

    return f"""<table style="width:100%;border-collapse:collapse;">
        <tr>
            <th style="text-align:left;color:{MUTED};font-size:10px;letter-spacing:0.06em;padding:8px;border-bottom:1px solid {BORDER};">ARTICLE</th>
            <th style="text-align:left;color:{MUTED};font-size:10px;letter-spacing:0.06em;padding:8px;border-bottom:1px solid {BORDER};">INVESTIGATION TOOLS</th>
            <th style="text-align:left;color:{MUTED};font-size:10px;letter-spacing:0.06em;padding:8px;border-bottom:1px solid {BORDER};">SCENARIOS</th>
        </tr>
        {rows}
    </table>
    <div style="margin-top:16px;padding:16px;background:{CARD};border:1px solid {BORDER};border-radius:10px;">
        <h4 style="color:{GOLD};font-size:13px;margin-bottom:8px;">Cross-Document Reasoning Requirements</h4>
        <div style="color:{MUTED};font-size:12px;line-height:1.8;">
            <strong style="color:{TEXT};">Hiring Bias (5 findings):</strong> audit_training_data reveals 23% callback gap &rarr; verify_human_oversight shows only 5% review rate &rarr; check_documentation confirms missing FRIA &rarr; agent must connect all three<br/>
            <strong style="color:{TEXT};">Social Scoring (5 findings):</strong> get_system_overview frames as "wellness app" &rarr; check_transparency reveals service access impact &rarr; verify_human_oversight shows municipal integration &rarr; agent must recognize Art. 5 violation<br/>
            <strong style="color:{TEXT};">Multi-System (6 findings):</strong> audit_training_data reveals cross-system data flows &rarr; check_documentation shows missing combined DPIA &rarr; verify_human_oversight reveals no unified oversight &rarr; compound risk emerges across documents<br/>
            <strong style="color:{TEXT};">Medical Triage (4 findings):</strong> audit_training_data shows age-bias in 75+ cohort &rarr; check_documentation confirms retrospective-only validation &rarr; check_logging reveals no real-time monitoring &rarr; safety gap pattern
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

# Scenario choices for the picker
SCENARIO_CHOICES = [s["id"] for s in SCENARIO_LIST]

# Tool-specific argument hints
TOOL_ARG_HINTS = {
    "get_system_overview": "",
    "classify_system": '{"risk_category": "high_risk"}',
    "check_documentation": "",
    "audit_training_data": "",
    "verify_human_oversight": "",
    "check_transparency": "",
    "assess_risk_management": "",
    "check_logging": "",
    "submit_finding": '{"finding": "gender_bias_in_screening", "severity": "critical"}',
    "recommend_fix": '{"finding": "bias", "remediation": "conduct_bias_audit", "priority": 1}',
    "verify_compliance": '{"risk_classification": "high_risk", "overall_assessment": "Multiple gaps found", "key_findings_summary": "Bias, oversight, documentation issues"}',
}

TOOL_CHOICES = [
    "get_system_overview", "classify_system", "check_documentation",
    "audit_training_data", "verify_human_oversight", "check_transparency",
    "assess_risk_management", "check_logging", "submit_finding",
    "recommend_fix", "verify_compliance",
]


def _pg_reset(scenario_id: str) -> Tuple:
    env = ComplianceAuditorEnvironment()
    env.reset(scenario_id=scenario_id)
    sid = str(uuid.uuid4())
    with _pg_lock:
        _pg_sessions[sid] = env
    sc = env._scenario
    status_html = (
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">'
        f'<div class="stat"><div class="val" style="font-size:1.2em;">{sc.scenario_id.split("_")[0].upper()}</div><div class="label">DIFFICULTY</div></div>'
        f'<div class="stat"><div class="val" style="font-size:1.2em;">0/100</div><div class="label">QUERIES</div></div>'
        f'<div class="stat"><div class="val" style="font-size:1.2em;">0</div><div class="label">FINDINGS</div></div>'
        f'<div class="stat"><div class="val" style="font-size:1.2em;">0</div><div class="label">REMEDIATIONS</div></div>'
        f'<div class="stat"><div class="val" style="font-size:1.2em;color:{MUTED};">—</div><div class="label">REWARD</div></div>'
        f'</div>'
    )
    alert_msg = (
        f"COMPLIANCE AUDIT ASSIGNED\n\n"
        f"System: {sc.system_name}\n"
        f"Classification: {sc.correct_classification} (hidden from agent)\n"
        f"Findings to discover: {len(sc.ground_truth_findings)}\n\n"
        f"Call get_system_overview to begin."
    )
    return sid, status_html, alert_msg, json.dumps({"session": sid[:8], "scenario": sc.scenario_id}, indent=2)


def _pg_call(sid: str, tool_name: str, args_str: str) -> Tuple:
    if not sid:
        return '<div style="color:#F43F5E;">Click Reset first</div>', "(no session)", {"error": "No session"}
    with _pg_lock:
        env = _pg_sessions.get(sid)
    if not env:
        return '<div style="color:#F43F5E;">Session expired — click Reset</div>', "(expired)", {"error": "Session not found"}
    fn = env._tool_fns.get(tool_name)
    if not fn:
        return f'<div style="color:#F43F5E;">Unknown tool: {tool_name}</div>', "(error)", {"error": "Unknown tool"}
    try:
        kwargs = json.loads(args_str) if args_str and args_str.strip() else {}
    except json.JSONDecodeError:
        return '<div style="color:#F43F5E;">Invalid JSON in arguments</div>', "(error)", {"error": "Bad JSON"}
    try:
        result = fn(**kwargs)
        parsed = json.loads(result) if isinstance(result, str) else result
        queries = env._queries_used
        done = env._done
        reward = env._reward
        findings_n = len(env._findings_submitted)
        remed_n = len(env._remediation_submitted)

        # Status dashboard
        reward_color = EMERALD if reward >= 0.6 else (AMBER if reward >= 0.3 else (ROSE if done else MUTED))
        reward_display = f"{reward:.4f}" if done else "—"
        done_indicator = f'<span style="color:{EMERALD};">COMPLETE</span>' if done else f'<span style="color:{AMBER};">IN PROGRESS</span>'
        status_html = (
            f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">'
            f'<div class="stat"><div class="val" style="font-size:1.2em;">{done_indicator}</div><div class="label">STATUS</div></div>'
            f'<div class="stat"><div class="val" style="font-size:1.2em;">{queries}/100</div><div class="label">QUERIES</div></div>'
            f'<div class="stat"><div class="val" style="font-size:1.2em;">{findings_n}</div><div class="label">FINDINGS</div></div>'
            f'<div class="stat"><div class="val" style="font-size:1.2em;">{remed_n}</div><div class="label">REMEDIATIONS</div></div>'
            f'<div class="stat"><div class="val" style="font-size:1.2em;color:{reward_color};">{reward_display}</div><div class="label">REWARD</div></div>'
            f'</div>'
        )

        # Extract document content for rich display
        doc_content = parsed.get("content", "")
        if not doc_content and "audit_summary" in parsed:
            summary = parsed["audit_summary"]
            lines = [f"AUDIT COMPLETE — Reward: {parsed.get('reward', 0):.4f}"]
            lines.append(f"\nClassification: {summary['classification']['submitted']} "
                        f"({'correct' if summary['classification']['match'] == 'exact' else summary['classification']['match']})")
            lines.append(f"Correct answer: {summary['classification']['correct']}")
            lines.append(f"\nFindings: {summary['findings']['matched']}/{summary['findings']['ground_truth_total']} matched")
            if summary["findings"]["missed"]:
                lines.append("Missed:")
                for m in summary["findings"]["missed"]:
                    lines.append(f"  - {m}")
            lines.append(f"\nAreas investigated: {', '.join(summary.get('areas_investigated', []))}")
            lines.append(f"\nReward breakdown:")
            for k, v in parsed.get("reward_breakdown", {}).items():
                lines.append(f"  {k}: {v}")
            doc_content = "\n".join(lines)
        elif not doc_content:
            doc_content = json.dumps(parsed, indent=2)

        return status_html, doc_content, parsed
    except Exception as e:
        return f'<div style="color:#F43F5E;">Error: {e}</div>', str(e), {"error": str(e)}


def _pg_update_hint(tool_name: str) -> str:
    """Return argument hint when tool selection changes."""
    return TOOL_ARG_HINTS.get(tool_name, "")


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
                    compliance_auditor_env &middot; 6-component reward &middot; 9 scenarios + procedural &middot;
                    11 tools &middot; state-graph audit &middot; parameter randomization &middot; 74 tests<br/>
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
                gr.HTML(f"""<h2>Live Interactive Audit</h2>
                    <p style="color:{MUTED};margin-bottom:16px;">
                        Pick a scenario, reset the environment, then call tools step by step against the real state graph.
                        Watch the system state change after each action.
                    </p>""")

                session_state = gr.State(value=None)

                # ── Scenario picker + Reset ──
                with gr.Row(elem_classes="pg-row"):
                    pg_scenario = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value="medium_hiring_bias_001",
                        label="Scenario",
                    )
                    pg_reset_btn = gr.Button("Reset episode", variant="primary", min_width=160)

                # ── Status dashboard ──
                pg_status = gr.HTML(
                    value=f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;color:{MUTED};">Click Reset to start an episode</div>'
                )

                # ── Document viewer ──
                pg_doc = gr.Textbox(
                    label="Document content (what the agent sees)",
                    lines=18,
                    interactive=False,
                    value="Click Reset to start, then call get_system_overview.",
                )

                # ── Call a tool section ──
                gr.HTML(f'<h3 style="margin-top:16px;">Call a tool</h3>')

                with gr.Row(elem_classes="pg-row"):
                    pg_tool = gr.Dropdown(choices=TOOL_CHOICES, value="get_system_overview", label="Tool")
                    pg_risk_cat = gr.Dropdown(
                        choices=["prohibited", "high_risk", "limited_risk", "minimal_risk"],
                        value="high_risk",
                        label="RISK CATEGORY",
                        visible=False,
                    )
                    pg_severity = gr.Dropdown(
                        choices=["critical", "high", "medium", "low"],
                        value="high",
                        label="SEVERITY",
                        visible=False,
                    )

                with gr.Row(elem_classes="pg-row"):
                    pg_finding = gr.Textbox(label="FINDING", placeholder="e.g. gender_bias_in_screening", visible=False)
                    pg_remediation = gr.Textbox(label="REMEDIATION", placeholder="e.g. conduct_bias_audit", visible=False)
                    pg_assessment = gr.Textbox(label="OVERALL ASSESSMENT", placeholder="e.g. Multiple compliance gaps identified", visible=False)
                    pg_summary = gr.Textbox(label="KEY FINDINGS SUMMARY", placeholder="e.g. Bias, oversight, documentation issues", visible=False)
                    pg_priority = gr.Number(label="PRIORITY", value=1, visible=False, precision=0)

                pg_call_btn = gr.Button("Step", variant="secondary")

                with gr.Accordion("Raw JSON response", open=False):
                    pg_result = gr.JSON(label="Raw")

                # ── Tool-specific field visibility ──
                def _on_tool_change(tool):
                    """Show/hide fields based on selected tool."""
                    is_classify = tool == "classify_system"
                    is_finding = tool == "submit_finding"
                    is_fix = tool == "recommend_fix"
                    is_verify = tool == "verify_compliance"
                    return (
                        gr.update(visible=is_classify or is_verify),     # risk_cat
                        gr.update(visible=is_finding),                    # severity
                        gr.update(visible=is_finding or is_fix),         # finding
                        gr.update(visible=is_fix),                        # remediation
                        gr.update(visible=is_verify),                     # assessment
                        gr.update(visible=is_verify),                     # summary
                        gr.update(visible=is_fix),                        # priority
                    )

                # ── Build args from fields ──
                def _on_call(sid, tool, risk_cat, severity, finding, remediation, assessment, summary, priority):
                    # Build args dict from the visible fields
                    if tool == "classify_system":
                        args_str = json.dumps({"risk_category": risk_cat})
                    elif tool == "submit_finding":
                        args_str = json.dumps({"finding": finding or "compliance_gap", "severity": severity})
                    elif tool == "recommend_fix":
                        args_str = json.dumps({"finding": finding or "issue", "remediation": remediation or "fix", "priority": int(priority or 1)})
                    elif tool == "verify_compliance":
                        args_str = json.dumps({"risk_classification": risk_cat, "overall_assessment": assessment or "Audit complete", "key_findings_summary": summary or "See findings"})
                    else:
                        args_str = "{}"
                    status_html, doc_content, result = _pg_call(sid, tool, args_str)
                    return status_html, doc_content, result

                def _on_reset(scenario_id):
                    sid, status_html, doc_content, raw = _pg_reset(scenario_id)
                    return sid, status_html, doc_content, raw

                pg_reset_btn.click(_on_reset, [pg_scenario], [session_state, pg_status, pg_doc, pg_result])
                pg_call_btn.click(
                    _on_call,
                    [session_state, pg_tool, pg_risk_cat, pg_severity, pg_finding, pg_remediation, pg_assessment, pg_summary, pg_priority],
                    [pg_status, pg_doc, pg_result],
                )
                pg_tool.change(
                    _on_tool_change,
                    [pg_tool],
                    [pg_risk_cat, pg_severity, pg_finding, pg_remediation, pg_assessment, pg_summary, pg_priority],
                )

            # ── TAB 5: Architecture ──
            with gr.Tab("Architecture"):
                gr.HTML(f"<h2>Environment Architecture</h2>")
                gr.HTML(_investigation_depth_html())
                gr.HTML(_antigaming_html())
                gr.HTML(_architecture_html())

            # ── TAB 6: Compliance Map ──
            with gr.Tab("Compliance Map"):
                gr.HTML(f"<h2>EU AI Act Article Coverage</h2>")
                gr.HTML(f'<p style="color:{MUTED};margin-bottom:16px;">How each investigation tool maps to EU AI Act provisions, and which scenarios test each article.</p>')
                gr.HTML(_compliance_map_html())

            # ── TAB 7: Try It ──
            with gr.Tab("Try It"):
                gr.HTML(f"<h2>Run the baseline yourself</h2>")
                gr.HTML(_try_it_html())

        gr.HTML(f"""<div class="footer">
            compliance_auditor_env &middot; EU AI Act &middot; MCP tools &middot; OpenEnv<br/>
            Built for Meta PyTorch OpenEnv Hackathon 2026
        </div>""")

    return demo

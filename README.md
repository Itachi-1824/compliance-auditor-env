---
title: EU AI Act Compliance Auditor
emoji: "🏛"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
---

# EU AI Act Compliance Auditor

An MCP environment where LLM agents audit AI systems for EU AI Act compliance. Tools return **investigation-grade regulatory documents** — statistical tables, documentation inventories, operational procedures — that require genuine analysis to identify violations. No pre-digested verdicts. The agent must reason about evidence across documents to find compliance gaps.

## What Makes This Different

Most compliance environments hand the agent pre-labeled answers: `"bias_assessment": "FAILED"`. This environment returns the **raw evidence**:

```
CALLBACK RATES BY DEMOGRAPHIC (Technical Roles Only):
  Group               Rate     vs Baseline
  Male applicants     34.2%    (baseline)
  Female applicants   26.3%    -23.1%
  Eastern EU          27.4%    -19.9%
```

The agent must identify the 23% callback disparity from the table, recognize it as gender bias, cross-reference with the oversight document showing only 5% of rejections are reviewed, and connect these into actionable findings.

## Stats

| Metric | Value |
|--------|-------|
| Fixed Scenarios | 9 across 3 difficulty tiers |
| Procedural Scenarios | Infinite (seed-based generation) |
| MCP Tools | 11 (8 investigation + 3 resolution) |
| Reward Components | 6 (weighted, anti-gaming) |
| Graph Topologies | 6 unique per-scenario |
| Document Depth | 500-3,275 chars per tool response |
| Total Document Content | 77K+ chars across all scenarios |
| Anti-Gaming Tests | 12 adversarial exploits proven ineffective |
| Test Suite | 74 tests across 8 files |
| Adaptive Depth | Repeat tool calls reveal forensic deep-dive |
| Dynamic State | Environment reacts to findings and remediations |
| Parameter Randomization | Company, region, version, dates per reset |

## Scenarios

### Easy (2) — Clear-cut systems, focused investigation
- **Customer Service Chatbot** — Limited-risk. Missing AI disclosure under Article 50. Agent checks transparency and oversight.
- **Music Recommendation Engine** — Minimal-risk. Voluntary code of conduct recommended. Short investigation path.

### Medium (3) — Statistical evidence, red herrings, multi-article violations
- **AI Resume Screener** — High-risk hiring AI (Annex III). 5 findings: gender bias (23% callback gap), insufficient oversight (5% review rate), missing FRIA, incomplete Annex IV docs, data governance gaps.
- **Credit Scoring Model** — High-risk fintech. Opaque alternative data features (social media, device metadata), no right to human review, missing conformity assessment.
- **Emergency Triage AI** — Medical device dual-regulation (MDR + AI Act). Age bias in 75+ cohort (76.3% sensitivity), retrospective-only validation, no real-time monitoring.
- **Workplace Emotion Recognition** — **PROHIBITED** under Article 5(1)(f). Webcam-based "engagement analytics" that's actually emotion recognition. Deployer frames it as productivity tool — agent must recognize it processes biometric data (facial action units, micro-expressions) without medical/safety exception.

### Hard (3) — Disguised systems, compound risks, multi-system dependencies
- **Citizen Wellness App** — **PROHIBITED** social scoring disguised as voluntary wellness tool. Deployer frames it as gamification, but investigation reveals it controls access to public services based on social behavior scores. Agent must see through the framing.
- **AI Content Studio** — Deepfake generation platform. Missing all Article 50 content labeling, no C2PA watermarking, no content provenance. Political content generated without disclosure.
- **Corporate AI Portfolio** — 4 interconnected AI systems sharing a data lake. Agent must identify cross-system data flows amplifying risks, recognize employee sentiment analysis as high-risk, and spot biometric categorization in safety monitoring.

## Procedural Scenario Generator

Beyond the 9 hand-crafted scenarios, a seed-based procedural generator produces **infinite unique scenarios** by combining:

- **5 system types**: Drone delivery (critical infrastructure), exam proctoring (education), insurance adjudication (essential services), legal research (limited risk), predictive policing (prohibited)
- **16 violation templates**: Gender bias, age discrimination, data governance gaps, missing conformity, logging inadequacies, and more
- **5 red herring templates**: GDPR confusion, compliant sibling systems, ISO certifications, voluntary ethics boards

```python
# Any seed produces a unique, coherent scenario
env.reset(scenario_id="procedural_medium_42")   # Seed 42, medium difficulty
env.reset(scenario_id="procedural_hard_12345")  # Seed 12345, hard difficulty
```

Each generated scenario has proper ground truth findings, matching state graph, violation-specific documents, and is fully compatible with the 6-component reward function.

## Action & Observation Spaces

### Action (ComplianceAction)
```python
class ComplianceAction(Action):
    tool_name: str    # Name of the audit tool to call
    arguments: dict   # Tool arguments as JSON (e.g. {"risk_category": "high_risk"})
```

### Observation (ComplianceObservation)
```python
class ComplianceObservation(Observation):
    done: bool          # Whether the episode is complete
    reward: float       # Current step reward (terminal on verify_compliance)
    metadata: dict      # Tool response content, audit context
    queries_remaining: int
```

### State (ComplianceState)
```python
class ComplianceState(BaseModel):
    episode_id: str
    step_count: int
    scenario_id: str
    difficulty: str           # easy / medium / hard
    queries_used: int
    findings_count: int
    compliance_verified: bool
    current_reward: float
```

## Tools

### Investigation
| Tool | Returns |
|------|---------|
| `get_system_overview` | Formal audit assignment brief with system description and deployment context |
| `classify_system` | Records risk classification (prohibited / high_risk / limited_risk / minimal_risk) |
| `check_documentation` | Annex IV cross-reference table with per-section compliance status |
| `audit_training_data` | Demographic statistics tables, data governance assessment, bias indicators |
| `verify_human_oversight` | Operational procedures extract with review statistics and override capabilities |
| `check_transparency` | User-facing UI/ToS text analysis with Article 50 compliance indicators |
| `assess_risk_management` | Risk register, conformity assessment tracker, Annex III classification analysis |
| `check_logging` | Audit log schema, Article 12 requirements gap analysis |

### Resolution
| Tool | Purpose |
|------|---------|
| `submit_finding` | Report a compliance violation (call once per finding) |
| `recommend_fix` | Propose remediation with priority |
| `verify_compliance` | Final determination — triggers terminal 6-component reward |

## 6-Component Reward

| Component | Weight | Anti-Gaming |
|-----------|--------|-------------|
| Classification | 20% | Adjacent-category partial credit (40%). Wrong by 2+ categories = 0. |
| Finding Completeness | 25% | Token-based fuzzy matching (Jaccard 40%, min 2 tokens). Prevents keyword stuffing. |
| Finding Precision | 15% | Red herring submissions penalized 15% each. False positives reduce score. |
| Remediation Quality | 15% | Presence (70%) + priority ordering (30%). Missing remediation = 0. |
| Methodology | 15% | Order violations penalized. Skipping investigation tools = 0. |
| Efficiency | 10% | Fewer steps than optimal = penalty (skipping investigation). More steps = diminishing returns. |

All rewards clamped to (0.001, 0.999). 12 adversarial tests prove robustness.

## Architecture

```
compliance_env/
  server/
    environment.py      # MCP environment, 11 tools, dynamic audit state
    engine.py           # State graph + 6-component reward computation
    app.py              # FastAPI + HTTP session API + Gradio UI
    gradio_landing.py   # 7-tab dashboard with investigation depth showcase
  scenarios/
    registry.py         # 8 scenarios with 77K+ chars of investigation documents
  tests/
    test_environment.py       # 14 environment + API tests
    test_reward_hacking.py    # 12 adversarial anti-gaming tests
    test_investigation_depth.py # 10 investigation quality tests
  inference.py          # OpenAI function-calling baseline agent
  client.py             # Zero-dependency HTTP client
  models.py             # Pydantic observation/state models
  Dockerfile            # python:3.11-slim, port 7860
  openenv.yaml          # OpenEnv manifest with tasks
```

## Quick Start

```bash
# Install
pip install "openenv-core[core]" fastmcp gradio httpx openai

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (NVIDIA NIM)
export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="stepfun-ai/step-3.5-flash"
export HF_TOKEN="nvapi-..."
python inference.py --space https://Itachi1824-compliance-auditor-env.hf.space

# Docker
docker build -t compliance-env . && docker run -p 7860:7860 compliance-env

# Tests
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reset` | POST | Create session, returns tools + initial observation |
| `/api/call_tool` | POST | Call an audit tool in an active session |
| `/api/close` | POST | End session and cleanup |
| `/tasks` | GET | List available scenarios |
| `/grader` | POST | Grade a completed episode |
| `/health` | GET | Health check |

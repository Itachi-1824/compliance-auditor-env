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

An MCP-based environment where LLM agents audit AI systems for EU AI Act compliance — from risk classification to violation identification to remediation planning. Scenarios based on real regulatory articles. Parameter randomization on every reset prevents memorization; agents must learn the **audit process**, not specific answers.

## Why This Environment

The EU AI Act's major enforcement deadline is **August 2, 2026** — less than 4 months away. Every company deploying AI in Europe faces fines up to **EUR 35 million or 7% of global revenue**. Yet no automated compliance auditing benchmark exists. This environment fills that gap with 8 realistic scenarios across the full spectrum of EU AI Act risk categories.

## Stats

| Metric | Value |
|--------|-------|
| Scenarios | 8 |
| MCP Tools | 11 |
| Reward Components | 6 |
| Difficulty Tiers | 3 (easy / medium / hard) |
| State Graph Nodes | 12 per scenario |
| Parameter Randomization | Company, region, version, dates per reset |

## Tools (MCP Interface)

### Investigation
| Tool | Description |
|------|-------------|
| `get_system_overview` | Gather system description, deployer info, deployment context |
| `classify_system` | Classify risk level (prohibited / high_risk / limited_risk / minimal_risk) |
| `check_documentation` | Review Annex IV technical documentation completeness |
| `audit_training_data` | Check bias, representativeness, data governance (Article 10) |
| `verify_human_oversight` | Verify Article 14 human-in-the-loop mechanisms |
| `check_transparency` | Check Article 50 transparency obligations |
| `assess_risk_management` | Review risk management system (Article 9) |
| `check_logging` | Verify automatic logging and traceability (Article 12) |

### Resolution
| Tool | Description |
|------|-------------|
| `submit_finding` | Report a compliance violation (call per finding) |
| `recommend_fix` | Propose remediation with priority |
| `verify_compliance` | Final determination — triggers terminal reward |

## Scenarios

### Easy
- **Customer Service Chatbot** — Limited-risk system missing AI disclosure (Article 50)
- **Music Recommendation Engine** — Minimal-risk system needing voluntary code of conduct

### Medium
- **AI Resume Screener** — High-risk hiring AI (Annex III) with gender bias, missing oversight, incomplete documentation
- **Credit Scoring Model** — High-risk fintech system with opaque features and no right to human review
- **Emergency Triage AI** — Medical device with age bias and no prospective clinical validation

### Hard
- **Citizen Wellness App** — **PROHIBITED** social scoring system disguised as a voluntary wellness tool. Must identify it as prohibited under Article 5(1)(c)
- **AI Content Studio** — Deepfake generation platform missing all Article 50 transparency obligations
- **Corporate AI Portfolio** — Multi-system audit with 4 interconnected AI systems sharing a data lake. Must identify compound risks and cross-system data flow issues

## 6-Component Reward

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification | 20% | Correct risk category identification |
| Finding Completeness | 25% | Recall of ground-truth violations |
| Finding Precision | 15% | Penalty for false positives / red herring findings |
| Remediation Quality | 15% | Correct fixes in priority order |
| Methodology | 15% | Followed correct audit sequence (overview → classify → investigate → find → fix → verify) |
| Efficiency | 10% | Queries used vs optimal path |

All rewards clamped to (0.01, 0.99) for OpenEnv validator compliance.

## Quick Start

```bash
# Install
pip install "openenv-core[core]" fastmcp gradio httpx openai

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference
export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="google/gemma-4-31b-it"
export HF_TOKEN="your-key"
python inference.py --space https://Itachi1824-compliance-auditor-env.hf.space

# Docker
docker build -t compliance-env . && docker run -p 7860:7860 compliance-env
```

## API

### Standard OpenEnv
- `POST /reset` — Start new episode
- `POST /step` — Execute action
- `GET /state` — Get episode state
- `GET /health` — Health check

### Custom HTTP Session API
- `POST /api/reset` — Create session, returns tools + observation
- `POST /api/call_tool` — Call an audit tool in a session
- `POST /api/close` — End session

## Architecture

```
compliance_env/
├── server/
│   ├── app.py              # FastAPI + sessions + Gradio UI
│   ├── environment.py      # MCP environment with 11 tools
│   └── engine.py           # State graph + 6-component reward
├── scenarios/
│   └── registry.py         # 8 scenarios with state graphs
├── client.py               # HTTP client for inference
├── inference.py             # OpenAI function-calling agent
├── models.py               # Pydantic observation/state models
├── Dockerfile              # Port 7860, python:3.11-slim
└── openenv.yaml            # OpenEnv manifest with tasks
```

## Baseline Scores

Tested against live HF Space with NVIDIA NIM models:

| Rank | Model | Easy | Medium | Hard | Overall |
|------|-------|------|--------|------|---------|
| 1 | stepfun-ai/step-3.5-flash | 0.473 | 0.425 | 0.404 | **0.434** |
| 2 | mistralai/mistral-small-4-119b | 0.457 | 0.425 | 0.348 | **0.410** |
| 3 | deepseek-ai/deepseek-v3.1 | 0.442 | 0.425 | 0.348 | **0.405** |

Hard scenarios genuinely challenge frontier models — the prohibited social scoring detection requires the agent to see through deliberate misdirection ("wellness app" that's actually social scoring affecting public service access).

## Sample Output

```
[START] task=easy_chatbot_transparency_001 env=compliance_auditor_env model=google/gemma-4-31b-it
[STEP] step=1 action=get_system_overview reward=0.00 done=false error=null
[STEP] step=2 action=classify_system reward=0.00 done=false error=null
[STEP] step=3 action=check_documentation reward=0.00 done=false error=null
[STEP] step=4 action=check_transparency reward=0.00 done=false error=null
[STEP] step=5 action=submit_finding reward=0.00 done=false error=null
[STEP] step=6 action=verify_compliance reward=0.46 done=true error=null
[END] success=true steps=6 score=0.457 rewards=0.00,0.00,0.00,0.00,0.00,0.46
```

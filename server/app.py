"""
FastAPI application for the EU AI Act Compliance Auditor.

Provides:
  - Standard OpenEnv endpoints (/reset, /step, /state, /health)
  - Custom HTTP session API (/api/reset, /api/call_tool, /api/close)
  - Custom Gradio landing page with dashboard
  - /tasks endpoint for hackathon validator
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    create_app = None

try:
    from server.environment import ComplianceAuditorEnvironment, QUERY_BUDGET
    from scenarios.registry import SCENARIO_LIST, DIFFICULTY_TIERS
except ImportError:
    from .environment import ComplianceAuditorEnvironment, QUERY_BUDGET
    from ..scenarios.registry import SCENARIO_LIST, DIFFICULTY_TIERS


# ---------------------------------------------------------------------------
# Create the base OpenEnv app
# ---------------------------------------------------------------------------

if create_app:
    from openenv.core.env_server.types import Action, Observation
    app = create_app(
        ComplianceAuditorEnvironment,
        Action,
        Observation,
        env_name="compliance_auditor_env",
        max_concurrent_envs=5,
    )
else:
    app = FastAPI(title="EU AI Act Compliance Auditor")
    @app.get("/health")
    def health():
        return {"status": "healthy"}


# ---------------------------------------------------------------------------
# /tasks endpoint (required by hackathon validator)
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    return {"tasks": SCENARIO_LIST}


# ---------------------------------------------------------------------------
# Custom HTTP Session API (like Maverick98's pattern)
# ---------------------------------------------------------------------------

_sessions: Dict[str, ComplianceAuditorEnvironment] = {}
_sessions_lock = threading.Lock()


class ResetRequest(BaseModel):
    difficulty: str = "medium"
    scenario_id: Optional[str] = None
    seed: Optional[int] = None


class CallToolRequest(BaseModel):
    session_id: str
    tool_name: str
    arguments: Dict[str, Any] = {}


class CloseRequest(BaseModel):
    session_id: str


def _get_server_tools(env: ComplianceAuditorEnvironment) -> Dict[str, Any]:
    """Get tool functions directly from the environment."""
    return env._tool_fns


def _get_tool_schemas(env: ComplianceAuditorEnvironment) -> list:
    """Get tool schemas for OpenAI function calling format."""
    import inspect
    schemas = []
    for name, fn in env._tool_fns.items():
        # Extract parameters from function signature
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            ptype = "string"
            desc = ""
            if param.annotation == int:
                ptype = "integer"
            elif param.annotation == float:
                ptype = "number"
            elif param.annotation == bool:
                ptype = "boolean"
            properties[pname] = {"type": ptype, "description": desc}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        schemas.append({
            "name": name,
            "description": (fn.__doc__ or "").strip().split("\n")[0],
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return schemas


@app.post("/api/reset")
async def api_reset(req: ResetRequest):
    """Create a new audit session."""
    session_id = str(uuid.uuid4())
    env = ComplianceAuditorEnvironment()
    obs = env.reset(
        difficulty=req.difficulty,
        scenario_id=req.scenario_id,
        seed=req.seed,
    )

    with _sessions_lock:
        _sessions[session_id] = env

    # Get tool schemas for the client
    tool_schemas = _get_tool_schemas(env)

    return {
        "session_id": session_id,
        "observation": obs.metadata if hasattr(obs, "metadata") else {},
        "tools": tool_schemas,
        "scenario_id": env._scenario.scenario_id,
        "difficulty": env._scenario.difficulty,
    }


@app.post("/api/call_tool")
async def api_call_tool(req: CallToolRequest):
    """Call a tool in an active session."""
    with _sessions_lock:
        env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call /api/reset first.")

    tool_fns = env._tool_fns
    fn = tool_fns.get(req.tool_name)
    if fn is None:
        return {
            "result": json.dumps({"error": f"Unknown tool: {req.tool_name}. Available: {list(tool_fns.keys())}"}),
            "done": False,
            "reward": 0.0,
        }

    # Call the tool function directly
    try:
        result = fn(**req.arguments)
    except Exception as e:
        result = json.dumps({"error": str(e)})

    if not isinstance(result, str):
        result = json.dumps(result) if result else ""

    return {
        "result": result,
        "done": env._done,
        "reward": env._reward,
    }


@app.post("/api/close")
async def api_close(req: CloseRequest):
    """Close a session and clean up."""
    with _sessions_lock:
        env = _sessions.pop(req.session_id, None)
    if env:
        env.close()
    return {"status": "closed"}


# ---------------------------------------------------------------------------
# Gradio Landing Page
# ---------------------------------------------------------------------------

_gradio_mounted = False

# Color system: Deep charcoal + warm gold (authority/prestige/trust)
# NO neon, NO AI-typical cyan/purple
_BG = "#09090B"          # base background
_CARD = "#18181B"        # card surface
_BORDER = "#27272A"      # borders
_TEXT = "#F8FAFC"        # primary text
_MUTED = "#94A3B8"       # secondary text (slate)
_GOLD = "#C9A84C"        # accent: authority, prestige
_EMERALD = "#10B981"     # success/easy
_AMBER = "#F59E0B"       # warning/medium
_ROSE = "#F43F5E"        # critical/hard

try:
    import gradio as gr

    def _hero():
        return f"""
        <div style="background:{_CARD};padding:48px 40px;border-radius:12px;margin-bottom:28px;border:1px solid {_BORDER};">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                <div style="width:8px;height:36px;background:{_GOLD};border-radius:4px;"></div>
                <h1 style="color:{_TEXT};font-size:2em;margin:0;font-weight:700;letter-spacing:-0.02em;">EU AI Act Compliance Auditor</h1>
            </div>
            <p style="color:{_MUTED};font-size:1.05em;margin:8px 0 32px 20px;line-height:1.6;max-width:720px;">
                An MCP-based environment where LLM agents audit AI systems for EU AI Act compliance.
                8 scenarios from chatbot transparency to prohibited social scoring.
                Parameter randomization prevents memorization.
            </p>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;">
                {"".join(f'''<div style="background:{_BG};padding:20px 16px;border-radius:10px;text-align:center;border:1px solid {_BORDER};">
                    <div style="color:{_GOLD};font-size:1.8em;font-weight:700;">{v}</div>
                    <div style="color:{_MUTED};font-size:0.78em;letter-spacing:0.05em;margin-top:4px;">{k}</div>
                </div>''' for k, v in [("SCENARIOS", 8), ("MCP TOOLS", 11), ("REWARD COMPS", 6), ("TIERS", 3), ("EU DEADLINE", "Aug '26")])}
            </div>
        </div>"""

    def _design_cards():
        cards = [
            ("Real Regulatory Scenarios", "Scenarios drawn from actual EU AI Act articles: prohibited social scoring (Art. 5), high-risk hiring (Annex III), deepfake transparency (Art. 50), medical device audits. Based on real compliance gaps companies face today."),
            ("Full Audit Toolkit", "11 MCP tools mirror a real compliance auditor's workflow: system overview, risk classification, documentation review, bias audit, oversight verification, transparency check, risk assessment, logging verification."),
            ("State-Graph Audit Process", "Each scenario is a directed graph with progress, no-effect, and worsened transitions. Partial credit computed via BFS depth along the optimal path. Wrong audit steps waste your query budget."),
            ("6-Component Reward", "Classification accuracy (20%), finding completeness (25%), finding precision (15%), remediation quality (15%), methodology adherence (15%), efficiency (10%). Designed to resist reward hacking."),
            ("Parameter Randomization", "Company names, deployment dates, regions, and system versions are re-rolled on every reset. Agents must learn the audit process, not memorize specific answers."),
            ("Enforcement: 113 Days Away", "EU AI Act enforcement begins August 2, 2026. Fines up to EUR 35M or 7% of global revenue. Every company deploying AI in Europe needs automated compliance auditing. This environment fills that gap."),
        ]
        html = f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px;">'
        for title, desc in cards:
            html += f'''<div style="background:{_CARD};padding:24px;border-radius:10px;border:1px solid {_BORDER};">
                <h3 style="color:{_TEXT};margin:0 0 10px 0;font-size:1em;font-weight:600;">{title}</h3>
                <p style="color:{_MUTED};margin:0;font-size:0.88em;line-height:1.6;">{desc}</p>
            </div>'''
        return html + "</div>"

    def _scenarios_html():
        diff_colors = {"easy": _EMERALD, "medium": _AMBER, "hard": _ROSE}
        html = ""
        for s in SCENARIO_LIST:
            c = diff_colors.get(s["difficulty"], _MUTED)
            html += f'''<div style="background:{_CARD};padding:18px 20px;border-radius:10px;margin-bottom:10px;border-left:3px solid {c};border:1px solid {_BORDER};display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="color:{_TEXT};font-size:0.98em;font-weight:600;">{s["title"]}</div>
                    <div style="color:{_MUTED};font-size:0.8em;margin-top:3px;font-family:monospace;">{s["id"]}</div>
                </div>
                <span style="background:{c}18;color:{c};padding:4px 14px;border-radius:6px;font-size:0.75em;font-weight:700;letter-spacing:0.04em;">{s["difficulty"].upper()}</span>
            </div>'''
        return html

    with gr.Blocks(title="EU AI Act Compliance Auditor") as landing_app:
        gr.HTML(f'<style>.gradio-container{{background:{_BG}!important;}} .tab-nav button{{color:{_MUTED}!important;background:transparent!important;border:none!important;font-weight:500;}} .tab-nav button.selected{{color:{_GOLD}!important;border-bottom:2px solid {_GOLD}!important;}}</style>')
        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.HTML(_hero())
                gr.HTML(f"<h2 style='color:{_TEXT};margin-bottom:14px;font-weight:600;'>Design Decisions</h2>")
                gr.HTML(_design_cards())
                gr.HTML(f"""<div style="background:{_CARD};padding:24px;border-radius:10px;border:1px solid {_BORDER};margin-top:8px;">
                    <p style="color:{_MUTED};font-size:0.9em;margin:0;line-height:1.7;">
                        <strong style="color:{_TEXT};">compliance_auditor_env</strong> &middot; 6-component reward &middot; 8 scenarios &middot;
                        11 tools &middot; state-graph audit &middot; parameter randomization &middot; deterministic grading
                    </p>
                </div>""")

            with gr.Tab("Scenarios"):
                gr.HTML(f"<h2 style='color:{_TEXT};margin-bottom:14px;font-weight:600;'>8 Compliance Audit Scenarios</h2>")
                gr.HTML(_scenarios_html())

            with gr.Tab("Playground"):
                gr.HTML(f"<h2 style='color:{_TEXT};margin-bottom:8px;font-weight:600;'>Interactive Audit</h2>")
                gr.HTML(f"<p style='color:{_MUTED};margin-bottom:16px;'>Reset to start a session, then call audit tools sequentially.</p>")
                session_state = gr.State(value=None)
                with gr.Row():
                    diff_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Difficulty")
                    reset_btn = gr.Button("Reset", variant="primary")
                with gr.Row():
                    tool_dropdown = gr.Dropdown(
                        choices=["get_system_overview", "classify_system", "check_documentation",
                                 "audit_training_data", "verify_human_oversight", "check_transparency",
                                 "assess_risk_management", "check_logging", "submit_finding",
                                 "recommend_fix", "verify_compliance"],
                        value="get_system_overview", label="Tool")
                    args_input = gr.Textbox(label="Arguments (JSON)", placeholder='{"risk_category": "high_risk"}')
                    call_btn = gr.Button("Call Tool", variant="secondary")
                output_box = gr.JSON(label="Result")

                def pg_reset(difficulty):
                    env = ComplianceAuditorEnvironment()
                    obs = env.reset(difficulty=difficulty)
                    sid = str(uuid.uuid4())
                    with _sessions_lock:
                        _sessions[sid] = env
                    return sid, obs.metadata

                def pg_call(sid, tool_name, args_str):
                    if not sid:
                        return sid, {"error": "Click Reset first"}
                    with _sessions_lock:
                        env = _sessions.get(sid)
                    if not env:
                        return sid, {"error": "Session expired"}
                    fn = env._tool_fns.get(tool_name)
                    if not fn:
                        return sid, {"error": f"Unknown tool: {tool_name}"}
                    try:
                        kwargs = json.loads(args_str) if args_str.strip() else {}
                    except json.JSONDecodeError:
                        return sid, {"error": "Invalid JSON"}
                    try:
                        result = fn(**kwargs)
                        return sid, json.loads(result) if isinstance(result, str) else result
                    except Exception as e:
                        return sid, {"error": str(e)}

                reset_btn.click(pg_reset, [diff_dropdown], [session_state, output_box])
                call_btn.click(pg_call, [session_state, tool_dropdown, args_input], [session_state, output_box])

            with gr.Tab("Try It"):
                gr.HTML(f"""<div style="background:{_CARD};padding:28px;border-radius:10px;border:1px solid {_BORDER};">
                    <h2 style="color:{_TEXT};margin-top:0;font-weight:600;">Run the Baseline Agent</h2>
                    <p style="color:{_MUTED};margin-bottom:16px;">Supports NVIDIA NIM and HuggingFace Inference API.</p>
                    <pre style="background:{_BG};padding:18px;border-radius:8px;color:{_GOLD};overflow-x:auto;border:1px solid {_BORDER};font-size:0.9em;line-height:1.7;"><code>export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="google/gemma-4-31b-it"
export HF_TOKEN="your-api-key"
python inference.py --space https://Itachi1824-compliance-auditor-env.hf.space</code></pre>
                </div>""")

        gr.mount_gradio_app(app, landing_app, path="/")
        gr.mount_gradio_app(app, landing_app, path="/web")
        _gradio_mounted = True

except Exception as e:
    import traceback
    print(f"[WARN] Gradio UI failed to mount: {e}", flush=True)
    traceback.print_exc()

# Fallback root handler if Gradio didn't mount
if not _gradio_mounted:
    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse)
    def root_fallback():
        return """<!DOCTYPE html><html><head><title>EU AI Act Compliance Auditor</title></head>
        <body style="background:#0a0a0a;color:#e0e0ff;font-family:sans-serif;padding:40px;">
        <h1 style="color:#00d4aa;">EU AI Act Compliance Auditor</h1>
        <p>MCP-based environment for auditing AI systems. 8 scenarios, 11 tools, 6-component reward.</p>
        <h3>API Endpoints</h3>
        <ul>
        <li><code>GET /health</code> — Health check</li>
        <li><code>GET /tasks</code> — List scenarios</li>
        <li><code>POST /api/reset</code> — Start audit session</li>
        <li><code>POST /api/call_tool</code> — Call an audit tool</li>
        <li><code>POST /api/close</code> — End session</li>
        </ul></body></html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

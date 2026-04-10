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

try:
    import gradio as gr

    def _build_hero_html():
        scenarios_count = len(SCENARIO_LIST)
        tools_count = 11  # 10 tools + verify
        tiers = len(DIFFICULTY_TIERS)
        return f"""
        <div style="background:linear-gradient(135deg,#0f0f23 0%,#1a1a3e 50%,#0f0f23 100%);padding:40px;border-radius:16px;margin-bottom:24px;border:1px solid #2a2a5a;">
            <h1 style="color:#00d4aa;font-size:2.2em;margin:0 0 8px 0;font-family:'Inter',sans-serif;">EU AI Act Compliance Auditor</h1>
            <p style="color:#8888bb;font-size:1.1em;margin:0 0 24px 0;">
                An MCP-based environment where LLM agents audit AI systems for EU AI Act compliance —
                from risk classification to finding identification to remediation planning.
                Parameter randomization on every reset prevents memorization.
            </p>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px;">
                <div style="background:#1a1a3e;padding:16px;border-radius:12px;text-align:center;border:1px solid #2a2a5a;">
                    <div style="color:#00d4aa;font-size:2em;font-weight:bold;">{scenarios_count}</div>
                    <div style="color:#8888bb;font-size:0.85em;">SCENARIOS</div>
                </div>
                <div style="background:#1a1a3e;padding:16px;border-radius:12px;text-align:center;border:1px solid #2a2a5a;">
                    <div style="color:#00d4aa;font-size:2em;font-weight:bold;">{tools_count}</div>
                    <div style="color:#8888bb;font-size:0.85em;">MCP TOOLS</div>
                </div>
                <div style="background:#1a1a3e;padding:16px;border-radius:12px;text-align:center;border:1px solid #2a2a5a;">
                    <div style="color:#00d4aa;font-size:2em;font-weight:bold;">6</div>
                    <div style="color:#8888bb;font-size:0.85em;">REWARD COMPS</div>
                </div>
                <div style="background:#1a1a3e;padding:16px;border-radius:12px;text-align:center;border:1px solid #2a2a5a;">
                    <div style="color:#00d4aa;font-size:2em;font-weight:bold;">{tiers}</div>
                    <div style="color:#8888bb;font-size:0.85em;">DIFFICULTY TIERS</div>
                </div>
                <div style="background:#1a1a3e;padding:16px;border-radius:12px;text-align:center;border:1px solid #2a2a5a;">
                    <div style="color:#00d4aa;font-size:2em;font-weight:bold;">Aug 2026</div>
                    <div style="color:#8888bb;font-size:0.85em;">EU DEADLINE</div>
                </div>
            </div>
        </div>
        """

    def _build_uniqueness_html():
        cards = [
            ("Real regulatory scenarios", "Scenarios based on actual EU AI Act articles — prohibited social scoring, high-risk hiring AI, deepfake compliance, medical device audits. Not toy problems."),
            ("Full audit workflow", "10 MCP tools mirror a real compliance auditor's toolkit: classify, review documentation, audit data, verify oversight, check transparency, assess risk management."),
            ("State-graph audit process", "Each scenario is a directed graph with progress/no_effect/worsened transitions. Partial credit via BFS depth. Wrong audit steps waste budget."),
            ("6-component reward", "Classification accuracy, finding completeness, finding precision, remediation quality, methodology adherence, and efficiency. Anti-exploit: no reward gaming."),
            ("Parameter randomization", "Company names, deployment dates, regions, and system versions re-rolled on every reset. Agents must learn the AUDIT PROCESS, not memorize answers."),
            ("Timely: Aug 2026 deadline", "EU AI Act enforcement begins August 2, 2026. Every company deploying AI in Europe needs compliance auditing. This environment trains agents for a real, urgent need."),
        ]
        html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px;">'
        for title, desc in cards:
            html += f"""
            <div style="background:#1a1a3e;padding:20px;border-radius:12px;border:1px solid #2a2a5a;">
                <h3 style="color:#00d4aa;margin:0 0 8px 0;font-size:1.05em;">{title}</h3>
                <p style="color:#8888bb;margin:0;font-size:0.9em;line-height:1.5;">{desc}</p>
            </div>
            """
        html += "</div>"
        return html

    def _build_scenarios_html():
        html = ""
        for s in SCENARIO_LIST:
            diff_color = {"easy": "#00d4aa", "medium": "#ffaa00", "hard": "#ff4444"}
            color = diff_color.get(s["difficulty"], "#8888bb")
            html += f"""
            <div style="background:#1a1a3e;padding:16px;border-radius:12px;margin-bottom:12px;border-left:4px solid {color};border:1px solid #2a2a5a;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h3 style="color:#e0e0ff;margin:0;font-size:1.05em;">{s['title']}</h3>
                    <span style="background:{color}22;color:{color};padding:4px 12px;border-radius:8px;font-size:0.8em;font-weight:bold;">{s['difficulty'].upper()}</span>
                </div>
                <p style="color:#8888bb;margin:6px 0 0 0;font-size:0.85em;">ID: {s['id']}</p>
            </div>
            """
        return html

    _css = """
    .gradio-container { background: #0a0a0a !important; }
    .tab-nav button { color: #8888bb !important; background: transparent !important; border: none !important; }
    .tab-nav button.selected { color: #00d4aa !important; border-bottom: 2px solid #00d4aa !important; }
    """

    with gr.Blocks(title="EU AI Act Compliance Auditor") as landing_app:
        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.HTML(_build_hero_html())
                gr.HTML("<h2 style='color:#e0e0ff;margin-bottom:12px;'>What makes this unique</h2>")
                gr.HTML(_build_uniqueness_html())

            with gr.Tab("Scenarios"):
                gr.HTML("<h2 style='color:#e0e0ff;margin-bottom:12px;'>8 Compliance Audit Scenarios</h2>")
                gr.HTML(_build_scenarios_html())

            with gr.Tab("Playground"):
                gr.HTML("<h2 style='color:#e0e0ff;margin-bottom:12px;'>Interactive Playground</h2>")
                gr.HTML("<p style='color:#8888bb;'>Click Reset to start a new audit session, then use the tool dropdown to call audit tools.</p>")

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
                        return sid, {"error": "Session expired. Click Reset."}
                    fn = env._tool_fns.get(tool_name)
                    if not fn:
                        return sid, {"error": f"Unknown tool: {tool_name}"}
                    try:
                        kwargs = json.loads(args_str) if args_str.strip() else {}
                    except json.JSONDecodeError:
                        return sid, {"error": "Invalid JSON in arguments"}
                    try:
                        result = fn(**kwargs)
                        return sid, json.loads(result) if isinstance(result, str) else result
                    except Exception as e:
                        return sid, {"error": str(e)}

                reset_btn.click(pg_reset, [diff_dropdown], [session_state, output_box])
                call_btn.click(pg_call, [session_state, tool_dropdown, args_input], [session_state, output_box])

            with gr.Tab("Try It"):
                gr.HTML("""
                <div style="background:#1a1a3e;padding:24px;border-radius:12px;border:1px solid #2a2a5a;">
                    <h2 style="color:#e0e0ff;margin-top:0;">Run the Baseline Agent</h2>
                    <pre style="background:#0a0a1a;padding:16px;border-radius:8px;color:#00d4aa;overflow-x:auto;"><code>export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="google/gemma-4-31b-it"
export HF_TOKEN="your-api-key"
python inference.py --space https://Itachi1824-compliance-auditor-env.hf.space</code></pre>
                </div>
                """)

        gr.mount_gradio_app(app, landing_app, path="/")
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

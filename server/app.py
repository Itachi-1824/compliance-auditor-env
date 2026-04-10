"""FastAPI application for EU AI Act Compliance Auditor.

Architecture (modeled on Maverick98's winning pattern):
- create_app() for standard OpenEnv endpoints (/reset, /step, /state, /health, /ws)
- Custom HTTP session API (/api/reset, /api/call_tool, /api/close)
- Custom Gradio landing mounted at '/' replacing default inspector
"""

import inspect
import json
import uuid
import asyncio
import uvicorn
from typing import Any, Dict, Optional

from fastapi import Body, HTTPException
from pydantic import BaseModel
from openenv.core.env_server import create_app

from models import ComplianceAction, ComplianceObservation
from server.environment import ComplianceAuditorEnvironment, QUERY_BUDGET
from scenarios.registry import SCENARIO_LIST, DIFFICULTY_TIERS

# ── Create base OpenEnv app ─────────────────────────────────────

app = create_app(
    ComplianceAuditorEnvironment,
    ComplianceAction,
    ComplianceObservation,
    env_name="compliance_auditor_env",
    max_concurrent_envs=5,
)


# ── /tasks endpoint (hackathon validator) ───────────────────────

@app.get("/tasks")
def list_tasks():
    return {"tasks": SCENARIO_LIST}


# ── HTTP Session API ────────────────────────────────────────────

_sessions: Dict[str, ComplianceAuditorEnvironment] = {}
_session_lock = asyncio.Lock()


class ResetBody(BaseModel):
    difficulty: str = "medium"
    scenario_id: Optional[str] = None
    seed: Optional[int] = None


class CallToolBody(BaseModel):
    session_id: str
    tool_name: str
    arguments: Dict[str, Any] = {}


class CloseBody(BaseModel):
    session_id: str


@app.post("/api/reset")
async def api_reset(body: ResetBody = Body(default_factory=ResetBody)):
    """Create session, reset env, return session_id + tools + observation."""
    env = ComplianceAuditorEnvironment()
    obs = env.reset(
        seed=body.seed,
        difficulty=body.difficulty,
        scenario_id=body.scenario_id,
    )
    session_id = str(uuid.uuid4())

    async with _session_lock:
        _sessions[session_id] = env

    # Build tool schemas from _tool_fns
    tools = []
    for name, fn in env._tool_fns.items():
        sig = inspect.signature(fn)
        props = {}
        required = []
        for pname, param in sig.parameters.items():
            ptype = "string"
            if param.annotation == int:
                ptype = "integer"
            props[pname] = {"type": ptype}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        tools.append({
            "name": name,
            "description": (fn.__doc__ or "").strip().split("\n")[0],
            "inputSchema": {"type": "object", "properties": props, "required": required},
        })

    return {
        "session_id": session_id,
        "observation": obs.metadata if hasattr(obs, "metadata") else {},
        "done": obs.done,
        "reward": obs.reward,
        "tools": tools,
    }


@app.post("/api/call_tool")
async def api_call_tool(body: CallToolBody):
    """Call a tool on an existing session."""
    async with _session_lock:
        env = _sessions.get(body.session_id)

    if env is None:
        raise HTTPException(404, f"Session not found: {body.session_id}")

    fn = env._tool_fns.get(body.tool_name)
    if fn is None:
        raise HTTPException(400, f"Tool not found: {body.tool_name}. Available: {list(env._tool_fns.keys())}")

    try:
        result = fn(**body.arguments)
    except Exception as e:
        return {"result": json.dumps({"error": str(e)}), "done": env._done, "reward": env._reward}

    return {"result": result, "done": env._done, "reward": env._reward}


@app.post("/api/close")
async def api_close(body: CloseBody):
    """Close and clean up a session."""
    async with _session_lock:
        env = _sessions.pop(body.session_id, None)
    if env:
        env.close()
    return {"closed": True, "session_id": body.session_id}


# ── Mount Gradio landing at '/' ─────────────────────────────────

try:
    import gradio as gr
    from server.gradio_landing import create_landing_app

    _landing = create_landing_app()
    # Mount at / — exactly like Maverick98's working pattern
    app = gr.mount_gradio_app(app, _landing, path="/")
    print(f"[gradio_landing] mounted at / — gradio {gr.__version__}", flush=True)
except Exception as e:
    import sys
    import traceback
    print(f"[gradio_landing] MOUNT FAILED: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


# ── Entry point ─────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860):
    uvicorn.run(app, host=host, port=port, ws_ping_interval=None, ws_ping_timeout=None)


if __name__ == "__main__":
    main()

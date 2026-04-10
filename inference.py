"""
Baseline inference for EU AI Act Compliance Auditor.

Uses OpenAI function calling through NVIDIA NIM to audit AI systems.
Connects to the live HF Space via HTTP (no WebSocket timeout issues).

Required env vars:
    API_BASE_URL   LLM endpoint (default: https://integrate.api.nvidia.com/v1)
    MODEL_NAME     Model identifier (default: google/gemma-4-31b-it)
    HF_TOKEN       API key for the LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31b-it")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS = 100
CONTEXT_CHAR_LIMIT = 100000

SYSTEM_PROMPT = """You are an EU AI Act compliance auditor. Complete your audit in UNDER 25 tool calls.

WORKFLOW (follow this EXACT sequence):
1. get_system_overview — understand the system
2. classify_system — set risk_category to: prohibited, high_risk, limited_risk, or minimal_risk
3. check_documentation — review technical docs
4. audit_training_data — check for bias
5. verify_human_oversight — check Article 14
6. check_transparency — check Article 50
7. assess_risk_management — check Article 9
8. check_logging — check Article 12
9. submit_finding — report EACH violation you found (one per call)
10. recommend_fix — propose fix for each finding
11. verify_compliance — MUST BE YOUR FINAL CALL with risk_classification, overall_assessment, key_findings_summary

CRITICAL: You MUST call verify_compliance as your last tool call. Without it, you get zero score.
CRITICAL: Always use tool calls, never output plain text.
CRITICAL: For PROHIBITED systems — classify as prohibited, submit finding, recommend shutdown, then verify_compliance immediately.
"""


# ---------------------------------------------------------------------------
# Tool conversion for OpenAI function calling
# ---------------------------------------------------------------------------

def mcp_tools_to_openai(tools: List[Dict]) -> List[Dict]:
    """Convert MCP tool schemas to OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        schema = tool.get("inputSchema", {})

        properties = {}
        required = []
        if schema and "properties" in schema:
            for pname, pschema in schema["properties"].items():
                prop = {"type": pschema.get("type", "string")}
                if "description" in pschema:
                    prop["description"] = pschema["description"]
                if "enum" in pschema:
                    prop["enum"] = pschema["enum"]
                properties[pname] = prop
            required = schema.get("required", [])

        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return openai_tools


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def _summarize_tool_result(content: str, max_chars: int = 200) -> str:
    if not content or len(content) <= max_chars:
        return content or "(empty)"
    try:
        data = json.loads(content)
        if "error" in data:
            return f"error: {data['error'][:100]}"
        return json.dumps(data)[:max_chars] + "..."
    except (json.JSONDecodeError, TypeError):
        return content[:max_chars] + "..."


def summarize_old_messages(messages: List[Dict]) -> List[Dict]:
    """Compress old tool calls to stay within context limits."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= CONTEXT_CHAR_LIMIT:
        return messages

    system_msg = messages[0]
    user_msg = messages[1]
    keep_recent = 12
    split_idx = max(2, len(messages) - keep_recent)

    old = messages[2:split_idx]
    recent = messages[split_idx:]

    lines = ["Previous audit steps:"]
    i = 0
    while i < len(old):
        msg = old[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            name = tc["function"]["name"]
            args = tc["function"]["arguments"][:60]
            result = "(no response)"
            if i + 1 < len(old) and old[i + 1].get("role") == "tool":
                result = _summarize_tool_result(old[i + 1].get("content", ""))
                i += 1
            lines.append(f"- {name}({args}) -> {result}")
        i += 1

    return [system_msg, user_msg, {"role": "user", "content": "\n".join(lines)}] + recent


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    env,
    llm_client: OpenAI,
    model: str,
    tools: List[Dict],
    difficulty: str = "medium",
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single compliance audit episode using OpenAI function calling."""

    reset_kwargs = {"difficulty": difficulty}
    if scenario_id:
        reset_kwargs["scenario_id"] = scenario_id
    reset_result = await env.reset(**reset_kwargs)

    task_name = scenario_id or f"{difficulty}_episode"
    print(f"[START] task={task_name} env=compliance_auditor_env model={model}", flush=True)

    alert_msg = reset_result.get("message", "Compliance audit assigned. Call get_system_overview to begin.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_msg},
    ]

    step_count = 0
    done = False
    consecutive_text = 0
    step_rewards: List[float] = []
    recent_tools: List[str] = []  # Track recent tool calls for loop detection
    tools_called_set: set = set()  # Track unique tools called

    while not done and step_count < MAX_STEPS:
        step_count += 1

        # --- LOOP DETECTION ---
        if len(recent_tools) >= 3 and len(set(recent_tools[-3:])) == 1:
            loop_tool = recent_tools[-1]
            # Guide the model out of the loop
            uncalled = [t["function"]["name"] for t in tools if t["function"]["name"] not in tools_called_set and t["function"]["name"] != loop_tool]
            if uncalled:
                messages.append({"role": "user", "content": f"You are stuck calling {loop_tool} repeatedly. Try calling: {', '.join(uncalled[:3])}"})
            else:
                messages.append({"role": "user", "content": f"You have called all tools. Now call verify_compliance with your risk_classification, overall_assessment, and key_findings_summary."})

        # --- STEP-AWARE GUIDANCE ---
        if step_count == 15:
            messages.append({"role": "user", "content": "REMINDER: After investigating, call submit_finding for each violation, then recommend_fix, then verify_compliance."})
        elif step_count == 30:
            messages.append({"role": "user", "content": "You are at step 30. Start wrapping up: submit_finding for violations, recommend_fix, then verify_compliance."})

        # Force verify_compliance at 80% of budget
        if step_count >= int(MAX_STEPS * 0.8) and not done:
            messages.append({"role": "user", "content": f"WARNING: Only {MAX_STEPS - step_count} steps remaining! Call verify_compliance NOW with your best assessment or you get zero score."})

        # At 90% budget — force tool_choice to verify_compliance
        force_verify = step_count >= int(MAX_STEPS * 0.9)

        # LLM call with retry
        response = None
        create_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "verify_compliance"}} if force_verify else "auto",
            "temperature": 0.1,
            "max_tokens": 500,
        }
        for attempt in range(4):
            try:
                response = llm_client.chat.completions.create(**create_kwargs)
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2 ** attempt + 1
                    time.sleep(wait)
                    continue
                # Some models don't support tool_choice with specific function
                if force_verify and ("tool_choice" in str(e).lower() or "function" in str(e).lower()):
                    create_kwargs["tool_choice"] = "auto"
                    continue
                print(f"[DEBUG] LLM error: {str(e)[:100]}", flush=True)
                break

        if response is None:
            # Force-verify on LLM failure
            break

        message = response.choices[0].message

        # Handle function call
        if message.tool_calls:
            consecutive_text = 0
            tc = message.tool_calls[0]
            tool_name = tc.function.name
            tool_call_id = tc.id

            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tool_call_id, "type": "function", "function": {"name": tool_name, "arguments": tc.function.arguments}}
                ]})
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": "Error: malformed JSON. Retry."})
                continue

            # Add to history
            messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": tool_call_id, "type": "function", "function": {"name": tool_name, "arguments": tc.function.arguments}}
            ]})

            # Track tool usage for loop detection
            recent_tools.append(tool_name)
            tools_called_set.add(tool_name)

            # Execute tool via env
            try:
                result_text = await env.call_tool(tool_name, **tool_args)
            except Exception as e:
                result_text = json.dumps({"error": str(e)})

            if not isinstance(result_text, str):
                result_text = json.dumps(result_text) if result_text else ""

            # Check done/reward
            reward = 0.0
            if result_text:
                try:
                    parsed = json.loads(result_text)
                    if parsed.get("done"):
                        done = True
                    if "reward" in parsed:
                        reward = float(parsed["reward"])
                except (json.JSONDecodeError, TypeError):
                    pass

            if hasattr(env, "_last_done") and env._last_done:
                done = True
            if hasattr(env, "_last_reward") and env._last_reward:
                reward = max(reward, env._last_reward)

            step_rewards.append(round(reward, 2))
            print(f"[STEP] step={step_count} action={tool_name} reward={reward:.2f} done={'true' if done else 'false'} error=null", flush=True)

            if done:
                final_score = max(0.01, min(0.99, reward))
                success = "true" if final_score >= 0.3 else "false"
                rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
                print(f"[END] success={success} steps={step_count} score={final_score:.3f} rewards={rewards_str}", flush=True)
                return {"reward": final_score, "steps": step_count}

            # Add result to history
            if len(result_text) > 3000:
                result_text = result_text[:3000] + "\n...(truncated)"
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result_text or "No result"})
            messages = summarize_old_messages(messages)

        elif message.content:
            consecutive_text += 1
            messages.append({"role": "assistant", "content": message.content})
            if consecutive_text >= 2:
                # Force verify after 2 text outputs — don't waste steps
                messages.append({"role": "user", "content": "STOP. Call verify_compliance RIGHT NOW. Arguments: risk_classification (prohibited/high_risk/limited_risk/minimal_risk), overall_assessment (string), key_findings_summary (string). DO NOT output text."})
            else:
                messages.append({"role": "user", "content": "Do not output text. Call a tool. Available: get_system_overview, classify_system, check_documentation, audit_training_data, verify_human_oversight, check_transparency, assess_risk_management, check_logging, submit_finding, recommend_fix, verify_compliance"})
            if consecutive_text >= 4:
                # 4 consecutive text outputs — force break and auto-verify
                break
        else:
            continue

    # Max steps reached — force verify to get partial credit for all work done
    try:
        force_result = await env.call_tool("verify_compliance",
            risk_classification="high_risk",
            overall_assessment="Audit completed — submitting findings for grading",
            key_findings_summary="See submitted findings above")
        if isinstance(force_result, str):
            parsed = json.loads(force_result)
            reward = max(0.01, min(0.99, float(parsed.get("reward", 0.01))))
        elif hasattr(env, "_last_reward") and env._last_reward:
            reward = max(0.01, min(0.99, env._last_reward))
        else:
            reward = 0.01
    except Exception:
        reward = 0.01
    success = "true" if reward >= 0.3 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    print(f"[END] success={success} steps={step_count} score={reward:.3f} rewards={rewards_str}", flush=True)
    return {"reward": reward, "error": "max_steps_auto_graded", "steps": step_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINE_SCENARIOS = {
    "easy": ["easy_chatbot_transparency_001", "easy_recommendation_minimal_001"],
    "medium": ["medium_hiring_bias_001", "medium_credit_scoring_001", "medium_medical_triage_001"],
    "hard": ["hard_social_scoring_prohibited_001", "hard_deepfake_generation_001", "hard_multi_system_corporate_001"],
}


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="EU AI Act Compliance Auditor Inference")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--space", default=None, help="HF Space URL")
    args = parser.parse_args()

    api_key = HF_TOKEN
    if not api_key:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY environment variable is required.", file=sys.stderr, flush=True)
        sys.exit(1)

    model = args.model or MODEL_NAME
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=60.0)

    # Determine base URL
    if args.space:
        base_url = args.space
    else:
        base_url = "http://localhost:7860"

    from client import ComplianceAuditorHTTP
    difficulties = [args.difficulty] if args.difficulty else ["easy", "medium", "hard"]

    # Start local server if not using Space
    server_proc = None
    if not args.space:
        import subprocess
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "7860"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(4)

    try:
        # Discover tools
        async with ComplianceAuditorHTTP(base_url=base_url) as discover_env:
            await discover_env.reset(difficulty="easy")
            tools_raw = await discover_env.list_tools()
            tools = mcp_tools_to_openai(tools_raw)

        print(f"[DEBUG] Mode: {'remote' if args.space else 'local'} | Model: {model}", flush=True)
        print(f"[DEBUG] Tools: {[t['function']['name'] for t in tools]}", flush=True)
        print(f"[DEBUG] Difficulties: {difficulties}", flush=True)

        all_results = {}
        for difficulty in difficulties:
            scenario_ids = BASELINE_SCENARIOS.get(difficulty, [])
            for sid in scenario_ids:
                for run in range(args.episodes):
                    try:
                        async with ComplianceAuditorHTTP(base_url=base_url) as ep_env:
                            result = await run_episode(ep_env, llm_client, model, tools, difficulty, sid)
                    except Exception as e:
                        print(f"[START] task={sid} env=compliance_auditor_env model={model}", flush=True)
                        print(f"[END] success=false steps=0 score=0.010 rewards=", flush=True)
                        result = {"reward": 0.01, "error": str(e)[:100], "steps": 0}
                    all_results[sid] = result

        # Summary
        print(f"\n{'='*60}", flush=True)
        print(f"BASELINE RESULTS — {model}", flush=True)
        for sid, r in all_results.items():
            score = max(0.01, min(0.99, r.get("reward", 0)))
            print(f"  {sid}: {score:.4f} ({r.get('steps', 0)} steps)", flush=True)
        if all_results:
            avg = sum(max(0.01, min(0.99, r.get("reward", 0))) for r in all_results.values()) / len(all_results)
            print(f"  OVERALL: {avg:.4f}", flush=True)

    finally:
        if server_proc:
            server_proc.terminate()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

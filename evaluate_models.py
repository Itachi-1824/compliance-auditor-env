"""
Multi-model leaderboard evaluation for EU AI Act Compliance Auditor.

Runs baseline episodes across multiple LLM models via NIM and HF Inference.
Outputs a leaderboard JSON for the Gradio dashboard.

Usage:
    # NIM models
    python evaluate_models.py --provider nim --space https://Itachi1824-compliance-auditor-env.hf.space

    # HF free models
    python evaluate_models.py --provider hf --space https://Itachi1824-compliance-auditor-env.hf.space

    # Single model test
    python evaluate_models.py --model google/gemma-4-31b-it --space https://...
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import ComplianceAuditorHTTP
from inference import mcp_tools_to_openai, run_episode

# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

NIM_BASE = "https://integrate.api.nvidia.com/v1"
HF_BASE = "https://router.huggingface.co/v1"

# Top NIM models for tool-calling (curated from nim-top50.txt)
NIM_MODELS = [
    "google/gemma-4-31b-it",
    "deepseek-ai/deepseek-v3.1",
    "qwen/qwen3.5-122b-a10b",
    "meta/llama-4-maverick-17b-128e-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "mistralai/mistral-small-4-119b-2603",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "stepfun-ai/step-3.5-flash",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "qwen/qwq-32b",
    "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "bytedance/seed-oss-36b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct-2503",
    "google/gemma-3-27b-it",
]

# HF free models (available via router.huggingface.co)
HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "google/gemma-2-27b-it",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
]

# Scenarios to test (1 per tier for speed)
EVAL_SCENARIOS = {
    "easy": "easy_chatbot_transparency_001",
    "medium": "medium_hiring_bias_001",
    "hard": "hard_social_scoring_prohibited_001",
}


async def evaluate_model(
    model: str,
    base_url: str,
    api_key: str,
    space_url: str,
) -> Dict[str, Any]:
    """Evaluate a single model across all difficulty tiers."""
    llm = OpenAI(base_url=base_url, api_key=api_key)
    results = {}

    # Discover tools once
    try:
        async with ComplianceAuditorHTTP(base_url=space_url) as env:
            await env.reset(difficulty="easy")
            tools_raw = await env.list_tools()
            tools = mcp_tools_to_openai(tools_raw)
    except Exception as e:
        print(f"  [SKIP] {model}: tool discovery failed: {e}")
        return {"model": model, "error": str(e), "scores": {}}

    scores = {}
    for tier, scenario_id in EVAL_SCENARIOS.items():
        try:
            async with ComplianceAuditorHTTP(base_url=space_url) as ep_env:
                result = await run_episode(
                    ep_env, llm, model, tools,
                    difficulty=tier, scenario_id=scenario_id,
                )
                score = max(0.01, min(0.99, result.get("reward", 0.01)))
                scores[scenario_id] = round(score, 4)
                print(f"  {tier}: {score:.4f} ({result.get('steps', 0)} steps)")
        except Exception as e:
            scores[scenario_id] = 0.01
            print(f"  {tier}: FAILED ({e})")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return {"model": model, "scores": scores, "average": round(avg, 4)}


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["nim", "hf", "both"], default="nim")
    parser.add_argument("--model", default=None, help="Single model to test")
    parser.add_argument("--space", required=True, help="HF Space URL")
    parser.add_argument("--output", default="outputs/leaderboard/scores.json")
    args = parser.parse_args()

    api_key = os.getenv("HF_TOKEN") or os.getenv("NVIDIA_API_KEY") or ""

    if args.model:
        models = [(args.model, NIM_BASE)]
    else:
        models = []
        if args.provider in ("nim", "both"):
            models.extend([(m, NIM_BASE) for m in NIM_MODELS])
        if args.provider in ("hf", "both"):
            models.extend([(m, HF_BASE) for m in HF_MODELS])

    print(f"Evaluating {len(models)} models against {args.space}")
    print(f"Scenarios: {list(EVAL_SCENARIOS.values())}")
    print("=" * 60)

    all_results = []
    for model, base_url in models:
        print(f"\n{model} ({base_url.split('/')[2]})")
        result = await evaluate_model(model, base_url, api_key, args.space)
        all_results.append(result)

        # Save incrementally
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print leaderboard
    print(f"\n{'='*60}")
    print("LEADERBOARD")
    print(f"{'='*60}")
    sorted_results = sorted(all_results, key=lambda r: r.get("average", 0), reverse=True)
    for i, r in enumerate(sorted_results, 1):
        avg = r.get("average", 0)
        print(f"  {i:2d}. {avg:.4f}  {r['model']}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

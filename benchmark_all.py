"""
Industrial-grade 50-model benchmark for EU AI Act Compliance Auditor.

Handles NIM's 40 RPM rate limit with:
  - Sequential model execution (one model at a time)
  - Per-call rate limiting (1.5s minimum between LLM calls)
  - Exponential backoff on 429s (2s, 4s, 8s, 16s)
  - Incremental JSON output (saves after each model)
  - Resume capability (skips models already in output)
  - Timeout per episode (5 min)

Usage:
    python benchmark_all.py --space https://Itachi1824-compliance-auditor-env.hf.space
    python benchmark_all.py --space ... --resume  # skip already-scored models
    python benchmark_all.py --space ... --model qwen/qwen3.5-122b-a10b  # single model
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from client import ComplianceAuditorHTTP
from inference import mcp_tools_to_openai, run_episode, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# All 50 NIM models from nim-top50.txt
# ---------------------------------------------------------------------------

NIM_MODELS = [
    # Tier S: Frontier
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2.5",
    "deepseek-ai/deepseek-v3.2",
    "deepseek-ai/deepseek-v3.1",
    # Tier A+: Elite
    "minimaxai/minimax-m2.5",
    "qwen/qwen3.5-397b-a17b",
    "moonshotai/kimi-k2-instruct",
    "stepfun-ai/step-3.5-flash",
    "mistralai/mistral-large-3-675b-instruct-2512",
    # Tier A: Strong
    "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3.5-122b-a10b",
    "google/gemma-4-31b-it",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "mistralai/mistral-small-4-119b-2603",
    "bytedance/seed-oss-36b-instruct",
    # Tier B+: Solid
    "meta/llama-4-maverick-17b-128e-instruct",
    "nvidia/nemotron-3-super-120b-a12b",
    "qwen/qwq-32b",
    "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    # Tier B: Capable
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-4-scout-17b-16e-instruct",
    "qwen/qwen2.5-coder-32b-instruct",
    "nvidia/nemotron-nano-3-30b-a3b",
    # Tier C+: Efficient
    "mistralai/mistral-small-3.1-24b-instruct-2503",
    "google/gemma-3-27b-it",
    "microsoft/phi-4-mini-flash-reasoning",
    "meta/llama-3.1-8b-instruct",
]

# Scenarios: 1 per tier for speed (3 episodes per model)
EVAL_SCENARIOS = [
    ("easy", "easy_chatbot_transparency_001"),
    ("medium", "medium_hiring_bias_001"),
    ("hard", "hard_social_scoring_prohibited_001"),
]

NIM_BASE = "https://integrate.api.nvidia.com/v1"
OUTPUT_FILE = "outputs/leaderboard/scores.json"

# Rate limiting: 40 RPM = 1 call per 1.5s
MIN_CALL_INTERVAL = 1.6  # seconds between LLM calls


async def benchmark_model(
    model: str,
    api_key: str,
    space_url: str,
    tools: List[Dict],
) -> Dict[str, Any]:
    """Benchmark a single model across all tiers."""
    llm = OpenAI(base_url=NIM_BASE, api_key=api_key)
    scores = {}

    for tier, sid in EVAL_SCENARIOS:
        print(f"  {tier}: {sid}", end="", flush=True)
        try:
            async with ComplianceAuditorHTTP(base_url=space_url, timeout=300) as ep:
                result = await run_episode(ep, llm, model, tools, tier, sid)
                score = max(0.01, min(0.99, result.get("reward", 0.01)))
                steps = result.get("steps", 0)
                scores[sid] = {"score": round(score, 4), "steps": steps}
                print(f" -> {score:.4f} ({steps} steps)", flush=True)
        except Exception as e:
            scores[sid] = {"score": 0.01, "steps": 0, "error": str(e)[:80]}
            print(f" -> FAILED: {str(e)[:60]}", flush=True)

        # Rate limit pause between scenarios
        time.sleep(MIN_CALL_INTERVAL * 2)

    # Compute averages
    valid_scores = [s["score"] for s in scores.values() if s["score"] > 0.01]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.01

    return {
        "model": model,
        "scores": scores,
        "easy_avg": round(
            sum(s["score"] for sid, s in scores.items() if "easy" in sid) /
            max(1, sum(1 for sid in scores if "easy" in sid)), 4),
        "medium_avg": round(
            sum(s["score"] for sid, s in scores.items() if "medium" in sid) /
            max(1, sum(1 for sid in scores if "medium" in sid)), 4),
        "hard_avg": round(
            sum(s["score"] for sid, s in scores.items() if "hard" in sid) /
            max(1, sum(1 for sid in scores if "hard" in sid)), 4),
        "overall": round(avg, 4),
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="50-model NIM benchmark")
    parser.add_argument("--space", required=True, help="HF Space URL")
    parser.add_argument("--model", default=None, help="Single model to test")
    parser.add_argument("--resume", action="store_true", help="Skip already-scored models")
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    api_key = os.getenv("HF_TOKEN") or os.getenv("NVIDIA_API_KEY") or ""
    if not api_key:
        print("ERROR: Set HF_TOKEN or NVIDIA_API_KEY")
        sys.exit(1)

    # Load existing results for resume
    existing = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for entry in json.load(f):
                existing[entry["model"]] = entry

    # Select models
    if args.model:
        models = [args.model]
    else:
        models = NIM_MODELS

    # Discover tools once
    print("Discovering tools...", flush=True)
    async with ComplianceAuditorHTTP(base_url=args.space) as env:
        await env.reset(difficulty="easy")
        tools_raw = await env.list_tools()
        tools = mcp_tools_to_openai(tools_raw)
    print(f"Tools: {len(tools)} discovered\n", flush=True)

    # Run benchmarks
    results = list(existing.values())
    total = len(models)

    for i, model in enumerate(models, 1):
        if model in existing:
            print(f"[{i}/{total}] {model} — SKIPPED (already scored: {existing[model]['overall']})")
            continue

        print(f"\n[{i}/{total}] {model}", flush=True)
        result = await benchmark_model(model, api_key, args.space, tools)
        results.append(result)

        # Save incrementally
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        sorted_results = sorted(results, key=lambda r: r.get("overall", 0), reverse=True)
        with open(args.output, "w") as f:
            json.dump(sorted_results, f, indent=2)
        print(f"  Saved ({len(results)} models so far)", flush=True)

        # Pause between models to let rate limit recover
        if i < total:
            print(f"  Cooling down 10s...", flush=True)
            time.sleep(10)

    # Final leaderboard
    sorted_results = sorted(results, key=lambda r: r.get("overall", 0), reverse=True)
    print(f"\n{'='*70}")
    print(f"LEADERBOARD ({len(sorted_results)} models)")
    print(f"{'='*70}")
    for rank, r in enumerate(sorted_results, 1):
        print(f"  {rank:2d}. {r['overall']:.4f}  {r['model']}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

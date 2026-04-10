"""
Leaderboard benchmark runner — 10 models across 3 NIM API keys.

Distributes models across keys to maximize throughput (40 RPM per key).
Runs all 9 fixed scenarios per model. Saves results to outputs/leaderboard/scores.json.

Usage:
  set NVIDIA_API_KEY_1=nvapi-...
  set NVIDIA_API_KEY_2=nvapi-...
  set NVIDIA_API_KEY_3=nvapi-...
  python benchmark_leaderboard.py --space https://Itachi1824-compliance-auditor-env.hf.space
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

# Import from our inference module
from inference import run_episode, mcp_tools_to_openai
from client import ComplianceAuditorHTTP
from scenarios.registry import SCENARIO_LIST

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://integrate.api.nvidia.com/v1"

# 10 models distributed across 3 API keys for parallel execution
MODEL_GROUPS = [
    # Key 1: Tier S + A models (4 models)
    {
        "key_env": "NVIDIA_API_KEY_1",
        "models": [
            "deepseek-ai/deepseek-v3.1",
            "stepfun-ai/step-3.5-flash",
            "qwen/qwen3.5-122b-a10b",
            "meta/llama-4-scout-17b-16e-instruct",
        ],
    },
    # Key 2: Tier A models (3 models)
    {
        "key_env": "NVIDIA_API_KEY_2",
        "models": [
            "mistralai/mistral-large-3-675b-instruct-2512",
            "google/gemma-4-31b-it",
            "meta/llama-4-maverick-17b-128e-instruct",
        ],
    },
    # Key 3: Tier A/B models (3 models)
    {
        "key_env": "NVIDIA_API_KEY_3",
        "models": [
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "nvidia/nemotron-3-super-120b-a12b",
            "meta/llama-3.3-70b-instruct",
        ],
    },
]

SCENARIOS = [s["id"] for s in SCENARIO_LIST if not s["id"].startswith("procedural")]


async def benchmark_model(
    model: str,
    api_key: str,
    base_url: str,
    tools: List[Dict],
) -> Dict:
    """Run all scenarios for a single model."""
    llm = OpenAI(base_url=API_BASE, api_key=api_key, timeout=120.0)
    results = {}

    for sid in SCENARIOS:
        difficulty = next(s["difficulty"] for s in SCENARIO_LIST if s["id"] == sid)
        try:
            async with ComplianceAuditorHTTP(base_url=base_url) as env:
                result = await run_episode(env, llm, model, tools, difficulty, sid)
                score = max(0.001, min(0.999, result.get("reward", 0.01)))
                results[sid] = {"score": round(score, 4), "steps": result.get("steps", 0)}
                print(f"  {model:50s} | {sid:50s} | score={score:.4f} | steps={result.get('steps', 0)}", flush=True)
        except Exception as e:
            err_msg = str(e)[:80]
            print(f"  {model:50s} | {sid:50s} | ERROR: {err_msg}", flush=True)
            results[sid] = {"score": 0.01, "steps": 0, "error": err_msg}

        # Rate limit: ~2s between episodes to stay under 40 RPM
        await asyncio.sleep(2)

    return results


async def benchmark_group(
    group: Dict,
    base_url: str,
    tools: List[Dict],
) -> List[Dict]:
    """Run all models in a key group sequentially (same API key)."""
    key = os.environ.get(group["key_env"], "")
    if not key:
        print(f"WARNING: {group['key_env']} not set — skipping {len(group['models'])} models", flush=True)
        return []

    entries = []
    for model in group["models"]:
        print(f"\n{'='*60}", flush=True)
        print(f"BENCHMARKING: {model}", flush=True)
        print(f"  Key: {group['key_env']} | Scenarios: {len(SCENARIOS)}", flush=True)
        print(f"{'='*60}", flush=True)

        start = time.time()
        scores = await benchmark_model(model, key, base_url, tools)
        elapsed = time.time() - start

        # Compute averages
        all_scores = [v["score"] for v in scores.values() if "error" not in v]
        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

        tier_avgs = {}
        for tier in ["easy", "medium", "hard"]:
            tier_scores = [
                v["score"] for sid, v in scores.items()
                if next((s["difficulty"] for s in SCENARIO_LIST if s["id"] == sid), "") == tier
                and "error" not in v
            ]
            tier_avgs[tier] = sum(tier_scores) / len(tier_scores) if tier_scores else 0.0

        entry = {
            "model": model,
            "scores": scores,
            "overall": round(avg, 4),
            "tier_averages": {k: round(v, 4) for k, v in tier_avgs.items()},
            "elapsed_seconds": round(elapsed, 1),
        }
        entries.append(entry)

        print(f"\n  RESULT: {model}", flush=True)
        print(f"    Overall: {avg:.4f}", flush=True)
        for tier, tavg in tier_avgs.items():
            print(f"    {tier}: {tavg:.4f}", flush=True)
        print(f"    Time: {elapsed:.0f}s", flush=True)

    return entries


async def main():
    parser = argparse.ArgumentParser(description="Leaderboard benchmark — 10 models")
    parser.add_argument("--space", required=True, help="HF Space URL")
    parser.add_argument("--output", default="outputs/leaderboard/scores.json")
    args = parser.parse_args()

    base_url = args.space.rstrip("/")
    print(f"Benchmarking against: {base_url}", flush=True)
    print(f"Scenarios: {len(SCENARIOS)}", flush=True)
    print(f"Model groups: {len(MODEL_GROUPS)} ({sum(len(g['models']) for g in MODEL_GROUPS)} total models)", flush=True)

    # Discover tools from the environment
    async with ComplianceAuditorHTTP(base_url=base_url) as env:
        await env.reset(difficulty="easy")
        tools_raw = await env.list_tools()
        tools = mcp_tools_to_openai(tools_raw)
    print(f"Tools discovered: {len(tools)}", flush=True)

    # Run all groups in parallel (one per API key)
    tasks = [benchmark_group(g, base_url, tools) for g in MODEL_GROUPS]
    group_results = await asyncio.gather(*tasks)

    # Flatten and save
    all_entries = []
    for group_entries in group_results:
        all_entries.extend(group_entries)

    # Sort by overall score descending
    all_entries.sort(key=lambda e: e["overall"], reverse=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("LEADERBOARD RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for i, entry in enumerate(all_entries, 1):
        m = entry["model"].split("/")[-1][:30]
        print(f"  {i:2d}. {m:30s} | overall={entry['overall']:.4f} | "
              f"easy={entry['tier_averages'].get('easy', 0):.4f} | "
              f"medium={entry['tier_averages'].get('medium', 0):.4f} | "
              f"hard={entry['tier_averages'].get('hard', 0):.4f}",
              flush=True)

    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

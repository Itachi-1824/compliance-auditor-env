"""
Leaderboard benchmark — 10 models, 3 NIM keys, parallel execution.
Resilient to network errors with retries per episode.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from client import ComplianceAuditorHTTP
from inference import run_episode, mcp_tools_to_openai
from scenarios.registry import SCENARIO_LIST

API_BASE = "https://integrate.api.nvidia.com/v1"
SPACE_URL = "https://Itachi1824-compliance-auditor-env.hf.space"
MAX_RETRIES = 2

KEYS = {
    1: os.environ.get("NVIDIA_API_KEY_1", ""),
    2: os.environ.get("NVIDIA_API_KEY_2", ""),
    3: os.environ.get("NVIDIA_API_KEY_3", ""),
}

GROUPS = [
    # Key 1
    ["stepfun-ai/step-3.5-flash", "deepseek-ai/deepseek-v3.1", "qwen/qwen3.5-122b-a10b"],
    # Key 2
    ["meta/llama-4-maverick-17b-128e-instruct", "google/gemma-4-31b-it", "nvidia/llama-3.1-nemotron-ultra-253b-v1", "meta/llama-4-scout-17b-16e-instruct"],
    # Key 3
    ["mistralai/mistral-large-3-675b-instruct-2512", "nvidia/nemotron-3-super-120b-a12b", "meta/llama-3.3-70b-instruct"],
]

SCENARIOS = [(s["id"], s["difficulty"]) for s in SCENARIO_LIST if not s["id"].startswith("procedural")]


async def run_one_episode(model, api_key, tools, sid, diff):
    """Run one episode with retries."""
    llm = OpenAI(base_url=API_BASE, api_key=api_key, timeout=120.0)
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with ComplianceAuditorHTTP(base_url=SPACE_URL) as env:
                result = await run_episode(env, llm, model, tools, diff, sid)
                score = max(0.001, min(0.999, result.get("reward", 0.01)))
                return round(score, 4)
        except Exception as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(3)
                continue
            return 0.01


async def run_model(model, api_key, tools, progress):
    """Run all scenarios for one model."""
    short = model.split("/")[-1][:28]
    scores = {}
    for sid, diff in SCENARIOS:
        score = await run_one_episode(model, api_key, tools, sid, diff)
        scores[sid] = score
        progress["done"] += 1
        total = progress["total"]
        pct = progress["done"] / total * 100
        print(f"  [{pct:5.1f}%] {short:28s} | {sid:50s} | {score:.4f}", flush=True)
        await asyncio.sleep(1.5)
    return scores


async def run_group(group_idx, models, api_key, tools, progress):
    """Run all models in a key group sequentially."""
    if not api_key:
        print(f"  Key {group_idx+1} not set — skipping {len(models)} models", flush=True)
        return []
    entries = []
    for model in models:
        short = model.split("/")[-1][:28]
        print(f"\n{'='*70}\n  KEY {group_idx+1} | {model}\n{'='*70}", flush=True)
        t0 = time.time()
        scores = await run_model(model, api_key, tools, progress)
        elapsed = time.time() - t0

        all_s = list(scores.values())
        avg = sum(all_s) / len(all_s) if all_s else 0
        tiers = {"easy": [], "medium": [], "hard": []}
        for sid, diff in SCENARIOS:
            if sid in scores:
                tiers[diff].append(scores[sid])
        tier_avgs = {t: (sum(v)/len(v) if v else 0) for t, v in tiers.items()}

        entries.append({
            "model": model,
            "scores": scores,
            "overall": round(avg, 4),
            "tier_averages": {k: round(v, 4) for k, v in tier_avgs.items()},
            "elapsed_seconds": round(elapsed, 1),
        })
        print(f"  DONE: {short:28s} | overall={avg:.4f} | e={tier_avgs['easy']:.4f} m={tier_avgs['medium']:.4f} h={tier_avgs['hard']:.4f} | {elapsed:.0f}s", flush=True)
    return entries


async def main():
    total_models = sum(len(g) for g in GROUPS)
    total_episodes = total_models * len(SCENARIOS)
    print(f"Benchmarking {total_models} models x {len(SCENARIOS)} scenarios = {total_episodes} episodes", flush=True)
    print(f"Space: {SPACE_URL}", flush=True)

    # Discover tools
    async with ComplianceAuditorHTTP(base_url=SPACE_URL) as env:
        await env.reset(difficulty="easy")
        tools = mcp_tools_to_openai(await env.list_tools())
    print(f"Tools: {len(tools)}\n", flush=True)

    progress = {"done": 0, "total": total_episodes}

    # Run 3 groups in parallel
    tasks = [run_group(i, GROUPS[i], KEYS[i+1], tools, progress) for i in range(3)]
    results = await asyncio.gather(*tasks)

    # Flatten + sort
    all_entries = [e for group in results for e in group]
    all_entries.sort(key=lambda e: e["overall"], reverse=True)

    # Save
    out = Path("outputs/leaderboard/scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print("FINAL LEADERBOARD", flush=True)
    print(f"{'='*70}", flush=True)
    for i, e in enumerate(all_entries, 1):
        m = e["model"].split("/")[-1][:28]
        print(f"  {i:2d}. {m:28s} | {e['overall']:.4f} | e={e['tier_averages']['easy']:.4f} m={e['tier_averages']['medium']:.4f} h={e['tier_averages']['hard']:.4f}", flush=True)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

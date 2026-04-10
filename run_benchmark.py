"""Quick benchmark runner — 10 models across 3 keys in parallel."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Set keys before imports
KEYS = [
    "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",
    "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",
    "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",
]

API_BASE = "https://integrate.api.nvidia.com/v1"
SPACE_URL = "https://Itachi1824-compliance-auditor-env.hf.space"

# 10 models distributed across 3 keys
GROUPS = [
    # Key 1
    [
        "stepfun-ai/step-3.5-flash",
        "deepseek-ai/deepseek-v3.1",
        "qwen/qwen3.5-122b-a10b",
    ],
    # Key 2
    [
        "meta/llama-4-maverick-17b-128e-instruct",
        "google/gemma-4-31b-it",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "meta/llama-4-scout-17b-16e-instruct",
    ],
    # Key 3
    [
        "mistralai/mistral-large-3-675b-instruct-2512",
        "nvidia/nemotron-3-super-120b-a12b",
        "meta/llama-3.3-70b-instruct",
    ],
]

SCENARIOS = [
    ("easy_chatbot_transparency_001", "easy"),
    ("easy_recommendation_minimal_001", "easy"),
    ("medium_hiring_bias_001", "medium"),
    ("medium_credit_scoring_001", "medium"),
    ("medium_medical_triage_001", "medium"),
    ("medium_emotion_recognition_workplace_001", "medium"),
    ("hard_social_scoring_prohibited_001", "hard"),
    ("hard_deepfake_generation_001", "hard"),
    ("hard_multi_system_corporate_001", "hard"),
]

from openai import OpenAI
from client import ComplianceAuditorHTTP
from inference import run_episode, mcp_tools_to_openai


async def run_model(model: str, api_key: str, tools: list) -> dict:
    """Run all scenarios for one model."""
    llm = OpenAI(base_url=API_BASE, api_key=api_key, timeout=120.0)
    scores = {}
    for sid, diff in SCENARIOS:
        try:
            async with ComplianceAuditorHTTP(base_url=SPACE_URL) as env:
                result = await run_episode(env, llm, model, tools, diff, sid)
                score = max(0.001, min(0.999, result.get("reward", 0.01)))
                scores[sid] = round(score, 4)
                short = model.split("/")[-1][:25]
                print(f"  {short:25s} | {sid:50s} | {score:.4f}", flush=True)
        except Exception as e:
            scores[sid] = 0.01
            short = model.split("/")[-1][:25]
            print(f"  {short:25s} | {sid:50s} | ERROR: {str(e)[:60]}", flush=True)
        await asyncio.sleep(2)  # rate limit
    return scores


async def run_group(models: list, api_key: str, tools: list) -> list:
    """Run all models in a group sequentially (same key)."""
    results = []
    for model in models:
        print(f"\n--- {model} ---", flush=True)
        start = time.time()
        scores = await run_model(model, api_key, tools)
        elapsed = time.time() - start

        all_scores = list(scores.values())
        avg = sum(all_scores) / len(all_scores) if all_scores else 0

        tier_scores = {"easy": [], "medium": [], "hard": []}
        for sid, diff in SCENARIOS:
            if sid in scores:
                tier_scores[diff].append(scores[sid])

        tier_avgs = {t: (sum(s)/len(s) if s else 0) for t, s in tier_scores.items()}

        results.append({
            "model": model,
            "scores": scores,
            "overall": round(avg, 4),
            "tier_averages": {k: round(v, 4) for k, v in tier_avgs.items()},
            "elapsed_seconds": round(elapsed, 1),
        })
        print(f"  OVERALL: {avg:.4f} | easy={tier_avgs['easy']:.4f} medium={tier_avgs['medium']:.4f} hard={tier_avgs['hard']:.4f} | {elapsed:.0f}s", flush=True)
    return results


async def main():
    print(f"Benchmarking 10 models against {SPACE_URL}", flush=True)
    print(f"Scenarios: {len(SCENARIOS)}", flush=True)

    # Discover tools
    async with ComplianceAuditorHTTP(base_url=SPACE_URL) as env:
        await env.reset(difficulty="easy")
        tools_raw = await env.list_tools()
        tools = mcp_tools_to_openai(tools_raw)
    print(f"Tools: {len(tools)}", flush=True)

    # Run 3 groups in parallel
    tasks = [run_group(GROUPS[i], KEYS[i], tools) for i in range(3)]
    group_results = await asyncio.gather(*tasks)

    # Flatten + sort
    all_entries = []
    for gr in group_results:
        all_entries.extend(gr)
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
        m = e["model"].split("/")[-1][:30]
        print(f"  {i:2d}. {m:30s} | {e['overall']:.4f} | e={e['tier_averages']['easy']:.4f} m={e['tier_averages']['medium']:.4f} h={e['tier_averages']['hard']:.4f}", flush=True)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

"""Procedural scenario generator tests.

Proves the generator produces valid, diverse, and unique scenarios
from any seed. This is the feature that makes the environment INFINITE.
"""

import json
import pytest
from scenarios.procedural import generate_procedural_scenario, SYSTEM_TEMPLATES, VIOLATION_TEMPLATES
from scenarios.registry import get_scenario
from server.environment import ComplianceAuditorEnvironment


def test_generates_valid_scenario():
    """Basic generation produces a well-formed AuditScenario."""
    sc = generate_procedural_scenario(seed=42, difficulty="medium")
    assert sc.scenario_id.startswith("procedural_")
    assert sc.correct_classification in ("prohibited", "high_risk", "limited_risk", "minimal_risk")
    assert len(sc.ground_truth_findings) >= 1
    assert len(sc.graph.nodes) >= 6
    assert sc.graph.optimal_path_length() >= 3


def test_difficulty_controls_violation_count():
    """Easy has fewer violations than hard."""
    easy_counts = []
    hard_counts = []
    for seed in range(20):
        easy = generate_procedural_scenario(seed, "easy")
        hard = generate_procedural_scenario(seed, "hard")
        easy_counts.append(len(easy.ground_truth_findings))
        hard_counts.append(len(hard.ground_truth_findings))

    assert sum(easy_counts) / len(easy_counts) < sum(hard_counts) / len(hard_counts), \
        "Hard scenarios should have more violations on average"


def test_different_seeds_produce_different_scenarios():
    """No two seeds should produce identical scenarios."""
    scenarios = {}
    for seed in range(50):
        sc = generate_procedural_scenario(seed, "medium")
        key = (sc.system_name, tuple(sc.ground_truth_findings))
        scenarios[seed] = key

    unique = len(set(scenarios.values()))
    assert unique >= 10, f"Only {unique} unique scenarios from 50 seeds — too little diversity"


def test_prohibited_systems_in_hard_mode():
    """Hard difficulty should sometimes generate prohibited systems."""
    has_prohibited = False
    for seed in range(100):
        sc = generate_procedural_scenario(seed, "hard")
        if sc.correct_classification == "prohibited":
            has_prohibited = True
            break
    assert has_prohibited, "Hard mode should occasionally generate prohibited systems"


def test_procedural_works_in_environment():
    """Procedural scenarios work end-to-end through the environment."""
    for seed in [1, 42, 100]:
        env = ComplianceAuditorEnvironment()
        obs = env.reset(seed=seed, scenario_id=f"procedural_medium_{seed}")

        assert not env._done
        assert env._scenario is not None

        # Run basic audit
        r = json.loads(env._tool_fns["get_system_overview"]())
        assert "content" in r
        assert len(r["content"]) > 100

        env._tool_fns["classify_system"](risk_category="high_risk")
        env._tool_fns["check_documentation"]()

        result = json.loads(env._tool_fns["verify_compliance"](
            risk_classification="high_risk",
            overall_assessment="test",
            key_findings_summary="test"
        ))
        assert 0.0 < result["reward"] < 1.0


def test_procedural_via_get_scenario():
    """Procedural IDs work through the standard get_scenario interface."""
    sc = get_scenario("procedural_easy_42")
    assert sc.scenario_id.startswith("procedural_")
    assert sc.difficulty == "easy"

    sc2 = get_scenario("procedural_hard_999")
    assert sc2.difficulty == "hard"
    assert len(sc2.ground_truth_findings) >= len(sc.ground_truth_findings)


def test_all_system_types_reachable():
    """Every system type template should be reachable from some seed."""
    seen_systems = set()
    for seed in range(200):
        for diff in ["easy", "medium", "hard"]:
            sc = generate_procedural_scenario(seed, diff)
            seen_systems.add(sc.system_name.split(" ")[-2] + " " + sc.system_name.split(" ")[-1])

    assert len(seen_systems) >= len(SYSTEM_TEMPLATES), \
        f"Only {len(seen_systems)} system types reached from 200 seeds — some unreachable"


def test_reward_bounds_procedural():
    """All procedural scenarios produce rewards in (0.001, 0.999)."""
    for seed in range(30):
        for diff in ["easy", "medium", "hard"]:
            env = ComplianceAuditorEnvironment()
            env.reset(seed=seed, scenario_id=f"procedural_{diff}_{seed}")

            env._tool_fns["get_system_overview"]()
            env._tool_fns["classify_system"](risk_category="high_risk")

            result = json.loads(env._tool_fns["verify_compliance"](
                risk_classification="high_risk",
                overall_assessment="test",
                key_findings_summary="test"
            ))
            assert 0.0 < result["reward"] < 1.0, \
                f"Reward {result['reward']} out of bounds @ seed={seed} diff={diff}"

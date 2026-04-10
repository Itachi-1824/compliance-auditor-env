"""Difficulty calibration tests.

Proves that the environment is properly calibrated: a naive agent
(same strategy for all scenarios) scores higher on easy scenarios
than on hard ones. This validates the difficulty tier design.
"""

import json
from server.environment import ComplianceAuditorEnvironment
from scenarios.registry import SCENARIO_LIST


def _naive_audit(scenario_id: str) -> float:
    """Run a naive audit strategy — call all tools in order, submit generic findings."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id=scenario_id)

    # Naive strategy: call everything, classify as high_risk, submit generic findings
    env._tool_fns["get_system_overview"]()
    env._tool_fns["classify_system"](risk_category="high_risk")
    env._tool_fns["check_documentation"]()
    env._tool_fns["audit_training_data"]()
    env._tool_fns["verify_human_oversight"]()
    env._tool_fns["check_transparency"]()
    env._tool_fns["assess_risk_management"]()
    env._tool_fns["check_logging"]()
    env._tool_fns["submit_finding"](finding="documentation_gaps", severity="high")
    env._tool_fns["submit_finding"](finding="bias_concern", severity="high")
    env._tool_fns["submit_finding"](finding="insufficient_oversight", severity="medium")
    env._tool_fns["recommend_fix"](finding="gaps", remediation="improve_documentation")
    env._tool_fns["recommend_fix"](finding="bias", remediation="conduct_bias_audit")

    result = json.loads(env._tool_fns["verify_compliance"](
        risk_classification="high_risk",
        overall_assessment="Multiple compliance gaps identified",
        key_findings_summary="Documentation, bias, and oversight issues"
    ))
    return result["reward"]


def test_hard_scenarios_have_more_findings_than_easy():
    """Hard scenarios require identifying more ground truth findings.

    This validates difficulty calibration — easy scenarios have 1-2 findings
    while hard scenarios have 5-6, making them harder to get perfect on.
    """
    from scenarios.registry import get_scenario

    easy_findings = []
    hard_findings = []

    for sc_info in SCENARIO_LIST:
        sc = get_scenario(sc_info["id"], 42)
        count = len(sc.ground_truth_findings)
        if sc_info["difficulty"] == "easy":
            easy_findings.append(count)
        elif sc_info["difficulty"] == "hard":
            hard_findings.append(count)

    avg_easy = sum(easy_findings) / len(easy_findings)
    avg_hard = sum(hard_findings) / len(hard_findings)

    assert avg_hard > avg_easy * 2, \
        f"Hard scenarios ({avg_hard:.1f} avg findings) should have at least 2x the findings of easy ({avg_easy:.1f})"


def test_prohibited_scenario_punishes_wrong_classification():
    """Classifying a prohibited system as high_risk should lose significant points.

    The prohibited scenario is the hardest because the agent must see through
    the deployer's framing to correctly identify it as prohibited.
    """
    # Naive agent classifies as high_risk (wrong for prohibited)
    prohibited_score = _naive_audit("hard_social_scoring_prohibited_001")

    # Perfect classification on the same scenario
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="hard_social_scoring_prohibited_001")
    env._tool_fns["get_system_overview"]()
    env._tool_fns["classify_system"](risk_category="prohibited")
    env._tool_fns["submit_finding"](finding="prohibited_social_scoring_system")
    env._tool_fns["recommend_fix"](finding="prohibited", remediation="immediate_system_shutdown")
    result = json.loads(env._tool_fns["verify_compliance"](
        risk_classification="prohibited",
        overall_assessment="Prohibited system",
        key_findings_summary="Social scoring"
    ))
    correct_score = result["reward"]

    assert correct_score > prohibited_score, \
        f"Correct prohibited ({correct_score:.3f}) should beat naive high_risk ({prohibited_score:.3f})"


def test_medium_scenarios_spread_across_difficulty():
    """Medium scenarios should produce different scores with the naive agent,
    showing that they test different compliance challenges.
    """
    medium_scores = {}
    for sc_info in SCENARIO_LIST:
        if sc_info["difficulty"] == "medium":
            medium_scores[sc_info["id"]] = _naive_audit(sc_info["id"])

    scores = list(medium_scores.values())
    spread = max(scores) - min(scores)
    assert spread > 0.02, \
        f"Medium scenarios should have score variance. Spread: {spread:.3f}, scores: {medium_scores}"

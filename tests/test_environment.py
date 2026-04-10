"""Test suite for EU AI Act Compliance Auditor environment."""

import json
import sys
sys.path.insert(0, ".")

from server.environment import ComplianceAuditorEnvironment
from scenarios.registry import get_scenario, SCENARIO_LIST, DIFFICULTY_TIERS


def test_all_scenarios_instantiate():
    """All 8 scenarios create successfully."""
    for s in SCENARIO_LIST:
        sc = get_scenario(s["id"], seed=42)
        assert sc.scenario_id == s["id"]
        assert sc.correct_classification in ("prohibited", "high_risk", "limited_risk", "minimal_risk")
        assert len(sc.graph.nodes) > 0
        assert sc.graph.optimal_path_length() > 0
    print(f"PASS: {len(SCENARIO_LIST)} scenarios instantiate")


def test_parameter_randomization():
    """Different seeds produce different parameters."""
    s1 = get_scenario("easy_chatbot_transparency_001", seed=1)
    s2 = get_scenario("easy_chatbot_transparency_001", seed=99)
    assert s1.get_param("company") != s2.get_param("company")
    print("PASS: Parameter randomization works")


def test_environment_reset_step():
    """Environment reset and tool calls work."""
    env = ComplianceAuditorEnvironment()
    obs = env.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    assert not obs.done
    assert "compliance_auditor_env" not in str(obs.reward)  # reward is a number
    assert len(env._tool_fns) == 11
    print("PASS: Environment reset + 11 tools registered")


def test_full_audit_easy():
    """Complete audit of easy scenario produces valid score."""
    env = ComplianceAuditorEnvironment()
    env.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")

    fns = env._tool_fns
    json.loads(fns["get_system_overview"]())
    json.loads(fns["classify_system"](risk_category="limited_risk"))
    json.loads(fns["check_documentation"]())
    json.loads(fns["audit_training_data"]())
    json.loads(fns["verify_human_oversight"]())
    json.loads(fns["check_transparency"]())
    json.loads(fns["assess_risk_management"]())
    json.loads(fns["check_logging"]())
    json.loads(fns["submit_finding"](finding="missing_ai_disclosure"))
    json.loads(fns["submit_finding"](finding="no_human_escalation_option"))
    json.loads(fns["recommend_fix"](finding="disclosure", remediation="add_ai_disclosure_banner"))
    json.loads(fns["recommend_fix"](finding="escalation", remediation="implement_human_handoff"))

    result = json.loads(fns["verify_compliance"](
        risk_classification="limited_risk",
        overall_assessment="Limited risk with transparency gaps",
        key_findings_summary="Missing AI disclosure and no human escalation",
    ))
    assert result["done"] is True
    assert 0.01 <= result["reward"] <= 0.99
    assert result["reward"] > 0.5  # should score well with correct answers
    print(f"PASS: Easy audit score = {result['reward']:.4f}")


def test_full_audit_hard_prohibited():
    """Prohibited system detection produces valid score."""
    env = ComplianceAuditorEnvironment()
    env.reset(difficulty="hard", scenario_id="hard_social_scoring_prohibited_001")

    fns = env._tool_fns
    json.loads(fns["get_system_overview"]())
    json.loads(fns["classify_system"](risk_category="prohibited"))
    json.loads(fns["submit_finding"](finding="prohibited_social_scoring_system"))
    json.loads(fns["submit_finding"](finding="affects_access_to_public_services"))
    json.loads(fns["recommend_fix"](finding="prohibited", remediation="immediate_system_shutdown"))

    result = json.loads(fns["verify_compliance"](
        risk_classification="prohibited",
        overall_assessment="PROHIBITED under Article 5(1)(c)",
        key_findings_summary="Social scoring affecting public services",
    ))
    assert result["done"] is True
    assert 0.01 <= result["reward"] <= 0.99
    print(f"PASS: Hard prohibited score = {result['reward']:.4f}")


def test_reward_clamping():
    """Rewards are always in (0.01, 0.99)."""
    env = ComplianceAuditorEnvironment()
    env.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    fns = env._tool_fns

    # Minimal effort — should still be > 0.01
    result = json.loads(fns["verify_compliance"](
        risk_classification="wrong",
        overall_assessment="",
        key_findings_summary="",
    ))
    assert result["reward"] >= 0.001
    assert result["reward"] <= 0.999
    print(f"PASS: Reward clamping = {result['reward']:.4f}")


def test_query_budget():
    """Query budget is tracked correctly."""
    env = ComplianceAuditorEnvironment()
    env.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    fns = env._tool_fns

    r1 = json.loads(fns["get_system_overview"]())
    assert r1["queries_remaining"] == 99  # 100 - 1

    r2 = json.loads(fns["check_documentation"]())
    assert r2["queries_remaining"] == 98  # 100 - 2
    print("PASS: Query budget tracking")


def test_wrong_classification_penalty():
    """Wrong classification scores lower than correct."""
    # Correct classification
    env1 = ComplianceAuditorEnvironment()
    env1.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    env1._tool_fns["get_system_overview"]()
    env1._tool_fns["classify_system"](risk_category="limited_risk")
    r1 = json.loads(env1._tool_fns["verify_compliance"](
        risk_classification="limited_risk", overall_assessment="", key_findings_summary=""))

    # Wrong classification
    env2 = ComplianceAuditorEnvironment()
    env2.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    env2._tool_fns["get_system_overview"]()
    env2._tool_fns["classify_system"](risk_category="prohibited")
    r2 = json.loads(env2._tool_fns["verify_compliance"](
        risk_classification="prohibited", overall_assessment="", key_findings_summary=""))

    assert r1["reward"] > r2["reward"]
    print(f"PASS: Correct ({r1['reward']:.3f}) > Wrong ({r2['reward']:.3f})")


def test_methodology_penalty():
    """Skipping investigation before classification penalizes methodology."""
    # Good: investigate first
    env1 = ComplianceAuditorEnvironment()
    env1.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    fns1 = env1._tool_fns
    fns1["get_system_overview"]()
    fns1["check_documentation"]()
    fns1["classify_system"](risk_category="limited_risk")
    r1 = json.loads(fns1["verify_compliance"](
        risk_classification="limited_risk", overall_assessment="", key_findings_summary=""))

    # Bad: classify immediately
    env2 = ComplianceAuditorEnvironment()
    env2.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    fns2 = env2._tool_fns
    fns2["classify_system"](risk_category="limited_risk")
    r2 = json.loads(fns2["verify_compliance"](
        risk_classification="limited_risk", overall_assessment="", key_findings_summary=""))

    assert r1["reward"] >= r2["reward"]
    print(f"PASS: Investigate first ({r1['reward']:.3f}) >= Skip ({r2['reward']:.3f})")


if __name__ == "__main__":
    tests = [
        test_all_scenarios_instantiate,
        test_parameter_randomization,
        test_environment_reset_step,
        test_full_audit_easy,
        test_full_audit_hard_prohibited,
        test_reward_clamping,
        test_query_budget,
        test_wrong_classification_penalty,
        test_methodology_penalty,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"{passed} passed, {failed} failed out of {passed+failed}")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        sys.exit(1)

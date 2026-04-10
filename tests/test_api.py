"""Test HTTP API endpoints and concurrent sessions."""

import json
import sys
import threading
import time
sys.path.insert(0, ".")


def test_http_api():
    """Test /api/reset, /api/call_tool, /api/close via HTTP."""
    import uvicorn
    import requests
    from server.app import app

    def run():
        uvicorn.run(app, host="127.0.0.1", port=17950, log_level="error")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(4)

    B = "http://127.0.0.1:17950"

    # Health
    r = requests.get(f"{B}/health")
    assert r.status_code == 200, f"Health: {r.status_code}"

    # Tasks
    r = requests.get(f"{B}/tasks")
    tasks = r.json()["tasks"]
    assert len(tasks) >= 3, f"Tasks: {len(tasks)}"

    # Grader
    r = requests.post(f"{B}/grader", json={"task_id": "easy_chatbot_transparency_001", "classification": "limited_risk", "findings": ["missing_ai_disclosure"]})
    assert r.status_code == 200
    assert 0.001 <= r.json()["score"] <= 0.999

    # API Reset
    r = requests.post(f"{B}/api/reset", json={"difficulty": "easy"})
    assert r.status_code == 200
    data = r.json()
    sid = data["session_id"]
    assert len(data["tools"]) == 11

    # API Call Tool
    r = requests.post(f"{B}/api/call_tool", json={"session_id": sid, "tool_name": "get_system_overview", "arguments": {}})
    assert r.status_code == 200
    assert r.json()["done"] is False

    # API Call verify_compliance
    r = requests.post(f"{B}/api/call_tool", json={"session_id": sid, "tool_name": "verify_compliance", "arguments": {
        "risk_classification": "limited_risk",
        "overall_assessment": "test",
        "key_findings_summary": "test"
    }})
    assert r.status_code == 200
    assert r.json()["done"] is True
    assert r.json()["reward"] > 0.001

    # API Close
    r = requests.post(f"{B}/api/close", json={"session_id": sid})
    assert r.status_code == 200

    print("PASS: HTTP API endpoints")


def test_concurrent_sessions():
    """Test that multiple sessions can run independently."""
    from server.environment import ComplianceAuditorEnvironment

    env1 = ComplianceAuditorEnvironment()
    env2 = ComplianceAuditorEnvironment()

    env1.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    env2.reset(difficulty="hard", scenario_id="hard_social_scoring_prohibited_001")

    # Both should have independent state
    assert env1._scenario.scenario_id != env2._scenario.scenario_id
    assert env1._scenario.correct_classification != env2._scenario.correct_classification

    # Tool calls on env1 don't affect env2
    env1._tool_fns["get_system_overview"]()
    assert env1._queries_used == 1
    assert env2._queries_used == 0

    # Both can complete independently
    r1 = json.loads(env1._tool_fns["verify_compliance"](
        risk_classification="limited_risk", overall_assessment="test", key_findings_summary="test"))
    r2 = json.loads(env2._tool_fns["verify_compliance"](
        risk_classification="prohibited", overall_assessment="test", key_findings_summary="test"))

    assert r1["done"] is True
    assert r2["done"] is True
    assert r1["reward"] != r2["reward"]  # Different scenarios = different scores

    print("PASS: Concurrent sessions")


def test_grader_determinism():
    """Same inputs must always produce same score."""
    from server.environment import ComplianceAuditorEnvironment

    scores = []
    for _ in range(3):
        env = ComplianceAuditorEnvironment()
        env.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001", seed=42)
        env._tool_fns["get_system_overview"]()
        env._tool_fns["classify_system"](risk_category="limited_risk")
        r = json.loads(env._tool_fns["verify_compliance"](
            risk_classification="limited_risk", overall_assessment="test", key_findings_summary="test"))
        scores.append(r["reward"])

    assert scores[0] == scores[1] == scores[2], f"Non-deterministic: {scores}"
    print(f"PASS: Grader deterministic (score={scores[0]:.4f} x3)")


def test_invalid_tool_handling():
    """Invalid tool calls should return error, not crash."""
    from server.environment import ComplianceAuditorEnvironment

    env = ComplianceAuditorEnvironment()
    env.reset(difficulty="easy")

    # Valid tool with wrong args shouldn't crash
    try:
        result = env._tool_fns["classify_system"](risk_category="invalid_category")
        # Should still work, just lower score
        print("PASS: Invalid args handled gracefully")
    except Exception as e:
        print(f"FAIL: Invalid args crashed: {e}")
        sys.exit(1)


def test_score_range():
    """Scores must always be in (0.001, 0.999)."""
    from server.environment import ComplianceAuditorEnvironment

    # Best possible score
    env = ComplianceAuditorEnvironment()
    env.reset(difficulty="easy", scenario_id="easy_chatbot_transparency_001")
    fns = env._tool_fns
    fns["get_system_overview"]()
    fns["classify_system"](risk_category="limited_risk")
    fns["check_documentation"]()
    fns["audit_training_data"]()
    fns["verify_human_oversight"]()
    fns["check_transparency"]()
    fns["assess_risk_management"]()
    fns["check_logging"]()
    fns["submit_finding"](finding="missing_ai_disclosure")
    fns["submit_finding"](finding="no_human_escalation_option")
    fns["recommend_fix"](finding="disclosure", remediation="add_ai_disclosure_banner")
    fns["recommend_fix"](finding="escalation", remediation="implement_human_handoff")
    r = json.loads(fns["verify_compliance"](
        risk_classification="limited_risk",
        overall_assessment="Complete audit of limited risk chatbot",
        key_findings_summary="Missing AI disclosure and no human escalation"))
    assert 0.001 < r["reward"] < 0.999, f"Best score out of range: {r['reward']}"

    # Worst possible score
    env2 = ComplianceAuditorEnvironment()
    env2.reset(difficulty="hard", scenario_id="hard_social_scoring_prohibited_001")
    r2 = json.loads(env2._tool_fns["verify_compliance"](
        risk_classification="minimal_risk",
        overall_assessment="", key_findings_summary=""))
    assert 0.001 <= r2["reward"] <= 0.999, f"Worst score out of range: {r2['reward']}"

    print(f"PASS: Score range OK (best={r['reward']:.3f}, worst={r2['reward']:.3f})")


if __name__ == "__main__":
    tests = [
        test_concurrent_sessions,
        test_grader_determinism,
        test_invalid_tool_handling,
        test_score_range,
        test_http_api,  # Last — starts server
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
    print(f"{passed} passed, {failed} failed")
    if failed == 0:
        print("ALL API TESTS PASSED")
    else:
        sys.exit(1)

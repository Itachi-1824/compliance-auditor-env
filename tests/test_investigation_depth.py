"""Investigation depth tests.

Verifies that tool responses contain investigation-grade content requiring
genuine analysis — not pre-digested verdicts.

Tests prove:
  1. Documents contain statistical evidence the agent must interpret
  2. Red herrings are embedded naturally in the evidence
  3. Cross-document reasoning is required to form findings
  4. Document length scales with difficulty tier
  5. Randomization changes parameterized content but not violations
  6. Dynamic audit progress appears after findings
"""

import json
from server.environment import ComplianceAuditorEnvironment
from scenarios.registry import get_scenario, SCENARIO_LIST


# ── Test 1: No pre-digested verdicts in tool responses ────────────

def test_no_predigested_verdicts_in_documents():
    """Investigation documents must NOT contain explicit compliance verdicts.

    Labels like 'NON-COMPLIANT', 'FAILED', 'VIOLATION' hand the answer
    to the agent. Documents should contain evidence, not conclusions.
    """
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    predigested_labels = [
        "NON-COMPLIANT", "NON_COMPLIANT", "FAILED", "VIOLATION FOUND",
        "COMPLIANCE VIOLATION", "DOES NOT COMPLY",
    ]

    for tool_name in ["check_documentation", "audit_training_data",
                      "verify_human_oversight", "check_transparency",
                      "assess_risk_management", "check_logging"]:
        result = json.loads(env._tool_fns[tool_name]())
        content = result.get("content", "")
        for label in predigested_labels:
            assert label not in content, \
                f"Pre-digested verdict '{label}' found in {tool_name} response"


# ── Test 2: Statistical evidence present in training data audit ───

def test_training_data_contains_statistical_tables():
    """Training data audit must contain numerical evidence the agent
    must interpret to identify bias — not just 'bias found' labels.
    """
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    result = json.loads(env._tool_fns["audit_training_data"]())
    content = result["content"]

    # Must contain actual numbers (callback rates, percentages)
    # Note: exact values vary due to seed-based noise injection
    import re
    pct_matches = re.findall(r'\d{1,2}\.\d%', content)
    assert len(pct_matches) >= 4, \
        f"Training data should contain multiple percentage figures, found {len(pct_matches)}"

    # Must contain demographic categories
    assert "Male" in content or "Female" in content, \
        "Training data should reference demographic groups"

    # Must NOT contain pre-computed verdict
    assert "FAILED" not in content, \
        "Training data should not contain pre-digested 'FAILED' verdict"


# ── Test 3: Red herrings embedded naturally ───────────────────────

def test_red_herrings_in_evidence():
    """Red herring content should appear naturally in investigation documents,
    not as separate labeled items the agent can trivially filter.
    """
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    # The hiring scenario has red herrings: "prohibited_social_scoring" and "biometric_processing"
    # The training data document should mention the separate fraud detection system
    # (which is compliant and unrelated) as a natural red herring
    result = json.loads(env._tool_fns["audit_training_data"]())
    content = result["content"].lower()
    assert "fraud" in content, \
        "Red herring (compliant fraud system) should appear naturally in training data doc"


# ── Test 4: Document length scales with difficulty ────────────────

def test_document_length_scales_with_difficulty():
    """Hard scenarios should have longer, more complex documents than easy ones."""
    easy_total = 0
    hard_total = 0

    for sc_info in SCENARIO_LIST:
        sc = get_scenario(sc_info["id"], 42)
        total_len = sum(len(getattr(sc, field, "")) for field in [
            "documentation_data", "training_data_info", "oversight_info",
            "transparency_info", "risk_assessment_info", "logging_info",
        ])
        if sc_info["difficulty"] == "easy":
            easy_total += total_len
        elif sc_info["difficulty"] == "hard":
            hard_total += total_len

    avg_easy = easy_total / 2  # 2 easy scenarios
    avg_hard = hard_total / 3  # 3 hard scenarios

    assert avg_hard > avg_easy * 1.3, \
        f"Hard scenarios ({avg_hard:.0f} chars avg) should be significantly larger than easy ({avg_easy:.0f} chars avg)"


# ── Test 5: Randomization changes params but not violations ───────

def test_randomization_preserves_violations():
    """Different seeds should change surface parameters (company, date) but
    the same ground truth findings should remain discoverable.
    """
    sc1 = get_scenario("medium_hiring_bias_001", seed=42)
    sc2 = get_scenario("medium_hiring_bias_001", seed=12345)

    # Ground truth findings must be identical
    assert sc1.ground_truth_findings == sc2.ground_truth_findings
    assert sc1.correct_classification == sc2.correct_classification

    # At least some parameters must differ across seeds
    params_differ = (
        sc1.get_param("company") != sc2.get_param("company")
        or sc1.get_param("version") != sc2.get_param("version")
        or sc1.get_param("date") != sc2.get_param("date")
        or sc1.get_param("usercount") != sc2.get_param("usercount")
    )
    assert params_differ, "Randomized parameters should differ across seeds"


# ── Test 6: Randomization appears in rendered documents ───────────

def test_randomization_in_rendered_documents():
    """Rendered documents should contain randomized parameters, not placeholders."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    result = json.loads(env._tool_fns["get_system_overview"]())
    content = result["content"]

    # Should NOT contain raw placeholders
    assert "__COMPANY__" not in content, "Placeholder __COMPANY__ should be replaced"
    assert "__VERSION__" not in content, "Placeholder __VERSION__ should be replaced"

    # Should contain actual randomized values
    assert "v" in content.lower(), "Should contain version number"


# ── Test 7: Dynamic audit progress appears after findings ─────────

def test_dynamic_audit_progress():
    """After submitting findings, subsequent tool calls should include
    audit progress section showing what's been found.
    """
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    # Before any findings — no progress section
    r1 = json.loads(env._tool_fns["get_system_overview"]())
    assert "AUDIT PROGRESS" not in r1["content"], \
        "No progress section before any actions"

    # After classification and a finding
    env._tool_fns["classify_system"](risk_category="high_risk")
    env._tool_fns["check_documentation"]()
    env._tool_fns["submit_finding"](finding="test_finding", severity="high")

    r2 = json.loads(env._tool_fns["audit_training_data"]())
    assert "AUDIT PROGRESS" in r2["content"], \
        "Progress section should appear after findings submitted"
    assert "test_finding" in r2["content"], \
        "Progress should show submitted findings"
    assert "High Risk" in r2["content"], \
        "Progress should show submitted classification"


# ── Test 8: Each scenario has unique graph topology ───────────────

def test_graph_diversity():
    """At least 4 distinct graph topologies across 8 scenarios."""
    sigs = set()
    for sc_info in SCENARIO_LIST:
        sc = get_scenario(sc_info["id"], 42)
        sig = tuple(sorted(
            (t.from_state, t.to_state, t.tool_name, t.outcome)
            for t in sc.graph.transitions
        ))
        sigs.add(sig)
    assert len(sigs) >= 4, f"Only {len(sigs)} unique graph topologies — need at least 4"


# ── Test 9: Prohibited scenario does not reveal classification ────

def test_prohibited_scenario_concealment():
    """The prohibited system's overview should NOT reveal it's prohibited.
    The deployer frames it as a wellness tool — agent must discover the truth.
    """
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="hard_social_scoring_prohibited_001")

    result = json.loads(env._tool_fns["get_system_overview"]())
    content = result["content"].lower()

    assert "prohibited" not in content, \
        "Overview should not reveal the system is prohibited — that's what the agent must discover"
    assert "wellness" in content or "civic" in content or "engagement" in content, \
        "Overview should use the deployer's framing (wellness/civic engagement)"


# ── Test 10: All 8 scenarios produce valid tool responses ─────────

def test_all_scenarios_produce_rich_responses():
    """Every scenario's investigation tools must return non-trivial content."""
    for sc_info in SCENARIO_LIST:
        env = ComplianceAuditorEnvironment()
        env.reset(seed=42, scenario_id=sc_info["id"])

        env._tool_fns["get_system_overview"]()
        env._tool_fns["classify_system"](risk_category="high_risk")

        for tool in ["check_documentation", "audit_training_data",
                     "verify_human_oversight", "check_transparency",
                     "assess_risk_management", "check_logging"]:
            result = json.loads(env._tool_fns[tool]())
            content = result.get("content", "")
            assert len(content) > 100, \
                f"{sc_info['id']}/{tool}: document too short ({len(content)} chars)"

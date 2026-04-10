"""Evidence chain validation tests.

Proves that the environment validates whether findings are supported by
actual investigation. This is a unique feature — most environments just
accept whatever findings are submitted without checking if the agent
actually read the relevant documents.

Tests:
  1. Finding without investigation → warning
  2. Finding after investigation → no warning
  3. Multiple keyword matching works correctly
  4. Verify_compliance shows missed findings
  5. Verify_compliance shows classification accuracy
"""

import json
from server.environment import ComplianceAuditorEnvironment


def test_finding_without_investigation_warns():
    """Submitting a bias finding without auditing training data triggers a warning."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    result = json.loads(env._tool_fns["submit_finding"](
        finding="gender_bias_in_training_data", severity="critical"
    ))
    assert "evidence_warnings" in result, "Should warn about missing investigation"
    assert any("training_data" in w for w in result["evidence_warnings"]), \
        "Warning should mention training_data area"


def test_finding_after_investigation_no_warning():
    """Submitting a bias finding after auditing training data has no warning."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    env._tool_fns["get_system_overview"]()
    env._tool_fns["audit_training_data"]()

    result = json.loads(env._tool_fns["submit_finding"](
        finding="gender_bias_in_training_data", severity="critical"
    ))
    assert "evidence_warnings" not in result, \
        "No warning when area was investigated"


def test_oversight_finding_warns_without_oversight_check():
    """Submitting an oversight finding without verify_human_oversight warns."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    result = json.loads(env._tool_fns["submit_finding"](
        finding="insufficient_human_oversight", severity="high"
    ))
    assert "evidence_warnings" in result
    assert any("oversight" in w for w in result["evidence_warnings"])


def test_transparency_finding_warns_without_transparency_check():
    """Submitting a transparency finding without check_transparency warns."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="easy_chatbot_transparency_001")

    result = json.loads(env._tool_fns["submit_finding"](
        finding="missing_ai_disclosure_transparency", severity="high"
    ))
    assert "evidence_warnings" in result
    assert any("transparency" in w for w in result["evidence_warnings"])


def test_generic_finding_no_keyword_match_no_warning():
    """A finding with no recognizable keywords produces no evidence warning."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    result = json.loads(env._tool_fns["submit_finding"](
        finding="general_compliance_concern", severity="medium"
    ))
    assert "evidence_warnings" not in result, \
        "Generic findings without keyword matches should not warn"


def test_verify_shows_missed_findings():
    """Verify_compliance response shows which ground truth findings were missed."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    env._tool_fns["get_system_overview"]()
    env._tool_fns["classify_system"](risk_category="high_risk")
    env._tool_fns["submit_finding"](finding="gender_bias_in_technical_screening")

    result = json.loads(env._tool_fns["verify_compliance"](
        risk_classification="high_risk",
        overall_assessment="Bias found",
        key_findings_summary="Gender bias"
    ))

    summary = result["audit_summary"]
    assert summary["findings"]["matched"] == 1
    assert summary["findings"]["ground_truth_total"] == 5
    assert len(summary["findings"]["missed"]) == 4
    assert "insufficient_human_oversight" in summary["findings"]["missed"]


def test_verify_shows_classification_accuracy():
    """Verify_compliance response shows classification match status."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="hard_social_scoring_prohibited_001")

    env._tool_fns["get_system_overview"]()

    # Wrong classification
    result = json.loads(env._tool_fns["verify_compliance"](
        risk_classification="high_risk",
        overall_assessment="High risk system",
        key_findings_summary="Various issues"
    ))
    assert result["audit_summary"]["classification"]["correct"] == "prohibited"
    assert result["audit_summary"]["classification"]["match"] == "partial"

    # Correct classification in a new episode
    env2 = ComplianceAuditorEnvironment()
    env2.reset(seed=42, scenario_id="hard_social_scoring_prohibited_001")
    env2._tool_fns["get_system_overview"]()
    result2 = json.loads(env2._tool_fns["verify_compliance"](
        risk_classification="prohibited",
        overall_assessment="Prohibited system",
        key_findings_summary="Social scoring"
    ))
    assert result2["audit_summary"]["classification"]["match"] == "exact"


def test_verify_shows_areas_investigated():
    """Verify response shows which investigation areas were actually explored."""
    env = ComplianceAuditorEnvironment()
    env.reset(seed=42, scenario_id="medium_hiring_bias_001")

    env._tool_fns["get_system_overview"]()
    env._tool_fns["classify_system"](risk_category="high_risk")
    env._tool_fns["check_documentation"]()
    env._tool_fns["audit_training_data"]()

    result = json.loads(env._tool_fns["verify_compliance"](
        risk_classification="high_risk",
        overall_assessment="Partial audit",
        key_findings_summary="Documentation and data issues"
    ))

    areas = result["audit_summary"]["areas_investigated"]
    assert "overview" in areas
    assert "documentation" in areas
    assert "training_data" in areas
    # These were NOT investigated
    assert "oversight" not in areas
    assert "transparency" not in areas

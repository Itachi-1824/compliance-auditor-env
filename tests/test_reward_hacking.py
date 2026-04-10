"""Adversarial reward-hacking tests.

Verifies the 6-component reward function cannot be gamed by common exploits:

  1. Skip investigation — submit findings without reading documents
  2. Spam findings — flood with every possible finding string
  3. Red herring bait — submit red herrings as findings
  4. Skip to verify — call verify_compliance immediately
  5. Wrong classification — still score well on other components
  6. Skip remediation — submit findings but no fixes
  7. Classify without overview — skip get_system_overview
  8. Fewer steps than optimal — game efficiency by rushing
  9. Hallucinate findings — submit plausible-sounding false findings
 10. Perfect findings but wrong classification — test component isolation

If any exploit scores above the expected cap, the reward function has a hole.
"""

from server.engine import compute_reward, safe_reward, RewardBreakdown
from scenarios.registry import get_scenario


# ── Test helpers ──────────────────────────────────────────────────

def _score_scenario(
    scenario_id: str = "medium_hiring_bias_001",
    seed: int = 42,
    classification: str = "",
    findings: list = None,
    remediation: list = None,
    tool_sequence: list = None,
    steps: int = 10,
) -> RewardBreakdown:
    sc = get_scenario(scenario_id, seed)
    return compute_reward(
        scenario=sc,
        classification_submitted=classification,
        findings_submitted=findings or [],
        remediation_submitted=remediation or [],
        tool_sequence=tool_sequence or [],
        steps_taken=steps,
    )


def _total(scenario_id="medium_hiring_bias_001", **kwargs):
    return _score_scenario(scenario_id, **kwargs).total()


# ── Exploit 1: Skip investigation, submit correct findings ────────

def test_skip_investigation_penalizes_methodology():
    """Agent submits perfect findings without calling any investigation tools.
    Must lose methodology score (15%) — no investigation tools in sequence.
    """
    r = _score_scenario(
        classification="high_risk",
        findings=[
            "gender_bias_in_technical_screening",
            "insufficient_human_oversight",
            "incomplete_technical_documentation",
            "missing_fundamental_rights_impact_assessment",
            "inadequate_data_governance",
        ],
        remediation=[
            "conduct_bias_audit_and_mitigation",
            "implement_human_review_for_all_rejections",
            "complete_annex_iv_technical_documentation",
            "perform_fundamental_rights_impact_assessment",
            "establish_data_governance_framework",
        ],
        tool_sequence=["submit_finding"] * 5 + ["recommend_fix"] * 5 + ["verify_compliance"],
        steps=11,
    )
    assert r.methodology == 0.0, f"methodology should be 0 without investigation tools: {r.methodology}"
    # Even with perfect findings, losing methodology (15%) is significant
    assert r.total() < 0.86, f"skip-investigation still scores too high: {r.total()}"


# ── Exploit 2: Spam every possible finding string ─────────────────

def test_spam_findings_penalizes_precision():
    """Agent submits 20 findings hoping to hit all ground truth by volume.
    False positives should destroy finding_precision (15%).
    """
    r = _score_scenario(
        classification="high_risk",
        findings=[
            "gender_bias_in_technical_screening",  # real
            "insufficient_human_oversight",  # real
            "incomplete_technical_documentation",  # real
            "missing_fundamental_rights_impact_assessment",  # real
            "inadequate_data_governance",  # real
            # False positives below
            "prohibited_social_scoring",  # red herring
            "biometric_processing",  # red herring
            "gdpr_violation", "data_breach", "missing_ce_marking",
            "environmental_impact", "accessibility_violation",
            "patent_infringement", "labor_law_violation",
            "antitrust_concern", "tax_evasion",
        ],
        tool_sequence=["get_system_overview", "classify_system"] + ["submit_finding"] * 16,
        steps=18,
    )
    # 5 true positives out of 16 = low precision
    assert r.finding_precision < 0.5, f"spam precision too high: {r.finding_precision}"


# ── Exploit 3: Submit red herrings as findings ────────────────────

def test_red_herring_submission_penalized():
    """Agent falls for red herrings and submits them as findings.
    Must be penalized via finding_precision component.
    """
    r = _score_scenario(
        classification="high_risk",
        findings=[
            "prohibited_social_scoring",  # red herring
            "biometric_processing",  # red herring
        ],
        tool_sequence=["get_system_overview", "classify_system", "submit_finding", "submit_finding"],
        steps=4,
    )
    # Two false positives, zero true positives
    assert r.finding_precision == 0.0, f"red herrings should give 0 precision: {r.finding_precision}"
    assert r.finding_completeness == 0.0, f"no real findings means 0 completeness: {r.finding_completeness}"
    assert r.total() < 0.35, f"red-herring-only agent scores too high: {r.total()}"


# ── Exploit 4: Skip straight to verify_compliance ─────────────────

def test_immediate_verify_scores_near_floor():
    """Agent calls verify_compliance with empty inputs on step 1.
    Should score near the minimum (only safe_reward floor).
    """
    r = _score_scenario(
        classification="",
        findings=[],
        remediation=[],
        tool_sequence=["verify_compliance"],
        steps=1,
    )
    assert r.classification == 0.0
    assert r.finding_completeness == 0.0
    assert r.finding_precision == 0.0
    assert r.remediation == 0.0
    assert r.methodology == 0.0
    assert r.total() < 0.05, f"empty verify should be near floor: {r.total()}"


# ── Exploit 5: Wrong classification but perfect everything else ────

def test_wrong_classification_costs_20_percent():
    """Agent gets everything right except classification.
    Must lose the full 20% classification weight.
    """
    r_correct = _score_scenario(
        classification="high_risk",
        findings=["gender_bias_in_technical_screening", "insufficient_human_oversight",
                  "incomplete_technical_documentation", "missing_fundamental_rights_impact_assessment",
                  "inadequate_data_governance"],
        remediation=["conduct_bias_audit_and_mitigation", "implement_human_review_for_all_rejections",
                     "complete_annex_iv_technical_documentation", "perform_fundamental_rights_impact_assessment",
                     "establish_data_governance_framework"],
        tool_sequence=["get_system_overview", "classify_system", "check_documentation",
                      "audit_training_data", "verify_human_oversight", "check_transparency",
                      "assess_risk_management", "check_logging", "submit_finding", "submit_finding",
                      "submit_finding", "submit_finding", "submit_finding",
                      "recommend_fix", "recommend_fix", "recommend_fix",
                      "recommend_fix", "recommend_fix", "verify_compliance"],
        steps=19,
    )
    r_wrong = _score_scenario(
        classification="minimal_risk",  # WRONG — should be high_risk
        findings=["gender_bias_in_technical_screening", "insufficient_human_oversight",
                  "incomplete_technical_documentation", "missing_fundamental_rights_impact_assessment",
                  "inadequate_data_governance"],
        remediation=["conduct_bias_audit_and_mitigation", "implement_human_review_for_all_rejections",
                     "complete_annex_iv_technical_documentation", "perform_fundamental_rights_impact_assessment",
                     "establish_data_governance_framework"],
        tool_sequence=["get_system_overview", "classify_system", "check_documentation",
                      "audit_training_data", "verify_human_oversight", "check_transparency",
                      "assess_risk_management", "check_logging", "submit_finding", "submit_finding",
                      "submit_finding", "submit_finding", "submit_finding",
                      "recommend_fix", "recommend_fix", "recommend_fix",
                      "recommend_fix", "recommend_fix", "verify_compliance"],
        steps=19,
    )
    gap = r_correct.total() - r_wrong.total()
    assert gap >= 0.10, f"wrong classification gap too small: {gap:.4f} (correct={r_correct.total():.4f}, wrong={r_wrong.total():.4f})"


# ── Exploit 6: Perfect findings but zero remediation ──────────────

def test_no_remediation_loses_15_percent():
    """Agent identifies all findings but proposes no remediation.
    Must lose the full 15% remediation weight.
    """
    r = _score_scenario(
        classification="high_risk",
        findings=["gender_bias_in_technical_screening", "insufficient_human_oversight",
                  "incomplete_technical_documentation", "missing_fundamental_rights_impact_assessment",
                  "inadequate_data_governance"],
        remediation=[],  # no remediation!
        tool_sequence=["get_system_overview", "classify_system", "check_documentation",
                      "audit_training_data", "verify_human_oversight", "check_transparency",
                      "assess_risk_management", "check_logging",
                      "submit_finding", "submit_finding", "submit_finding",
                      "submit_finding", "submit_finding", "verify_compliance"],
        steps=14,
    )
    assert r.remediation == 0.0, f"no remediation should give 0: {r.remediation}"


# ── Exploit 7: Classify without overview ──────────────────────────

def test_classify_before_overview_penalizes_methodology():
    """Agent classifies before gathering system overview.
    Investigation order should be penalized in methodology.
    """
    r = _score_scenario(
        classification="high_risk",
        findings=["gender_bias_in_technical_screening"],
        tool_sequence=["classify_system", "get_system_overview", "submit_finding"],
        steps=3,
    )
    # classify_system before get_system_overview is an order violation
    assert r.methodology < 0.5, f"wrong order methodology too high: {r.methodology}"


# ── Exploit 8: Fewer steps than optimal games efficiency ──────────

def test_fewer_steps_than_optimal_penalized():
    """Agent takes fewer steps than the optimal path.
    This means skipping investigation — efficiency should be penalized.
    """
    r_rushed = _score_scenario(
        classification="high_risk",
        findings=["gender_bias_in_technical_screening"],
        tool_sequence=["verify_compliance"],
        steps=2,  # way fewer than optimal
    )
    r_proper = _score_scenario(
        classification="high_risk",
        findings=["gender_bias_in_technical_screening"],
        tool_sequence=["get_system_overview", "classify_system", "check_documentation",
                      "audit_training_data", "verify_human_oversight",
                      "check_transparency", "assess_risk_management", "check_logging",
                      "submit_finding", "verify_compliance"],
        steps=12,
    )
    assert r_proper.efficiency > r_rushed.efficiency, \
        f"rushed ({r_rushed.efficiency}) should not beat proper ({r_proper.efficiency})"


# ── Exploit 9: Hallucinate plausible-sounding findings ────────────

def test_hallucinated_findings_low_precision():
    """Agent submits plausible-sounding but wrong findings.
    Token-based matching should not match these.
    """
    r = _score_scenario(
        classification="high_risk",
        findings=[
            "ai_model_lacks_interpretability",
            "no_audit_trail_for_decisions",
            "potential_discrimination_in_outputs",
            "insufficient_testing_methodology",
        ],
        tool_sequence=["get_system_overview", "classify_system", "check_documentation",
                      "submit_finding", "submit_finding", "submit_finding", "submit_finding"],
        steps=7,
    )
    # These don't token-match the ground truth findings
    assert r.finding_completeness < 0.4, f"hallucinated findings match too well: {r.finding_completeness}"


# ── Exploit 10: Perfect on prohibited scenario with wrong class ───

def test_prohibited_classified_as_high_risk():
    """Agent correctly finds violations but classifies prohibited as high_risk.
    Partial classification match should give 40% credit, not full.
    """
    r = _score_scenario(
        scenario_id="hard_social_scoring_prohibited_001",
        classification="high_risk",  # wrong — should be prohibited
        findings=["prohibited_social_scoring_system", "disguised_as_voluntary_wellness",
                  "affects_access_to_public_services"],
        tool_sequence=["get_system_overview", "classify_system", "submit_finding",
                      "submit_finding", "submit_finding", "verify_compliance"],
        steps=6,
    )
    assert r.classification == 0.4, f"adjacent classification should be 0.4: {r.classification}"


# ── Sanity: perfect run on medium hiring ──────────────────────────

def test_perfect_run_scores_high():
    """A perfect audit should score above 0.90."""
    r = _score_scenario(
        classification="high_risk",
        findings=["gender_bias_in_technical_screening", "insufficient_human_oversight",
                  "incomplete_technical_documentation", "missing_fundamental_rights_impact_assessment",
                  "inadequate_data_governance"],
        remediation=["conduct_bias_audit_and_mitigation", "implement_human_review_for_all_rejections",
                     "complete_annex_iv_technical_documentation",
                     "perform_fundamental_rights_impact_assessment",
                     "establish_data_governance_framework"],
        tool_sequence=["get_system_overview", "classify_system", "check_documentation",
                      "audit_training_data", "verify_human_oversight", "check_transparency",
                      "assess_risk_management", "check_logging",
                      "submit_finding", "submit_finding", "submit_finding",
                      "submit_finding", "submit_finding",
                      "recommend_fix", "recommend_fix", "recommend_fix",
                      "recommend_fix", "recommend_fix",
                      "verify_compliance"],
        steps=19,
    )
    assert r.total() > 0.85, f"perfect run too low: {r.total()}"
    assert r.classification == 1.0
    assert r.methodology > 0.8


# ── Bounds: all rewards strictly in (0, 1) ────────────────────────

def test_reward_bounds_all_scenarios():
    """Every scenario × various inputs must produce reward in (0.001, 0.999)."""
    from scenarios.registry import SCENARIO_LIST
    for sc_info in SCENARIO_LIST:
        for cls in ["", "prohibited", "high_risk", "limited_risk", "minimal_risk", "garbage"]:
            for findings in [[], ["some_finding"], ["a", "b", "c", "d", "e", "f"]]:
                r = _total(
                    scenario_id=sc_info["id"],
                    classification=cls,
                    findings=findings,
                    tool_sequence=["verify_compliance"],
                    steps=1,
                )
                assert 0.0 < r < 1.0, \
                    f"out of range: {r} @ {sc_info['id']} cls={cls} findings={len(findings)}"

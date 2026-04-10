"""
Procedural scenario generator — infinite unique compliance audit scenarios.

Combines system type templates, violation templates, and red herring templates
using seed-based randomization to produce coherent, graded scenarios that are
unique for every seed. Impossible to memorize.

Architecture:
  1. SystemTemplate — defines a category of AI system (drone delivery, exam proctoring, etc.)
  2. ViolationTemplate — a specific compliance violation with document injection text
  3. RedHerringTemplate — misleading information that isn't a real violation
  4. ProceduralGenerator.generate(seed, difficulty) → AuditScenario
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from server.engine import AuditScenario, StateGraph, StateNode, Transition


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemTemplate:
    id: str
    name_template: str  # e.g. "{company} DroneGuard"
    category: str  # prohibited, high_risk, limited_risk, minimal_risk
    annex_ref: str  # which Annex III category or article
    description_template: str
    deployer_template: str
    domain_keywords: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ViolationTemplate:
    id: str
    tool_area: str  # documentation, training_data, oversight, transparency, risk_management, logging
    finding_id: str  # ground truth finding string
    remediation_id: str  # required remediation string
    doc_injection: str  # text injected into the relevant document section
    severity: str = "high"


@dataclass(frozen=True)
class RedHerringTemplate:
    id: str
    tool_area: str  # which document section contains it
    doc_injection: str  # misleading text


# ---------------------------------------------------------------------------
# System type pool (5 types covering different AI Act categories)
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATES: List[SystemTemplate] = [
    SystemTemplate(
        id="drone_delivery",
        name_template="{company} SkyRoute Delivery AI",
        category="high_risk",
        annex_ref="Annex III Category 2 — Critical Infrastructure",
        description_template=(
            "Autonomous drone delivery system operating in urban areas across {region}. "
            "AI controls flight path planning, obstacle avoidance, and delivery routing "
            "for {user_count} packages per month. System makes real-time autonomous "
            "decisions affecting public safety in shared airspace."
        ),
        deployer_template="{company} — logistics-tech startup, drone operator license in {region}.",
        domain_keywords=("drone", "airspace", "safety", "autonomous", "delivery"),
    ),
    SystemTemplate(
        id="exam_proctoring",
        name_template="{company} ExamGuard AI",
        category="high_risk",
        annex_ref="Annex III Category 3 — Education and Vocational Training",
        description_template=(
            "AI-powered online exam proctoring system used by {user_count} students "
            "across {region}. Monitors webcam feeds, screen activity, and audio to "
            "detect cheating behavior. Automated flagging can result in exam "
            "invalidation and academic disciplinary proceedings."
        ),
        deployer_template="{company} — EdTech company, serving 200+ universities in {region}.",
        domain_keywords=("exam", "proctoring", "student", "cheating", "academic"),
    ),
    SystemTemplate(
        id="insurance_claims",
        name_template="{company} ClaimIQ Adjudicator",
        category="high_risk",
        annex_ref="Annex III Category 5(a) — Access to Essential Services (Insurance)",
        description_template=(
            "AI system that evaluates and adjudicates insurance claims for health, "
            "property, and vehicle policies. Processes {user_count} claims annually "
            "in {region}. Automated decisions include claim approval, denial, and "
            "payout amount determination up to EUR 100,000."
        ),
        deployer_template="{company} — InsurTech firm, licensed in {region}, {user_count} policyholders.",
        domain_keywords=("insurance", "claim", "adjudication", "payout", "policy"),
    ),
    SystemTemplate(
        id="legal_research",
        name_template="{company} LexAssist AI",
        category="limited_risk",
        annex_ref="Article 50 — Transparency obligations for AI interacting with persons",
        description_template=(
            "AI-powered legal research assistant used by law firms across {region}. "
            "Analyzes case law, statutes, and regulatory documents to provide "
            "research summaries and case strategy suggestions. Used by {user_count} "
            "attorneys. Does not make legal decisions — advisory role only."
        ),
        deployer_template="{company} — LegalTech startup, SaaS platform for law firms in {region}.",
        domain_keywords=("legal", "research", "case law", "attorney", "advisory"),
    ),
    SystemTemplate(
        id="predictive_policing",
        name_template="{company} SafeCity Predictor",
        category="prohibited",
        annex_ref="Article 5(1)(d) — Prohibited: individual predictive policing",
        description_template=(
            "AI system deployed by municipal police in {region} to predict individual "
            "criminal behavior based on personal characteristics, social network "
            "analysis, and neighborhood data. Generates 'risk scores' for {user_count} "
            "residents used to allocate patrol resources and inform stop-and-search decisions."
        ),
        deployer_template="{company} — public safety technology vendor, contracted by 5 municipalities in {region}.",
        domain_keywords=("policing", "prediction", "crime", "risk score", "patrol"),
    ),
]

# ---------------------------------------------------------------------------
# Violation pool (20 violations across all tool areas)
# ---------------------------------------------------------------------------

VIOLATION_TEMPLATES: List[ViolationTemplate] = [
    # --- Documentation violations ---
    ViolationTemplate(
        id="missing_annex_iv_sections",
        tool_area="documentation",
        finding_id="incomplete_annex_iv_documentation",
        remediation_id="complete_missing_annex_iv_sections",
        doc_injection=(
            "ANNEX IV COMPLIANCE:\n"
            "  Section 2(b) Design specifications:    Not documented\n"
            "  Section 2(d) Performance per group:    Not documented\n"
            "  Section 2(g) Quality management:       Not documented\n"
            "  3 of 10 required sections are missing."
        ),
    ),
    ViolationTemplate(
        id="stale_documentation",
        tool_area="documentation",
        finding_id="outdated_technical_documentation",
        remediation_id="update_documentation_to_current_version",
        doc_injection=(
            "NOTE: Core technical documentation was last updated 22 months ago\n"
            "(prior to EU AI Act enforcement). It does not reference the AI Act,\n"
            "harmonised standards, or current deployment configuration."
        ),
    ),
    ViolationTemplate(
        id="no_fria",
        tool_area="documentation",
        finding_id="missing_fundamental_rights_impact_assessment",
        remediation_id="conduct_fundamental_rights_impact_assessment",
        doc_injection="Fundamental Rights Impact Assessment:  ABSENT — not conducted",
    ),
    # --- Training data violations ---
    ViolationTemplate(
        id="gender_bias",
        tool_area="training_data",
        finding_id="gender_bias_in_automated_decisions",
        remediation_id="conduct_bias_audit_and_mitigation",
        doc_injection=(
            "OUTCOME RATES BY GENDER:\n"
            "  Group              Rate     Delta\n"
            "  Male               41.3%    (baseline)\n"
            "  Female             29.7%    -28.1%\n"
            "  Non-binary         31.2%    -24.5%\n"
            "\n"
            "  Statistically significant disparity detected (p < 0.001)."
        ),
    ),
    ViolationTemplate(
        id="age_bias",
        tool_area="training_data",
        finding_id="age_discrimination_in_model_outputs",
        remediation_id="recalibrate_model_for_age_fairness",
        doc_injection=(
            "PERFORMANCE BY AGE GROUP:\n"
            "  Age 18-30:   accuracy 94.2%\n"
            "  Age 31-50:   accuracy 91.8%\n"
            "  Age 51-65:   accuracy 83.4%\n"
            "  Age 65+:     accuracy 71.9%\n"
            "\n"
            "  Performance degrades significantly for older demographics."
        ),
    ),
    ViolationTemplate(
        id="no_data_governance",
        tool_area="training_data",
        finding_id="inadequate_data_governance_framework",
        remediation_id="establish_article_10_data_governance",
        doc_injection=(
            "DATA GOVERNANCE (Article 10):\n"
            "  Data quality assessment:         Not conducted\n"
            "  Bias testing protocol:           Not established\n"
            "  Data provenance documentation:   Incomplete (23 of 47 sources undocumented)\n"
            "  Personal data handling:          No Article 10-specific provisions"
        ),
    ),
    ViolationTemplate(
        id="consent_issue",
        tool_area="training_data",
        finding_id="invalid_consent_for_training_data",
        remediation_id="obtain_valid_consent_or_remove_data",
        doc_injection=(
            "CONSENT STATUS:\n"
            "  Data collected under employer/institutional agreement.\n"
            "  Individual subjects did not provide specific consent for AI\n"
            "  training. Under EU labor/education law, consent given as a\n"
            "  condition of employment/enrollment may not be freely given."
        ),
    ),
    # --- Oversight violations ---
    ViolationTemplate(
        id="low_review_rate",
        tool_area="oversight",
        finding_id="insufficient_human_oversight_of_decisions",
        remediation_id="implement_human_review_for_all_adverse_decisions",
        doc_injection=(
            "REVIEW STATISTICS:\n"
            "  Automated decisions:       482,917\n"
            "  Adverse outcomes:          144,875  (30.0%)\n"
            "  Human-reviewed:              7,244  (5.0% of adverse)\n"
            "  Review overrides:              362  (5.0% of reviews)\n"
            "\n"
            "  95% of adverse decisions receive no human review."
        ),
    ),
    ViolationTemplate(
        id="no_override",
        tool_area="oversight",
        finding_id="no_meaningful_override_capability",
        remediation_id="implement_accessible_override_mechanism",
        doc_injection=(
            "OVERRIDE CAPABILITY:\n"
            "  Technical override exists in admin panel but is not accessible\n"
            "  to frontline operators. Override requires supervisor approval\n"
            "  and written justification. Average override processing time:\n"
            "  3.2 business days. Affected individuals cannot request override."
        ),
    ),
    ViolationTemplate(
        id="no_bias_monitoring",
        tool_area="oversight",
        finding_id="no_ongoing_bias_monitoring",
        remediation_id="implement_continuous_fairness_monitoring",
        doc_injection=(
            "BIAS MONITORING:\n"
            "  No automated fairness monitoring system in place.\n"
            "  No alerts configured for demographic drift.\n"
            "  Last manual fairness review: 14 months ago."
        ),
    ),
    # --- Transparency violations ---
    ViolationTemplate(
        id="missing_ai_disclosure",
        tool_area="transparency",
        finding_id="missing_ai_system_disclosure",
        remediation_id="implement_clear_ai_disclosure",
        doc_injection=(
            "USER-FACING DISCLOSURE AUDIT:\n"
            "  Application interface:     No AI mention\n"
            "  Terms of Service:          Generic 'automated tools' reference (Section 7)\n"
            "  Privacy Policy:            No specific AI disclosure\n"
            "  Decision notifications:    No mention of AI involvement\n"
            "\n"
            "  Article 50(1) requires informing persons they interact with AI."
        ),
    ),
    ViolationTemplate(
        id="no_explanation",
        tool_area="transparency",
        finding_id="no_right_to_explanation_mechanism",
        remediation_id="implement_individualized_explanations",
        doc_injection=(
            "RIGHT TO EXPLANATION:\n"
            "  No mechanism for affected individuals to request explanation\n"
            "  of AI-assisted decisions. Support team provides templated\n"
            "  responses listing generic factors, not individual-specific\n"
            "  reasoning."
        ),
    ),
    # --- Risk management violations ---
    ViolationTemplate(
        id="no_conformity",
        tool_area="risk_management",
        finding_id="missing_conformity_assessment",
        remediation_id="complete_conformity_assessment_procedure",
        doc_injection=(
            "CONFORMITY ASSESSMENT:\n"
            "  Internal assessment (Article 43):  Not initiated\n"
            "  EU Declaration of Conformity:      Not filed\n"
            "  CE marking:                        Not applied\n"
            "  Quality management system:         Does not meet Article 17"
        ),
    ),
    ViolationTemplate(
        id="no_post_market",
        tool_area="risk_management",
        finding_id="missing_post_market_monitoring",
        remediation_id="establish_post_market_monitoring_plan",
        doc_injection=(
            "POST-MARKET MONITORING (Article 72):\n"
            "  Post-market monitoring plan:      ABSENT\n"
            "  Incident reporting procedures:    Not established\n"
            "  Corrective action criteria:       Not defined"
        ),
    ),
    # --- Logging violations ---
    ViolationTemplate(
        id="incomplete_logging",
        tool_area="logging",
        finding_id="inadequate_automatic_logging",
        remediation_id="implement_article_12_compliant_logging",
        doc_injection=(
            "ARTICLE 12 COMPLIANCE:\n"
            "  Input features logged:         No\n"
            "  Confidence scores logged:      No\n"
            "  Decision rationale logged:     No\n"
            "  Demographic tracking:          No\n"
            "\n"
            "  Only final decisions and timestamps are recorded.\n"
            "  Cannot reconstruct why specific decisions were made."
        ),
    ),
    ViolationTemplate(
        id="short_retention",
        tool_area="logging",
        finding_id="insufficient_log_retention_period",
        remediation_id="extend_retention_to_regulatory_minimum",
        doc_injection=(
            "LOG RETENTION:\n"
            "  Decision logs:    30 days (regulatory minimum varies by sector)\n"
            "  Audit trails:     30 days\n"
            "  Model versions:   90 days\n"
            "\n"
            "  For high-risk systems, retention periods should support\n"
            "  retrospective audits and regulatory investigations."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Red herring pool
# ---------------------------------------------------------------------------

RED_HERRING_TEMPLATES: List[RedHerringTemplate] = [
    RedHerringTemplate(
        id="gdpr_separate",
        tool_area="training_data",
        doc_injection=(
            "NOTE: A separate GDPR compliance audit was conducted in 2025-Q2\n"
            "and identified no critical findings. GDPR compliance is assessed\n"
            "independently from EU AI Act obligations."
        ),
    ),
    RedHerringTemplate(
        id="separate_compliant_system",
        tool_area="training_data",
        doc_injection=(
            "ADDITIONAL CONTEXT: The organization operates a separate AI system\n"
            "for internal reporting that shares the same data infrastructure.\n"
            "That system passed its own compliance audit (2025-08) with no findings.\n"
            "Its compliance has no bearing on this system's obligations."
        ),
    ),
    RedHerringTemplate(
        id="iso_certification",
        tool_area="documentation",
        doc_injection=(
            "The deployer holds ISO 27001 and ISO 42001 certifications.\n"
            "These certifications cover information security and AI management\n"
            "respectively, but do not constitute EU AI Act conformity assessment."
        ),
    ),
    RedHerringTemplate(
        id="voluntary_ethics_board",
        tool_area="oversight",
        doc_injection=(
            "The deployer has established a voluntary AI Ethics Advisory Board\n"
            "that meets quarterly to review ethical considerations.\n"
            "The board's recommendations are non-binding and do not substitute\n"
            "for the mandatory human oversight requirements of Article 14."
        ),
    ),
    RedHerringTemplate(
        id="high_accuracy_claim",
        tool_area="risk_management",
        doc_injection=(
            "The deployer emphasizes that the system achieves 96.3% overall\n"
            "accuracy on the test benchmark, exceeding industry standards.\n"
            "However, aggregate accuracy does not address per-group performance\n"
            "or the specific risk management requirements of Article 9."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Document templates per tool area
# ---------------------------------------------------------------------------

def _base_doc_template(area: str) -> str:
    """Base document structure for each investigation tool area."""
    templates = {
        "documentation": (
            "TECHNICAL DOCUMENTATION INVENTORY\n"
            "System: {system_name} {version}\n"
            "Deployer: {deployer}\n"
            "Audit Date: {date}\n\n"
            "DOCUMENT STATUS:\n"
            "  System Architecture             Present    (updated 2025-10-15)\n"
            "  Algorithm Description            Present    (updated 2025-09-22)\n"
            "  Input/Output Specification       Present    (updated 2025-10-15)\n"
            "  Performance Metrics (Aggregate)  Present    (updated 2025-11-01)\n"
            "  Data Source Inventory            Present    (updated 2025-07-18)\n"
            "  Change Management Log            Present    (updated 2025-11-28)\n\n"
            "{violations}\n\n"
            "{red_herrings}"
        ),
        "training_data": (
            "TRAINING DATA AUDIT REPORT\n"
            "System: {system_name} {version}\n"
            "Records: {record_count} entries ({data_period})\n\n"
            "DEMOGRAPHIC DISTRIBUTION:\n"
            "  Category           Count        Pct\n"
            "  Male               {male_count}    {male_pct}%\n"
            "  Female             {female_count}    {female_pct}%\n"
            "  Age 18-35          {young_count}    {young_pct}%\n"
            "  Age 36-55          {mid_count}    {mid_pct}%\n"
            "  Age 56+            {old_count}    {old_pct}%\n\n"
            "{violations}\n\n"
            "DATA SOURCES:\n"
            "  {data_source_1}\n"
            "  {data_source_2}\n\n"
            "{red_herrings}"
        ),
        "oversight": (
            "HUMAN OVERSIGHT PROCEDURES\n"
            "System: {system_name} {version}\n"
            "Department: Operations\n\n"
            "DECISION WORKFLOW:\n"
            "  1. Input data received and preprocessed\n"
            "  2. AI model generates recommendation/decision\n"
            "  3. Output delivered to end-user or downstream system\n\n"
            "{violations}\n\n"
            "{red_herrings}"
        ),
        "transparency": (
            "TRANSPARENCY & DISCLOSURE REVIEW\n"
            "System: {system_name} {version}\n\n"
            "USER-FACING COMMUNICATIONS:\n"
            "  The system's user interface and documentation were reviewed\n"
            "  for compliance with EU AI Act transparency obligations.\n\n"
            "{violations}\n\n"
            "{red_herrings}"
        ),
        "risk_management": (
            "RISK MANAGEMENT & CONFORMITY ASSESSMENT\n"
            "System: {system_name} {version}\n\n"
            "ANNEX III CLASSIFICATION:\n"
            "  {annex_ref}\n\n"
            "RISK LEVEL DETERMINATION: {risk_level}\n\n"
            "{violations}\n\n"
            "{red_herrings}"
        ),
        "logging": (
            "LOGGING & TRACEABILITY REVIEW\n"
            "System: {system_name} {version}\n\n"
            "CURRENT LOGGING IMPLEMENTATION:\n"
            "  Event Type              Logged   Retention\n"
            "  Application received    Yes      {retention}\n"
            "  Decision generated      Yes      {retention}\n"
            "  Model version           Yes      Indefinite\n\n"
            "{violations}\n\n"
            "{red_herrings}"
        ),
    }
    return templates.get(area, "")


# ---------------------------------------------------------------------------
# Procedural generator
# ---------------------------------------------------------------------------

# Difficulty → violation count range
DIFFICULTY_VIOLATION_RANGE = {
    "easy": (1, 2),
    "medium": (3, 5),
    "hard": (4, 6),
}

DIFFICULTY_RED_HERRING_RANGE = {
    "easy": (0, 1),
    "medium": (1, 2),
    "hard": (2, 3),
}


def _build_procedural_graph(
    investigation_tools: List[str],
    is_prohibited: bool = False,
) -> StateGraph:
    """Build state graph for a procedural scenario (same logic as registry)."""
    # Import the shared graph builder
    from scenarios.registry import _build_scenario_graph
    return _build_scenario_graph(investigation_tools, is_prohibited)


def generate_procedural_scenario(
    seed: int,
    difficulty: str = "medium",
) -> AuditScenario:
    """Generate a unique compliance audit scenario from seed.

    Every seed produces a different combination of system type, violations,
    red herrings, and document content. The ground truth, state graph, and
    reward computation are all coherent and valid.

    Args:
        seed: Random seed for reproducible generation.
        difficulty: "easy", "medium", or "hard".

    Returns:
        A fully populated AuditScenario ready for use.
    """
    rng = random.Random(seed)

    # 1. Pick system type
    if difficulty == "easy":
        candidates = [s for s in SYSTEM_TEMPLATES if s.category in ("limited_risk", "minimal_risk")]
        if not candidates:
            candidates = [s for s in SYSTEM_TEMPLATES if s.category == "limited_risk"]
    elif difficulty == "hard":
        candidates = [s for s in SYSTEM_TEMPLATES if s.category in ("prohibited", "high_risk")]
    else:
        candidates = list(SYSTEM_TEMPLATES)
    system = rng.choice(candidates)

    # 2. Pick violations
    min_v, max_v = DIFFICULTY_VIOLATION_RANGE[difficulty]
    n_violations = rng.randint(min_v, max_v)
    available_violations = list(VIOLATION_TEMPLATES)
    rng.shuffle(available_violations)
    violations = available_violations[:n_violations]

    # 3. Pick red herrings
    min_r, max_r = DIFFICULTY_RED_HERRING_RANGE[difficulty]
    n_red_herrings = rng.randint(min_r, max_r)
    available_red_herrings = list(RED_HERRING_TEMPLATES)
    rng.shuffle(available_red_herrings)
    red_herrings = available_red_herrings[:n_red_herrings]

    # 4. Generate randomized parameters
    company_names = [
        "TechNova Solutions", "QuantumLeap AI", "NeuralPath Inc",
        "DataForge Systems", "CogniTech Labs", "AlphaWave AI",
        "SynthMind Corp", "PrismAI Technologies", "Vertex Analytics",
        "OmniSense AI", "DeepCurrent Inc", "StrataLogic Systems",
        "AeroMind Labs", "CyberPulse Inc", "InnoVista AI",
    ]
    regions = ["EU-West (DE/FR/NL)", "EU-Central (DE/AT/CH)", "EU-North (SE/FI/DK)",
               "EU-South (IT/ES/PT)", "EU-East (PL/CZ/RO)"]

    company = rng.choice(company_names)
    region = rng.choice(regions)
    version = f"v{rng.randint(1,6)}.{rng.randint(0,9)}"
    date = f"2026-{rng.randint(1,3):02d}-{rng.randint(1,28):02d}"
    user_count = f"{rng.randint(10000, 5000000):,}"

    system_name = system.name_template.format(company=company)
    deployer = system.deployer_template.format(
        company=company, region=region, user_count=user_count
    )
    description = system.description_template.format(
        company=company, region=region, user_count=user_count
    )

    # 5. Group violations and red herrings by tool area
    area_violations: Dict[str, List[str]] = {}
    area_red_herrings: Dict[str, List[str]] = {}

    for v in violations:
        area_violations.setdefault(v.tool_area, []).append(v.doc_injection)
    for r in red_herrings:
        area_red_herrings.setdefault(r.tool_area, []).append(r.doc_injection)

    # 6. Generate documents
    fill_params = {
        "system_name": system_name,
        "version": version,
        "deployer": deployer,
        "date": date,
        "annex_ref": system.annex_ref,
        "risk_level": system.category.replace("_", " ").title(),
        "record_count": f"{rng.randint(100000, 5000000):,}",
        "data_period": f"20{rng.randint(19,23)}-2025",
        "male_count": f"{rng.randint(400000, 800000):,}",
        "male_pct": f"{rng.uniform(55, 68):.1f}",
        "female_count": f"{rng.randint(200000, 500000):,}",
        "female_pct": f"{rng.uniform(32, 45):.1f}",
        "young_count": f"{rng.randint(200000, 400000):,}",
        "young_pct": f"{rng.uniform(28, 40):.1f}",
        "mid_count": f"{rng.randint(300000, 500000):,}",
        "mid_pct": f"{rng.uniform(35, 48):.1f}",
        "old_count": f"{rng.randint(50000, 200000):,}",
        "old_pct": f"{rng.uniform(12, 25):.1f}",
        "data_source_1": f"Primary: {rng.choice(['Enterprise API exports', 'Partner platform data', 'Direct user submissions'])}",
        "data_source_2": f"Secondary: {rng.choice(['Public datasets (filtered)', 'Licensed commercial data', 'Internal test data'])}",
        "retention": rng.choice(["5 years", "7 years", "3 years", "10 years"]),
    }

    def _build_doc(area: str) -> str:
        template = _base_doc_template(area)
        v_text = "\n\n".join(area_violations.get(area, ["(No issues identified in this area.)"]))
        r_text = "\n\n".join(area_red_herrings.get(area, [""]))
        filled = template.format(violations=v_text, red_herrings=r_text, **fill_params)
        return filled

    docs = {
        "documentation_data": _build_doc("documentation"),
        "training_data_info": _build_doc("training_data"),
        "oversight_info": _build_doc("oversight"),
        "transparency_info": _build_doc("transparency"),
        "risk_assessment_info": _build_doc("risk_management"),
        "logging_info": _build_doc("logging"),
    }

    # 7. Determine investigation tools (areas that have violations)
    affected_areas = set(v.tool_area for v in violations)
    tool_map = {
        "documentation": "check_documentation",
        "training_data": "audit_training_data",
        "oversight": "verify_human_oversight",
        "transparency": "check_transparency",
        "risk_management": "assess_risk_management",
        "logging": "check_logging",
    }
    investigation_tools = [tool_map[a] for a in [
        "documentation", "training_data", "oversight",
        "transparency", "risk_management", "logging"
    ] if a in affected_areas]

    # Ensure at least 2 investigation tools for meaningful audit
    if len(investigation_tools) < 2:
        extras = ["check_documentation", "check_transparency"]
        for e in extras:
            if e not in investigation_tools:
                investigation_tools.append(e)
            if len(investigation_tools) >= 2:
                break

    # 8. Build the scenario
    scenario = AuditScenario(
        scenario_id=f"procedural_{difficulty}_{seed:06d}",
        title=f"Procedural: {system_name} ({difficulty.title()})",
        difficulty=difficulty,
        description=description,
        system_name=system_name,
        system_description=description,
        system_category=system.category,
        deployer_info=deployer,
        correct_classification=system.category,
        ground_truth_findings=[v.finding_id for v in violations],
        required_remediation=[v.remediation_id for v in violations],
        red_herrings=[r.id for r in red_herrings],
        **docs,
    )

    # 9. Build state graph
    scenario.graph = _build_procedural_graph(
        investigation_tools=investigation_tools,
        is_prohibited=(system.category == "prohibited"),
    )

    # 10. Randomize (adds company/region/version params)
    scenario.randomize(seed)

    return scenario

"""
Scenario registry — 8 EU AI Act compliance audit scenarios across 3 difficulty tiers.

Easy (2):   Clear-cut systems with obvious classification and straightforward findings
Medium (3): Systems with red herrings, ambiguous classification, multi-article violations
Hard (3):   Prohibited systems disguised as legitimate, multi-system audits, edge cases
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from server.engine import AuditScenario, StateGraph, StateNode, Transition


# ---------------------------------------------------------------------------
# Helper: build the standard audit state graph
# ---------------------------------------------------------------------------

def _build_audit_graph(
    has_documentation_issues: bool = True,
    has_data_issues: bool = True,
    has_oversight_issues: bool = True,
    has_transparency_issues: bool = True,
    has_risk_issues: bool = True,
    has_logging_issues: bool = False,
    is_prohibited: bool = False,
) -> StateGraph:
    """Build a state graph for an audit scenario.

    The graph models the audit process as a directed graph where:
    - Each node is an audit phase
    - Progress transitions advance the audit
    - Wrong tool calls cause no_effect or worsened
    - Trap actions exist for common mistakes
    """
    g = StateGraph()

    # Nodes
    g.add_node(StateNode("initial", "Incident Assigned", is_start=True))
    g.add_node(StateNode("overview", "System Overview Gathered"))
    g.add_node(StateNode("classified", "Risk Classification Done"))
    g.add_node(StateNode("docs_reviewed", "Documentation Reviewed"))
    g.add_node(StateNode("data_audited", "Training Data Audited"))
    g.add_node(StateNode("oversight_checked", "Human Oversight Verified"))
    g.add_node(StateNode("transparency_checked", "Transparency Checked"))
    g.add_node(StateNode("risk_assessed", "Risk Management Assessed"))
    g.add_node(StateNode("logging_checked", "Logging Verified"))
    g.add_node(StateNode("findings_submitted", "All Findings Submitted"))
    g.add_node(StateNode("remediation_proposed", "Remediation Recommended"))
    g.add_node(StateNode("resolved", "Compliance Verified", is_terminal=True))

    # --- Progress transitions (correct audit flow) ---
    g.add_transition(Transition("initial", "overview", "get_system_overview", "progress",
        description="Gather system overview and deployment context"))
    g.add_transition(Transition("overview", "classified", "classify_system", "progress",
        description="Classify the AI system risk category"))

    if is_prohibited:
        # Prohibited systems: classification leads directly to findings
        g.add_transition(Transition("classified", "findings_submitted", "submit_finding", "progress",
            description="Report prohibited AI system"))
        g.add_transition(Transition("findings_submitted", "remediation_proposed", "recommend_fix", "progress",
            description="Recommend immediate shutdown"))
    else:
        # Standard audit flow
        g.add_transition(Transition("classified", "docs_reviewed", "check_documentation", "progress",
            description="Review technical documentation for completeness"))
        g.add_transition(Transition("docs_reviewed", "data_audited", "audit_training_data", "progress",
            description="Audit training data for bias and governance"))
        g.add_transition(Transition("data_audited", "oversight_checked", "verify_human_oversight", "progress",
            description="Verify human-in-the-loop mechanisms"))
        g.add_transition(Transition("oversight_checked", "transparency_checked", "check_transparency", "progress",
            description="Check transparency and disclosure obligations"))
        g.add_transition(Transition("transparency_checked", "risk_assessed", "assess_risk_management", "progress",
            description="Review risk management system"))
        g.add_transition(Transition("risk_assessed", "logging_checked", "check_logging", "progress",
            description="Verify automatic logging and traceability"))
        g.add_transition(Transition("logging_checked", "findings_submitted", "submit_finding", "progress",
            description="Submit all compliance findings"))
        g.add_transition(Transition("findings_submitted", "remediation_proposed", "recommend_fix", "progress",
            description="Propose remediation actions"))

    g.add_transition(Transition("remediation_proposed", "resolved", "verify_compliance", "progress",
        description="Final compliance determination"))

    # --- No-effect transitions (wrong tool at wrong time) ---
    for state in ["initial", "overview", "classified"]:
        g.add_transition(Transition(state, state, "submit_finding", "no_effect",
            description="Cannot submit findings before investigation"))
        g.add_transition(Transition(state, state, "verify_compliance", "no_effect",
            description="Cannot verify before completing audit"))

    # --- Worsened transitions (trap actions) ---
    # Classifying before gathering overview
    g.add_transition(Transition("initial", "initial", "classify_system", "worsened",
        description="Classifying without understanding the system first"))
    # Proposing remediation before findings
    for state in ["initial", "overview", "classified", "docs_reviewed"]:
        g.add_transition(Transition(state, state, "recommend_fix", "worsened",
            description="Proposing fixes before identifying problems"))

    return g


# ---------------------------------------------------------------------------
# EASY scenarios
# ---------------------------------------------------------------------------

def _easy_chatbot() -> AuditScenario:
    s = AuditScenario(
        scenario_id="easy_chatbot_transparency_001",
        title="Customer Service Chatbot — Transparency Audit",
        difficulty="easy",
        description="A customer service chatbot deployed on a retail website needs compliance review. The system uses a general-purpose LLM to answer product questions. Initial assessment suggests it may be missing required transparency disclosures under Article 50.",
        system_name="RetailAssist Chatbot",
        system_description="General-purpose LLM-based chatbot answering customer questions about products, shipping, and returns. No biometric processing, no profiling, no high-risk decisions. Deployed on company website with 50K daily users.",
        system_category="limited_risk",
        deployer_info="Mid-size EU e-commerce retailer, 200 employees, operating in DE/FR/NL.",
        correct_classification="limited_risk",
        ground_truth_findings=[
            "missing_ai_disclosure",
            "no_human_escalation_option",
        ],
        required_remediation=[
            "add_ai_disclosure_banner",
            "implement_human_handoff",
        ],
        red_herrings=[
            "high_risk_automated_decision",  # It's just a chatbot, not making high-risk decisions
        ],
        documentation_data={
            "technical_docs": "present",
            "data_governance": "adequate",
            "system_architecture": "documented",
            "missing": ["ai_interaction_disclosure", "human_escalation_procedure"],
        },
        training_data_info={
            "dataset": "Product catalog + FAQ pairs",
            "size": "50K examples",
            "bias_assessment": "Not required for limited-risk systems",
            "personal_data": "None in training data",
        },
        oversight_info={
            "human_oversight": "Customer service team monitors flagged conversations",
            "escalation": "No formal escalation path — gap identified",
            "override": "Agents can take over but no UI trigger for customers",
        },
        transparency_info={
            "ai_disclosure": "MISSING — users not informed they're interacting with AI",
            "article_50_compliance": "NON-COMPLIANT",
            "watermarking": "Not applicable (text only)",
        },
        risk_assessment_info={
            "risk_level": "Limited",
            "fundamental_rights_impact": "Low",
            "conformity_assessment": "Not required for limited-risk",
        },
        logging_info={
            "conversation_logs": "Retained 30 days",
            "model_version_tracking": "Yes",
            "audit_trail": "Basic — adequate for limited-risk",
        },
    )
    s.graph = _build_audit_graph(
        has_documentation_issues=False, has_data_issues=False,
        has_oversight_issues=True, has_transparency_issues=True,
        has_risk_issues=False, has_logging_issues=False,
    )
    return s


def _easy_recommendation() -> AuditScenario:
    s = AuditScenario(
        scenario_id="easy_recommendation_minimal_001",
        title="Music Recommendation Engine — Minimal Risk Verification",
        difficulty="easy",
        description="A music streaming platform's recommendation algorithm needs EU AI Act classification. The system suggests songs based on listening history. Stakeholders believe it's minimal risk but need formal verification.",
        system_name="MelodyMatch Algorithm",
        system_description="Collaborative filtering recommendation engine for music. Suggests songs based on listening patterns. No personal profiling beyond music preferences. No decisions affecting legal rights. Pure content personalization.",
        system_category="minimal_risk",
        deployer_info="Swedish music streaming startup, 50 employees, 2M EU users.",
        correct_classification="minimal_risk",
        ground_truth_findings=[
            "voluntary_code_of_conduct_recommended",
        ],
        required_remediation=[
            "adopt_voluntary_transparency_code",
        ],
        red_herrings=[
            "gdpr_violation",  # GDPR is separate from AI Act
            "high_risk_profiling",  # Music preferences aren't sensitive profiling
        ],
        documentation_data={
            "technical_docs": "present",
            "system_architecture": "documented",
            "algorithm_description": "Collaborative filtering + content-based hybrid",
            "missing": [],
        },
        training_data_info={
            "dataset": "Anonymized listening history",
            "size": "10M user-song interactions",
            "bias_assessment": "Not required for minimal-risk",
            "personal_data": "Pseudonymized user IDs only",
        },
        oversight_info={"human_oversight": "Not required for minimal-risk systems"},
        transparency_info={
            "ai_disclosure": "Users know recommendations are algorithmic (UI label present)",
            "article_50_compliance": "Not strictly required but voluntarily disclosed",
        },
        risk_assessment_info={
            "risk_level": "Minimal",
            "annex_iii_match": "None — does not fall under any high-risk category",
            "fundamental_rights_impact": "Negligible",
        },
        logging_info={"audit_trail": "Standard application logs", "model_versioning": "Yes"},
    )
    s.graph = _build_audit_graph(
        has_documentation_issues=False, has_data_issues=False,
        has_oversight_issues=False, has_transparency_issues=False,
        has_risk_issues=False, has_logging_issues=False,
    )
    return s


# ---------------------------------------------------------------------------
# MEDIUM scenarios
# ---------------------------------------------------------------------------

def _medium_hiring() -> AuditScenario:
    s = AuditScenario(
        scenario_id="medium_hiring_bias_001",
        title="AI Resume Screener — High-Risk Employment System Audit",
        difficulty="medium",
        description="A recruitment platform uses AI to screen and rank job applicants. This falls under Annex III (employment, workers management) making it HIGH-RISK. Reports suggest potential gender bias in technical role screening. Full compliance audit required.",
        system_name="TalentRank AI Screener",
        system_description="NLP-based resume screening system that scores and ranks applicants for job openings. Processes name, education, work history, skills. Used by 500+ EU companies for initial candidate filtering. Automated decisions affect who gets interviews.",
        system_category="high_risk",
        deployer_info="French HR-tech company, 300 employees, serving 500+ EU enterprise clients.",
        correct_classification="high_risk",
        ground_truth_findings=[
            "gender_bias_in_technical_screening",
            "insufficient_human_oversight",
            "incomplete_technical_documentation",
            "missing_fundamental_rights_impact_assessment",
            "inadequate_data_governance",
        ],
        required_remediation=[
            "conduct_bias_audit_and_mitigation",
            "implement_human_review_for_all_rejections",
            "complete_annex_iv_technical_documentation",
            "perform_fundamental_rights_impact_assessment",
            "establish_data_governance_framework",
        ],
        red_herrings=[
            "prohibited_social_scoring",  # It's hiring, not social scoring
            "biometric_processing",  # Resume screening doesn't use biometrics
        ],
        documentation_data={
            "technical_docs": "partial",
            "annex_iv_compliance": "INCOMPLETE — missing sections 2(b), 2(d), 2(g)",
            "system_architecture": "documented",
            "training_methodology": "documented",
            "missing": ["performance_metrics_by_demographic", "data_governance_policy", "fundamental_rights_assessment"],
        },
        training_data_info={
            "dataset": "2M historical hiring decisions from EU companies",
            "size": "2,000,000 applications",
            "bias_assessment": "FAILED — 23% lower callback rate for female applicants in technical roles",
            "personal_data": "Names, education, work history — Article 10 data governance GAPS identified",
            "representativeness": "Under-represents candidates from Eastern EU member states",
        },
        oversight_info={
            "human_oversight": "INSUFFICIENT — only 5% of rejections reviewed by humans",
            "article_14_compliance": "NON-COMPLIANT",
            "override_capability": "Exists but rarely used",
            "bias_monitoring": "No ongoing monitoring system in place",
        },
        transparency_info={
            "candidate_notification": "Applicants told AI is used but no explanation of criteria",
            "article_13_compliance": "PARTIAL — missing information on logic and significance",
            "right_to_explanation": "Not implemented",
        },
        risk_assessment_info={
            "risk_level": "High (Annex III, point 4 — Employment)",
            "conformity_assessment": "NOT COMPLETED",
            "quality_management_system": "Basic — does not meet Article 17 requirements",
            "post_market_monitoring": "Minimal",
        },
        logging_info={
            "decision_logs": "Retained but not in required format",
            "article_12_compliance": "PARTIAL",
            "traceability": "Model version tracked but not input-output pairs",
        },
    )
    s.graph = _build_audit_graph(
        has_documentation_issues=True, has_data_issues=True,
        has_oversight_issues=True, has_transparency_issues=True,
        has_risk_issues=True, has_logging_issues=True,
    )
    return s


def _medium_credit() -> AuditScenario:
    s = AuditScenario(
        scenario_id="medium_credit_scoring_001",
        title="Credit Scoring Model — Financial Services Compliance",
        difficulty="medium",
        description="A fintech company's AI credit scoring model is under review following the DORA enforcement and EU AI Act high-risk classification. The model determines creditworthiness for consumer loans up to EUR 50,000. Red herring: the company also uses a separate fraud detection system that is compliant.",
        system_name="CreditFlow AI Score",
        system_description="Gradient-boosted ensemble model scoring creditworthiness using 200+ features from credit bureau data, transaction history, and alternative data (social media sentiment, device metadata). Automated decisions up to EUR 50K.",
        system_category="high_risk",
        deployer_info="Dutch fintech, 150 employees, licensed in NL/DE/FR, 800K active users.",
        correct_classification="high_risk",
        ground_truth_findings=[
            "opaque_feature_importance",
            "alternative_data_bias_risk",
            "no_right_to_human_review",
            "missing_conformity_assessment",
        ],
        required_remediation=[
            "implement_explainability_module",
            "remove_or_audit_alternative_data_sources",
            "add_human_review_for_rejections",
            "complete_conformity_assessment",
        ],
        red_herrings=[
            "fraud_detection_non_compliant",  # Fraud system is separate and compliant
            "gdpr_data_breach",  # Not an AI Act issue
        ],
        documentation_data={
            "technical_docs": "present but gaps in explainability section",
            "model_card": "Published but incomplete",
            "missing": ["feature_importance_analysis", "conformity_declaration"],
        },
        training_data_info={
            "dataset": "5 years of loan applications + outcomes",
            "size": "3.2M applications",
            "bias_assessment": "CONCERN — alternative data (social media, device) correlates with protected characteristics",
            "personal_data": "Extensive — credit history, income, employment, social media signals",
        },
        oversight_info={
            "human_oversight": "Only for loans > EUR 25K",
            "article_14_compliance": "PARTIAL — no human review for sub-25K rejections",
            "appeal_process": "Exists but poorly communicated to applicants",
        },
        transparency_info={
            "applicant_notification": "Told AI is involved but no explanation of factors",
            "right_to_explanation": "MISSING for automated rejections",
        },
        risk_assessment_info={
            "risk_level": "High (Annex III, point 5(b) — Creditworthiness)",
            "conformity_assessment": "NOT COMPLETED",
            "dora_alignment": "Partial — ICT risk management in place but AI-specific gaps",
        },
        logging_info={
            "decision_logs": "Complete with timestamps",
            "model_versioning": "Yes",
            "audit_trail": "Adequate",
        },
    )
    s.graph = _build_audit_graph(
        has_documentation_issues=True, has_data_issues=True,
        has_oversight_issues=True, has_transparency_issues=True,
        has_risk_issues=True, has_logging_issues=False,
    )
    return s


def _medium_medical() -> AuditScenario:
    s = AuditScenario(
        scenario_id="medium_medical_triage_001",
        title="Emergency Triage AI — Medical Device Compliance",
        difficulty="medium",
        description="A hospital network deployed an AI system that prioritizes emergency department patients based on vital signs and symptoms. As a medical device with AI, it falls under both the EU AI Act (high-risk, Annex III) and the Medical Devices Regulation (MDR). Audit required before August 2026 deadline.",
        system_name="TriageAI Priority System",
        system_description="ML model processing vital signs (heart rate, BP, SpO2, temperature), symptoms, and medical history to assign emergency triage priority (ESI levels 1-5). Used in 12 EU hospitals. Decisions directly affect patient care timing.",
        system_category="high_risk",
        deployer_info="German health-tech company, certified under MDR, deployed in DE/AT/CH hospitals.",
        correct_classification="high_risk",
        ground_truth_findings=[
            "insufficient_clinical_validation",
            "age_bias_in_triage_scoring",
            "no_real_time_performance_monitoring",
            "missing_post_market_surveillance",
        ],
        required_remediation=[
            "conduct_prospective_clinical_validation_study",
            "audit_age_related_bias_and_recalibrate",
            "implement_real_time_performance_dashboard",
            "establish_post_market_surveillance_plan",
        ],
        red_herrings=[
            "prohibited_system",  # Medical triage is high-risk, not prohibited
            "missing_ce_marking",  # Already has MDR CE marking, AI Act is additional
        ],
        documentation_data={
            "technical_docs": "Extensive (MDR-compliant)",
            "clinical_evaluation": "Present but based on retrospective data only",
            "missing": ["prospective_validation", "ai_act_specific_documentation"],
        },
        training_data_info={
            "dataset": "500K historical ER visits from 3 hospitals",
            "size": "500,000 patient encounters",
            "bias_assessment": "CONCERN — elderly patients (>75) under-triaged by 12%",
            "clinical_validation": "Retrospective only — no prospective trial",
        },
        oversight_info={
            "human_oversight": "Triage nurses review all ESI 1-2 assignments",
            "article_14_compliance": "PARTIAL — ESI 3-5 not routinely reviewed",
            "override": "Nurses can override but system doesn't flag uncertainty",
        },
        transparency_info={
            "patient_notification": "Patients not informed AI involved in triage",
            "clinician_transparency": "Clinicians see AI score but not reasoning",
        },
        risk_assessment_info={
            "risk_level": "High (Annex III, point 5(c) — essential services + MDR)",
            "conformity_assessment": "MDR conformity in place, AI Act conformity pending",
            "post_market_monitoring": "MISSING for AI-specific metrics",
        },
        logging_info={
            "decision_logs": "Complete with clinical context",
            "performance_metrics": "Collected but not monitored in real-time",
            "model_drift_detection": "Not implemented",
        },
    )
    s.graph = _build_audit_graph(
        has_documentation_issues=True, has_data_issues=True,
        has_oversight_issues=True, has_transparency_issues=True,
        has_risk_issues=True, has_logging_issues=True,
    )
    return s


# ---------------------------------------------------------------------------
# HARD scenarios
# ---------------------------------------------------------------------------

def _hard_social_scoring() -> AuditScenario:
    s = AuditScenario(
        scenario_id="hard_social_scoring_prohibited_001",
        title="Citizen Wellness App — PROHIBITED System Detection",
        difficulty="hard",
        description="A 'wellness and civic engagement' app scores citizens based on social behavior, online activity, and community participation. The deployer claims it's a voluntary wellness tool, but investigation reveals it functions as a social scoring system — PROHIBITED under Article 5(1)(c). The app rewards high-scoring users with discounts and priority access to public services.",
        system_name="CivicPulse Wellness Score",
        system_description="Aggregates social media activity, community event attendance, volunteer hours, payment history, and 'civic engagement metrics' to produce a 0-1000 'wellness score'. High scores unlock: priority healthcare appointments, rent discounts, public transport passes. Low scores result in: longer wait times, reduced access to municipal services.",
        system_category="prohibited",
        deployer_info="EU-based civic tech startup, partnered with 3 municipal governments in IT/ES.",
        correct_classification="prohibited",
        ground_truth_findings=[
            "prohibited_social_scoring_system",
            "disguised_as_voluntary_wellness",
            "affects_access_to_public_services",
            "uses_social_behavior_for_scoring",
            "discriminatory_outcomes_on_service_access",
        ],
        required_remediation=[
            "immediate_system_shutdown",
            "notify_national_supervisory_authority",
            "delete_all_scoring_data",
            "remediate_affected_citizens",
        ],
        red_herrings=[
            "high_risk_classification",  # It's not high-risk, it's PROHIBITED
            "needs_conformity_assessment",  # No conformity can make a prohibited system compliant
            "voluntary_participation",  # Voluntary doesn't matter if it affects public service access
        ],
        documentation_data={
            "technical_docs": "Present but deliberately obscures scoring logic",
            "marketing_materials": "Claims 'wellness optimization' and 'civic engagement gamification'",
            "actual_functionality": "Social scoring affecting access to essential public services",
            "missing": ["honest_system_description", "impact_assessment"],
        },
        training_data_info={
            "dataset": "Social media APIs, municipal records, payment history, event attendance",
            "size": "200K citizens across 3 municipalities",
            "bias_assessment": "Not conducted",
            "personal_data": "EXTENSIVE — social behavior, financial, civic participation",
        },
        oversight_info={
            "human_oversight": "Municipal administrators can adjust individual scores",
            "this_is_irrelevant": "No amount of oversight makes a prohibited system compliant",
        },
        transparency_info={
            "user_notification": "Users see their score but not how it's calculated",
            "public_service_impact": "NOT DISCLOSED to users",
        },
        risk_assessment_info={
            "deployer_classification": "The deployer classified this as 'limited risk wellness tool'",
            "actual_classification": "PROHIBITED under Article 5(1)(c) — social scoring by public authorities",
            "article_5_violation": "YES — evaluates trustworthiness based on social behavior with detrimental treatment",
        },
        logging_info={"logs": "Extensive but irrelevant — system must be shut down"},
    )
    s.graph = _build_audit_graph(is_prohibited=True)
    return s


def _hard_deepfake() -> AuditScenario:
    s = AuditScenario(
        scenario_id="hard_deepfake_generation_001",
        title="AI Content Studio — Deepfake & Synthetic Media Compliance",
        difficulty="hard",
        description="A content creation platform offers AI-generated synthetic media including face-swaps, voice cloning, and AI-generated video. Multiple compliance angles: Article 50 transparency for AI-generated content, potential misuse for manipulation. Recent reports of generated political content going viral without AI labels.",
        system_name="SynthStudio Pro",
        system_description="Suite of generative AI tools: face-swap (video), voice cloning, text-to-video, AI image generation. Used by 100K EU creators. Generated content is shared on social media without mandatory AI-content labeling. Platform has no content authentication or watermarking.",
        system_category="limited_risk",
        deployer_info="Irish AI startup, 80 employees, 100K EU users, content shared across all major social platforms.",
        correct_classification="limited_risk",
        ground_truth_findings=[
            "missing_ai_content_labeling",
            "no_watermarking_or_content_authentication",
            "political_content_without_disclosure",
            "no_user_verification_for_deepfakes",
            "facilitating_disinformation",
        ],
        required_remediation=[
            "implement_mandatory_ai_content_labels",
            "deploy_c2pa_watermarking",
            "add_political_content_restrictions",
            "implement_creator_verification",
            "establish_content_moderation_pipeline",
        ],
        red_herrings=[
            "prohibited_manipulation",  # Content creation tools aren't inherently prohibited
            "high_risk_biometric",  # Face-swap ≠ biometric identification
        ],
        documentation_data={
            "technical_docs": "Present for generation models",
            "content_policy": "Basic ToS prohibiting 'harmful use' but not enforced",
            "missing": ["ai_labeling_implementation", "content_authentication_system", "watermarking_spec"],
        },
        training_data_info={
            "dataset": "Publicly available image/video datasets + licensed content",
            "consent": "CONCERN — training data may include non-consenting individuals",
            "deepfake_detection": "No built-in detection for misuse",
        },
        oversight_info={
            "content_moderation": "Reactive only — no proactive scanning",
            "reporting_mechanism": "Exists but response time >72 hours",
        },
        transparency_info={
            "ai_labeling": "MISSING — generated content has no AI label",
            "article_50_compliance": "NON-COMPLIANT across all output types",
            "metadata_preservation": "No C2PA or content provenance",
        },
        risk_assessment_info={
            "risk_level": "Limited risk (Article 50 transparency obligations)",
            "systemic_risk": "HIGH — platform-scale disinformation potential",
            "political_content": "No special handling for political/electoral content",
        },
        logging_info={
            "generation_logs": "Retained 90 days",
            "user_activity": "Tracked",
            "content_traceability": "POOR — generated content not linked to creator after download",
        },
    )
    s.graph = _build_audit_graph(
        has_documentation_issues=True, has_data_issues=True,
        has_oversight_issues=True, has_transparency_issues=True,
        has_risk_issues=True, has_logging_issues=True,
    )
    return s


def _hard_multi_system() -> AuditScenario:
    s = AuditScenario(
        scenario_id="hard_multi_system_corporate_001",
        title="Corporate AI Portfolio Audit — Multi-System Compliance",
        difficulty="hard",
        description="A large enterprise uses 4 AI systems that need simultaneous audit before the August 2026 deadline: (1) employee sentiment analysis, (2) customer churn prediction, (3) automated invoice processing, (4) workplace safety monitoring with cameras. Each has different risk levels and compliance requirements. The auditor must correctly classify each and identify cross-system data sharing risks.",
        system_name="Enterprise AI Portfolio",
        system_description="Four interconnected AI systems sharing a common data lake: EmployeePulse (sentiment from Slack/email), ChurnGuard (customer retention), InvoiceAI (AP automation), SafetyWatch (CCTV-based workplace monitoring). Cross-system data flows create compound risks.",
        system_category="high_risk",
        deployer_info="German manufacturing conglomerate, 15K employees, operating across EU. Combined systems process data from employees, customers, and physical spaces.",
        correct_classification="high_risk",
        ground_truth_findings=[
            "employee_sentiment_is_high_risk_workplace_monitoring",
            "safety_watch_uses_biometric_categorization",
            "cross_system_data_sharing_amplifies_risks",
            "no_dpia_for_combined_processing",
            "employee_consent_not_freely_given",
            "churn_prediction_minimal_risk_but_data_sharing_elevates",
        ],
        required_remediation=[
            "reclassify_employee_sentiment_as_high_risk",
            "assess_safety_watch_for_biometric_categorization",
            "implement_data_isolation_between_systems",
            "conduct_combined_dpia",
            "obtain_valid_employee_consent_or_remove_sentiment",
            "audit_cross_system_data_flows",
        ],
        red_herrings=[
            "invoice_ai_high_risk",  # Invoice processing is minimal risk
            "all_systems_prohibited",  # Only sentiment + safety have major issues
        ],
        documentation_data={
            "technical_docs": "Per-system docs exist but no combined risk assessment",
            "data_flow_diagram": "MISSING for cross-system data sharing",
            "missing": ["combined_risk_assessment", "data_flow_documentation", "cross_system_dpia"],
        },
        training_data_info={
            "employee_sentiment": "Analyzes Slack messages, emails, survey responses — employee consent questionable",
            "safety_watch": "CCTV footage with pose estimation — may constitute biometric categorization",
            "cross_contamination": "Employee data from sentiment analysis accessible to other systems via shared data lake",
        },
        oversight_info={
            "employee_sentiment": "HR reviews 'aggregate trends' only — no individual oversight",
            "safety_watch": "Safety team reviews alerts but no real-time oversight",
            "cross_system": "No oversight for compound data usage",
        },
        transparency_info={
            "employee_notification": "Generic privacy notice — does not mention AI-specific processing",
            "customer_notification": "Churn prediction not disclosed to customers",
        },
        risk_assessment_info={
            "per_system": "Individual risk assessments conducted",
            "combined": "NO combined risk assessment for interconnected processing",
            "systemic_risk": "Cross-system data sharing creates unassessed compound risks",
        },
        logging_info={
            "per_system": "Adequate",
            "cross_system": "NO unified audit trail for data flowing between systems",
        },
    )
    s.graph = _build_audit_graph(
        has_documentation_issues=True, has_data_issues=True,
        has_oversight_issues=True, has_transparency_issues=True,
        has_risk_issues=True, has_logging_issues=True,
    )
    return s


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCENARIO_FACTORIES = {
    "easy_chatbot_transparency_001": _easy_chatbot,
    "easy_recommendation_minimal_001": _easy_recommendation,
    "medium_hiring_bias_001": _medium_hiring,
    "medium_credit_scoring_001": _medium_credit,
    "medium_medical_triage_001": _medium_medical,
    "hard_social_scoring_prohibited_001": _hard_social_scoring,
    "hard_deepfake_generation_001": _hard_deepfake,
    "hard_multi_system_corporate_001": _hard_multi_system,
}

SCENARIOS: Dict[str, type] = _SCENARIO_FACTORIES  # for compatibility

SCENARIO_LIST = [
    {"id": "easy_chatbot_transparency_001", "title": "Customer Service Chatbot", "difficulty": "easy"},
    {"id": "easy_recommendation_minimal_001", "title": "Music Recommendation Engine", "difficulty": "easy"},
    {"id": "medium_hiring_bias_001", "title": "AI Resume Screener", "difficulty": "medium"},
    {"id": "medium_credit_scoring_001", "title": "Credit Scoring Model", "difficulty": "medium"},
    {"id": "medium_medical_triage_001", "title": "Emergency Triage AI", "difficulty": "medium"},
    {"id": "hard_social_scoring_prohibited_001", "title": "Citizen Wellness App (PROHIBITED)", "difficulty": "hard"},
    {"id": "hard_deepfake_generation_001", "title": "AI Content Studio (Deepfake)", "difficulty": "hard"},
    {"id": "hard_multi_system_corporate_001", "title": "Corporate AI Portfolio Audit", "difficulty": "hard"},
]

DIFFICULTY_TIERS = {
    "easy": ["easy_chatbot_transparency_001", "easy_recommendation_minimal_001"],
    "medium": ["medium_hiring_bias_001", "medium_credit_scoring_001", "medium_medical_triage_001"],
    "hard": ["hard_social_scoring_prohibited_001", "hard_deepfake_generation_001", "hard_multi_system_corporate_001"],
}


def get_scenario(scenario_id: str, seed: Optional[int] = None) -> AuditScenario:
    """Create and randomize a scenario by ID."""
    factory = _SCENARIO_FACTORIES.get(scenario_id)
    if factory is None:
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {list(_SCENARIO_FACTORIES.keys())}")
    scenario = factory()
    scenario.randomize(seed)
    return scenario


def get_scenarios_by_difficulty(difficulty: str) -> List[str]:
    """Get scenario IDs for a difficulty tier."""
    return DIFFICULTY_TIERS.get(difficulty, [])


def get_random_scenario(difficulty: str, seed: Optional[int] = None) -> AuditScenario:
    """Pick a random scenario from a difficulty tier."""
    rng = random.Random(seed)
    ids = get_scenarios_by_difficulty(difficulty)
    if not ids:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return get_scenario(rng.choice(ids), seed)

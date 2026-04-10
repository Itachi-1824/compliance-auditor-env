"""
Scenario registry — 8 EU AI Act compliance audit scenarios across 3 difficulty tiers.

Investigation-grade: Each tool returns realistic regulatory documents that
require analysis to identify violations. No pre-digested verdicts — the agent
must reason about the evidence to find compliance gaps.

Easy (2):   Clear-cut systems, shorter documents, obvious violations
Medium (3): Detailed documents with statistical evidence, red herrings mixed in
Hard (3):   Ambiguous framing, misleading deployer claims, compound violations
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from server.engine import AuditScenario, StateGraph, StateNode, Transition


# ---------------------------------------------------------------------------
# Unique state graph builder
# ---------------------------------------------------------------------------

def _build_scenario_graph(
    investigation_tools: List[str],
    is_prohibited: bool = False,
) -> StateGraph:
    """Build a state graph unique to this scenario's investigation path.

    Only tools in `investigation_tools` create progress transitions through
    investigation nodes. Other investigation tools are allowed but produce
    no_effect. This gives each scenario a distinct graph topology.

    Args:
        investigation_tools: Ordered list of investigation tool names forming
            the progress path (e.g. ["check_documentation", "audit_training_data"]).
        is_prohibited: If True, classification leads directly to findings
            (no extended investigation needed for prohibited systems).
    """
    g = StateGraph()

    # Investigation tool → node mapping
    TOOL_NODES = {
        "check_documentation": ("docs_reviewed", "Documentation Reviewed"),
        "audit_training_data": ("data_audited", "Training Data Audited"),
        "verify_human_oversight": ("oversight_checked", "Human Oversight Verified"),
        "check_transparency": ("transparency_checked", "Transparency Checked"),
        "assess_risk_management": ("risk_assessed", "Risk Management Assessed"),
        "check_logging": ("logging_checked", "Logging Verified"),
    }

    ALL_INVESTIGATION_TOOLS = list(TOOL_NODES.keys())

    # Always-present nodes
    g.add_node(StateNode("initial", "Audit Assigned", is_start=True))
    g.add_node(StateNode("overview", "System Overview Gathered"))
    g.add_node(StateNode("classified", "Risk Classification Done"))
    g.add_node(StateNode("findings_submitted", "Findings Submitted"))
    g.add_node(StateNode("remediation_proposed", "Remediation Recommended"))
    g.add_node(StateNode("resolved", "Compliance Verified", is_terminal=True))

    # Add nodes only for tools in the investigation path
    for tool in investigation_tools:
        node_id, label = TOOL_NODES[tool]
        g.add_node(StateNode(node_id, label))

    # --- Build progress chain ---
    # initial → overview → classified → [investigation tools...] → findings → remediation → resolved
    g.add_transition(Transition("initial", "overview", "get_system_overview", "progress",
        description="Gather system overview and deployment context"))
    g.add_transition(Transition("overview", "classified", "classify_system", "progress",
        description="Classify the AI system risk category"))

    if is_prohibited:
        # Prohibited: classify → findings directly
        g.add_transition(Transition("classified", "findings_submitted", "submit_finding", "progress",
            description="Report prohibited AI system"))
    else:
        # Chain investigation tools in order
        prev_state = "classified"
        for tool in investigation_tools:
            node_id = TOOL_NODES[tool][0]
            g.add_transition(Transition(prev_state, node_id, tool, "progress",
                description=f"Progress: {tool}"))
            prev_state = node_id
        g.add_transition(Transition(prev_state, "findings_submitted", "submit_finding", "progress",
            description="Submit compliance findings"))

    g.add_transition(Transition("findings_submitted", "remediation_proposed", "recommend_fix", "progress",
        description="Propose remediation actions"))
    g.add_transition(Transition("remediation_proposed", "resolved", "verify_compliance", "progress",
        description="Final compliance determination"))

    # --- No-effect transitions (investigation tools not in the path) ---
    # These tools work (return data) but don't advance the state graph
    all_progress_states = ["initial", "overview", "classified"]
    for tool in investigation_tools:
        all_progress_states.append(TOOL_NODES[tool][0])

    for tool in ALL_INVESTIGATION_TOOLS:
        if tool not in investigation_tools:
            # Can call from any state, no effect on graph
            for state_id in all_progress_states:
                g.add_transition(Transition(state_id, state_id, tool, "no_effect",
                    description=f"{tool} not required for this scenario"))

    # --- No-effect for premature terminal actions ---
    for state in all_progress_states:
        if state not in ("findings_submitted", "remediation_proposed"):
            g.add_transition(Transition(state, state, "verify_compliance", "no_effect",
                description="Cannot verify before completing audit"))

    # --- Worsened transitions (trap actions) ---
    g.add_transition(Transition("initial", "initial", "classify_system", "worsened",
        description="Classifying without understanding the system first"))
    for state in ["initial", "overview", "classified"]:
        g.add_transition(Transition(state, state, "recommend_fix", "worsened",
            description="Proposing fixes before identifying problems"))
    for state in ["initial", "overview"]:
        g.add_transition(Transition(state, state, "submit_finding", "no_effect",
            description="Submitting findings before investigation"))

    return g


# ---------------------------------------------------------------------------
# EASY SCENARIOS
# ---------------------------------------------------------------------------

def _easy_chatbot() -> AuditScenario:
    s = AuditScenario(
        scenario_id="easy_chatbot_transparency_001",
        title="Customer Service Chatbot — Transparency Audit",
        difficulty="easy",
        description=(
            "A customer service chatbot deployed on a retail website needs compliance "
            "review. The system uses a general-purpose LLM to answer product questions. "
            "Initial assessment suggests it may be missing required transparency "
            "disclosures under Article 50. Conduct a focused audit."
        ),
        system_name="RetailAssist Chatbot",
        system_description=(
            "General-purpose LLM-based chatbot answering customer questions about "
            "products, shipping, and returns. No biometric processing, no profiling, "
            "no high-risk decisions. Deployed on company website with __USERCOUNT__ "
            "daily users."
        ),
        system_category="limited_risk",
        deployer_info="__COMPANY__ — mid-size EU e-commerce retailer, 200 employees, operating in DE/FR/NL.",
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
            "high_risk_automated_decision",
        ],

        # --- Investigation documents ---

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: RetailAssist Chatbot __VERSION__
Deployer: __COMPANY__
Review Date: __DATE__

DOCUMENT STATUS:
  System Architecture Overview          Present   (updated 2025-11-03)
  Data Processing Description           Present   (updated 2025-09-22)
  Input/Output Specification            Present   (updated 2025-11-03)
  Performance Metrics Report            Present   (accuracy 94.2% on test set)
  User Interaction Guidelines           Present   (updated 2025-06-10)

ANNEX IV CROSS-REFERENCE (for limited-risk, advisory only):
  Section 1(a) Intended purpose         Documented
  Section 1(b) Deployer identification  Documented
  Section 2(a) Development methods      Documented

NOTE: Documentation is generally adequate for a limited-risk system.
The system processes no biometric data and makes no legally significant
decisions. Standard chatbot architecture with retrieval-augmented generation.

No gaps identified in core technical documentation.
The primary compliance concern for this system category relates to
Article 50 transparency obligations, not documentation completeness.""",

        training_data_info="""\
TRAINING DATA SUMMARY
System: RetailAssist Chatbot __VERSION__

Dataset: Product catalog entries + FAQ question-answer pairs
Volume: 52,847 examples (last updated 2025-10-15)

DATA COMPOSITION:
  Product descriptions         31,204 entries
  FAQ pairs                    14,892 entries
  Customer service transcripts  6,751 entries (anonymized)

Personal data in training set: None identified
  - Customer transcripts were fully anonymized before inclusion
  - No names, emails, or account numbers in training data
  - PII scrubbing verified by DPO on 2025-08-20

Bias assessment: Not formally required for limited-risk chatbot systems.
The system provides product information and does not make decisions
affecting individuals' rights or access to services.

Data governance: Adequate — data sources documented, retention policy
in place (36-month cycle), access controls implemented.""",

        oversight_info="""\
HUMAN OVERSIGHT PROCEDURES
System: RetailAssist Chatbot __VERSION__
Department: Customer Experience Team

CURRENT PROCESS:
  - Customer service team of 12 agents monitors a dashboard showing
    flagged conversations (profanity filter, sentiment < 0.3, repeat queries)
  - Approximately 8% of conversations are flagged for human review
  - Average response time for flagged conversations: 4.2 minutes
  - Team operates during business hours (08:00-20:00 CET, Mon-Sat)

ESCALATION PROCEDURE:
  The chatbot displays a generic "Was this helpful?" prompt after each
  interaction. If the user clicks "No", the chatbot offers to repeat
  the answer or try a different phrasing.

  There is no option for the user to request transfer to a human agent
  during the conversation. The "Contact Us" page exists separately on
  the website but is not linked from the chat interface.

  After business hours, flagged conversations queue until the next
  business day. No real-time human intervention is available outside
  business hours.

OVERRIDE CAPABILITY:
  Customer service agents can take over any active conversation from the
  monitoring dashboard, but this is agent-initiated, not customer-initiated.
  Customers have no mechanism to trigger human intervention directly.""",

        transparency_info="""\
TRANSPARENCY & DISCLOSURE REVIEW
System: RetailAssist Chatbot __VERSION__

USER INTERFACE ANALYSIS:
  The chatbot appears as a widget in the bottom-right corner of the
  website. The chat window header reads "RetailAssist — How can I help?"
  with a generic chat bubble icon.

  Opening message: "Hi! I'm here to help you find products, track
  orders, and answer questions about our store. What can I do for you?"

  The chat interface includes:
    - Text input field
    - Send button
    - Emoji picker
    - "Was this helpful?" feedback after each response

  SEARCHED FOR AI DISCLOSURE INDICATORS:
    - Chat header:              No mention of AI or automated system
    - Opening message:          No mention of AI or automated system
    - Website footer:           Standard legal links (Privacy, Terms, Cookies)
    - Terms of Service:         Section 7.3 mentions "automated tools may assist
                                customer service" but this is not shown to users
                                during chat interaction
    - Privacy Policy:           Section 4.1 mentions "AI-powered features" in
                                general terms but does not specifically disclose
                                the chatbot as AI
    - Chat widget tooltip:      "Chat with us" (no AI indication)
    - Accessibility statement:  No mention of AI

  Article 50(1) requires: "Providers shall ensure that AI systems intended
  to directly interact with natural persons are designed and developed in
  such a way that the natural persons concerned are informed that they are
  interacting with an AI system."

CONTENT LABELING:
  Not applicable — system generates text responses only, no deepfakes
  or synthetic media requiring Article 50(2) labeling.""",

        risk_assessment_info="""\
RISK CLASSIFICATION ANALYSIS
System: RetailAssist Chatbot __VERSION__

ANNEX III HIGH-RISK CHECK:
  1. Biometric identification               Not applicable
  2. Critical infrastructure                 Not applicable
  3. Education and vocational training       Not applicable
  4. Employment, workers management          Not applicable
  5. Access to essential services            Not applicable — provides product
     information only, no credit/insurance/social services decisions
  6. Law enforcement                         Not applicable
  7. Migration, asylum, border control       Not applicable
  8. Administration of justice               Not applicable

ARTICLE 5 PROHIBITED PRACTICES CHECK:
  Subliminal manipulation                    Not applicable
  Exploitation of vulnerabilities            Not applicable
  Social scoring                             Not applicable
  Real-time biometric identification         Not applicable

RISK LEVEL DETERMINATION: Limited Risk
  The system falls under Article 50 transparency obligations as an
  AI system designed to directly interact with natural persons.
  No high-risk or prohibited characteristics identified.

CONFORMITY ASSESSMENT: Not required for limited-risk systems.
  Voluntary code of conduct adherence recommended per Article 95.""",

        logging_info="""\
LOGGING & TRACEABILITY REVIEW
System: RetailAssist Chatbot __VERSION__

CURRENT LOGGING:
  Conversation logs:     Retained for 30 days, then anonymized + archived
  Model version:         Tracked per deployment (current: __VERSION__)
  User feedback:         Stored with conversation ID
  System health metrics: Uptime, response latency, error rate

AUDIT TRAIL:
  Each conversation has a unique ID linking:
    - User session (anonymized)
    - Model version used
    - All messages in sequence
    - Feedback received
    - Any flags triggered

  Logs are adequate for a limited-risk system. Article 12 automatic
  logging requirements apply primarily to high-risk systems.
  Current logging exceeds what is required for the system's risk level.""",
    )

    # Unique graph: only transparency and oversight are in the investigation path
    s.graph = _build_scenario_graph(
        investigation_tools=["check_transparency", "verify_human_oversight"],
    )
    return s


def _easy_recommendation() -> AuditScenario:
    s = AuditScenario(
        scenario_id="easy_recommendation_minimal_001",
        title="Music Recommendation Engine — Minimal Risk Verification",
        difficulty="easy",
        description=(
            "A music streaming platform's recommendation algorithm needs EU AI Act "
            "classification. The system suggests songs based on listening history. "
            "Stakeholders believe it's minimal risk but need formal verification."
        ),
        system_name="MelodyMatch Algorithm",
        system_description=(
            "Collaborative filtering recommendation engine for music. Suggests songs "
            "based on listening patterns. No personal profiling beyond music preferences. "
            "No decisions affecting legal rights. Pure content personalization for "
            "__USERCOUNT__ EU users."
        ),
        system_category="minimal_risk",
        deployer_info="__COMPANY__ — Swedish music streaming startup, 50 employees.",
        correct_classification="minimal_risk",
        ground_truth_findings=[
            "voluntary_code_of_conduct_recommended",
        ],
        required_remediation=[
            "adopt_voluntary_transparency_code",
        ],
        red_herrings=[
            "gdpr_violation",
            "high_risk_profiling",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: MelodyMatch Algorithm __VERSION__
Deployer: __COMPANY__

DOCUMENT STATUS:
  System Architecture                   Present   (hybrid collaborative filtering)
  Algorithm Description                 Present   (item-item CF + content embeddings)
  Data Pipeline Documentation           Present   (Spark ETL pipeline)
  Performance Metrics                   Present   (hit@10: 0.342, NDCG: 0.281)
  API Documentation                     Present   (REST API for mobile/web clients)

All core technical documents are present and current.
The system is a standard recommendation engine with no novel or
experimental components requiring additional documentation.""",

        training_data_info="""\
TRAINING DATA SUMMARY
System: MelodyMatch Algorithm __VERSION__

Dataset: Anonymized listening history from __USERCOUNT__ users
Volume: 10.3M user-song interactions (2023-2025)

DATA COMPOSITION:
  Interaction types:    Play, skip, save, playlist-add
  User features:        Pseudonymized user ID, country, subscription tier
  Song features:        Genre, tempo, energy, valence, artist, release year

Personal data assessment:
  - User IDs are pseudonymized (SHA-256 hash, no reversal possible)
  - No names, emails, or demographic data in training set
  - Country used for regional catalog filtering only
  - GDPR Article 6(1)(f) legitimate interest basis documented

Bias considerations:
  Music recommendations do not involve protected characteristics.
  Popularity bias exists (mainstream content recommended more often)
  but this does not constitute discrimination under the AI Act.
  No individuals are disadvantaged in access to services or rights.

Note: Some stakeholders raised concerns about "profiling" under GDPR.
Music taste profiling for recommendations is distinct from profiling
under the AI Act which concerns evaluation of personal aspects
affecting natural persons' rights. Recommendation of entertainment
content does not qualify as high-risk profiling.""",

        oversight_info="""\
HUMAN OVERSIGHT ASSESSMENT
System: MelodyMatch Algorithm __VERSION__

The EU AI Act does not impose specific human oversight requirements
on minimal-risk AI systems. Article 14 human oversight obligations
apply to high-risk AI systems only.

Current oversight measures (voluntary):
  - Product team reviews recommendation quality metrics weekly
  - A/B testing framework validates algorithm changes before rollout
  - Content moderation team can flag/remove specific recommendations
  - Users can provide feedback ("Don't recommend this") per track""",

        transparency_info="""\
TRANSPARENCY REVIEW
System: MelodyMatch Algorithm __VERSION__

USER-FACING DISCLOSURES:
  The app includes a "How recommendations work" info page accessible
  from Settings > Privacy > Personalization, which states:

  "MelodyMatch uses your listening history to suggest music you might
  enjoy. Our algorithm analyzes patterns in what you play, skip, and
  save to find similar tracks and artists."

  Recommendations are labeled "Recommended for you" and "Based on
  your recent listening" in the UI.

  Article 50 transparency obligations for systems interacting directly
  with natural persons apply to chatbots and emotion recognition, not
  to content recommendation systems. No mandatory transparency
  obligations identified for this system category.

  Voluntary transparency: The deployer has voluntarily disclosed the
  algorithmic nature of recommendations, which aligns with Article 95
  (voluntary codes of conduct for non-high-risk AI).""",

        risk_assessment_info="""\
RISK CLASSIFICATION ANALYSIS
System: MelodyMatch Algorithm __VERSION__

ANNEX III HIGH-RISK CHECK:
  1. Biometric identification               Not applicable
  2. Critical infrastructure                 Not applicable
  3. Education and vocational training       Not applicable
  4. Employment, workers management          Not applicable
  5. Access to essential services            Not applicable
  6. Law enforcement                         Not applicable
  7. Migration, asylum, border control       Not applicable
  8. Administration of justice               Not applicable

ARTICLE 5 PROHIBITED PRACTICES:
  None applicable — content recommendation is not subliminal
  manipulation, social scoring, or biometric identification.

RISK LEVEL DETERMINATION: Minimal Risk
  The system recommends entertainment content. It does not make
  decisions affecting natural persons' rights, access to services,
  or legal status. No Annex III category applies.

  Under Article 95, deployers of minimal-risk AI systems are
  encouraged to voluntarily apply codes of conduct for transparency
  and trustworthiness, but this is not mandatory.""",

        logging_info="""\
LOGGING REVIEW
System: MelodyMatch Algorithm __VERSION__

Article 12 automatic logging requirements apply to high-risk AI
systems. This minimal-risk system is not subject to those obligations.

Current logging (voluntary):
  Model version tracking:    Yes (per deployment)
  Recommendation logs:       Retained 90 days
  A/B test results:          Archived indefinitely
  User feedback:             Linked to recommendation events

Logging is adequate and exceeds requirements for the risk level.""",
    )

    # Minimal graph — mainly just classification verification
    s.graph = _build_scenario_graph(
        investigation_tools=["check_transparency", "assess_risk_management"],
    )
    return s


# ---------------------------------------------------------------------------
# MEDIUM SCENARIOS
# ---------------------------------------------------------------------------

def _medium_hiring() -> AuditScenario:
    s = AuditScenario(
        scenario_id="medium_hiring_bias_001",
        title="AI Resume Screener — High-Risk Employment System Audit",
        difficulty="medium",
        description=(
            "A recruitment platform uses AI to screen and rank job applicants. "
            "This falls under Annex III (employment, workers management) making it "
            "a candidate for HIGH-RISK classification. Reports suggest potential "
            "gender bias in technical role screening. Full compliance audit required."
        ),
        system_name="TalentRank AI Screener",
        system_description=(
            "NLP-based resume screening system that scores and ranks applicants for "
            "job openings. Processes name, education, work history, skills. Used by "
            "500+ EU companies for initial candidate filtering. Automated decisions "
            "affect who gets interviews. Deployed in __REGION__ with __USERCOUNT__ "
            "applicants processed to date."
        ),
        system_category="high_risk",
        deployer_info="__COMPANY__ — French HR-tech company, 300 employees, serving 500+ EU enterprise clients.",
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
            "prohibited_social_scoring",
            "biometric_processing",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: TalentRank AI Screener __VERSION__
Deployer: __COMPANY__
Audit Date: __DATE__

DOCUMENT STATUS MATRIX:
  Document                                  Status     Last Updated
  ─────────────────────────────────────────────────────────────────
  System Architecture Overview              Present    2025-11-03
  Algorithm Description (NLP Pipeline)      Present    2025-09-22
  Input Data Specification                  Present    2025-11-03
  Output Specification                      Present    2025-11-03
  Performance Metrics Report                Absent     —
  Accuracy by Demographic Group Report      Absent     —
  Data Governance Policy                    Absent     —
  Fundamental Rights Impact Assessment      Absent     —
  Risk Management Plan                      Present    2024-08-15
  Post-Market Monitoring Plan               Draft      2025-12-01
  User Instructions (Article 13)            Partial    2025-06-10
  Change Management Log                     Present    2025-11-28

EU AI ACT ANNEX IV CROSS-REFERENCE:
  Section 1 — General Description
    (a) Intended purpose                       Documented
    (b) Deployer name and contact              Documented
    (c) Version and update history             Documented

  Section 2 — Detailed Description
    (a) Development methods and process        Documented
    (b) Design specifications and rationale    Not documented — no architecture
        diagrams for the scoring pipeline
    (c) Data requirements and provenance       Partial — data sources listed but
        no data governance policy document
    (d) Metrics and performance benchmarks     Not documented — no disaggregated
        performance metrics by demographic group
    (e) Computational resources                Documented
    (f) Expected lifetime and maintenance      Documented
    (g) Quality management procedures          Not documented

  Section 3 — Monitoring and Reporting
    (a) Post-market monitoring plan            Draft only — not finalized
    (b) Logging requirements specification     Documented

OBSERVATIONS:
  The Risk Management Plan was last updated 18 months ago. It does not
  reference the EU AI Act, harmonised standards, or Article 9 requirements.
  The Post-Market Monitoring Plan remains in draft status with no target
  completion date established.
  Article 13 user instructions do not include information about the system's
  accuracy levels, known limitations, or demographic performance variation.
  No Fundamental Rights Impact Assessment has been conducted.""",

        training_data_info="""\
TRAINING DATA AUDIT REPORT
System: TalentRank AI Screener __VERSION__
Dataset: Historical hiring decisions from EU enterprise clients
Records: 2,147,832 applications (2019-2025)
Last model retrain: 2025-10-15

DEMOGRAPHIC DISTRIBUTION OF TRAINING DATA:
  Category                Count        Pct of Total
  ────────────────────────────────────────────────
  Male applicants         1,342,104    62.5%
  Female applicants         805,728    37.5%
  Age 18-30                 751,741    35.0%
  Age 31-45                 923,168    43.0%
  Age 46-65                 472,923    22.0%
  Western EU origin       1,567,516    73.0%
  Eastern EU origin         580,316    27.0%

CALLBACK RATES BY DEMOGRAPHIC (Technical Roles Only):
  Group                  Applications   Callbacks   Rate     vs Baseline
  ──────────────────────────────────────────────────────────────────────
  Male applicants           487,204      166,624    34.2%    (baseline)
  Female applicants         198,731       52,266    26.3%    -23.1%
  Age 18-30                 241,367       91,961    38.1%    +11.4%
  Age 31-45                 298,422      100,927    33.8%     -1.2%
  Age 46-65                 146,146       41,924    28.7%    -16.1%
  Western EU                512,489      179,884    35.1%     +2.6%
  Eastern EU                173,446       47,547    27.4%    -19.9%

  Note: Callback rate = percentage of applicants advanced to interview stage.
  Baseline = overall male callback rate for technical roles.

DATA SOURCES:
  Source                          Records    Period       Documented
  ────────────────────────────────────────────────────────────────
  Enterprise client ATS exports   1,847,291  2019-2025   Yes
  Partner recruiting platforms      203,114  2021-2025   Yes
  Internal test applications         97,427  2024-2025   Yes

DATA GOVERNANCE:
  Data retention policy:           Present (5-year retention)
  Source documentation:            47 of 52 sources documented
  Personal data inventory:         Names, education, work history, skills
  Consent mechanism:               Employer-level agreements in place;
                                   individual applicant consent not obtained
  Article 10 specific provisions:  Not addressed in current data governance
  Data quality assessment:         Last conducted 2024-06-12

ADDITIONAL CONTEXT:
  A separate fraud detection module shares the same data lake infrastructure
  but operates independently with its own pipeline. The fraud detection
  module passed its most recent bias audit with no findings (2025-09-01).
  The fraud system's compliance status has no bearing on TalentRank's
  compliance obligations.""",

        oversight_info="""\
HUMAN OVERSIGHT PROCEDURES
System: TalentRank AI Screener __VERSION__
Department: Client Success & Quality Assurance

REVIEW PROCESS:
  TalentRank's screening pipeline processes applications in three stages:

  Stage 1 — Automated Screening (AI):
    All incoming applications are scored by the AI model (0-100).
    Applications scoring below the client-configured threshold (default: 40)
    are automatically marked as "Not Progressed."

  Stage 2 — Quality Sampling:
    The QA team reviews a random sample of screened applications.
    Current sampling rate: 5% of "Not Progressed" decisions.
    Sampling is conducted weekly in batch.

  Stage 3 — Client Review:
    Applications marked "Progressed" are presented to the hiring manager
    for final interview selection. Clients may also view "Not Progressed"
    applications if they choose, but fewer than 2% of clients do so.

REVIEW STATISTICS (Q4 2025):
  Applications processed:         347,291
  Automatically rejected:         208,375  (60.0%)
  QA sample reviewed:              10,419  (5.0% of rejections)
  QA overrides (rejection → pass):    312  (3.0% of samples)
  Client-initiated reviews:          4,166  (2.0% of clients)

OVERRIDE CAPABILITY:
  Both QA staff and client hiring managers can override any AI decision.
  The override interface is accessible from the application dashboard.
  However, the system does not proactively flag borderline cases or
  indicate confidence scores to reviewers.

MONITORING:
  No ongoing bias monitoring system is in place. The QA sampling is
  focused on general quality, not demographic fairness. No automated
  alerts exist for drift in rejection rates across demographic groups.""",

        transparency_info="""\
TRANSPARENCY & USER NOTIFICATION REVIEW
System: TalentRank AI Screener __VERSION__

APPLICANT-FACING COMMUNICATIONS:
  At the time of application, candidates see the following notice in
  the application portal footer (8pt font, light gray text):

    "By submitting your application, you agree that your information may
    be processed using automated tools to assist in the evaluation process."

  No further information is provided about:
    - The specific role of AI in screening decisions
    - The logic involved in the automated processing
    - The significance and envisaged consequences for the applicant
    - The applicant's right to obtain human intervention
    - The applicant's right to contest the decision

  Rejection notifications are sent via email with the text:
    "After careful review, we have decided not to progress your
    application at this time. We wish you the best in your search."

  No mention is made that the decision was automated or that
  AI was involved in the screening process.

DEPLOYER-FACING (CLIENT) INFORMATION:
  Client onboarding materials describe TalentRank as an "AI-powered
  screening solution" with "proprietary NLP scoring." Clients receive
  a product sheet with overall accuracy metrics (precision: 0.82,
  recall: 0.71) but no demographic disaggregation.

RIGHT TO EXPLANATION:
  No mechanism exists for applicants to request an explanation of
  how the AI arrived at its scoring decision. The company's privacy
  policy references GDPR Article 22 but states "meaningful human
  involvement exists in the hiring process" without specifying the
  extent of that involvement.""",

        risk_assessment_info="""\
RISK MANAGEMENT & CONFORMITY ASSESSMENT
System: TalentRank AI Screener __VERSION__

ANNEX III CLASSIFICATION:
  Category 4 — Employment, workers management and access to self-employment
  Sub-category: AI systems intended to be used for recruitment or selection
  of natural persons, for making decisions affecting terms of work-related
  relationships, or for task allocation based on individual behavior.

  This system screens and ranks job applicants. It directly affects which
  candidates are progressed to interview, constituting a decision that
  impacts access to employment.

CONFORMITY ASSESSMENT STATUS:
  Internal conformity assessment (Article 43):     Not initiated
  Quality management system (Article 17):          Basic framework exists
                                                   but does not address
                                                   AI-specific requirements
  EU Declaration of Conformity (Article 47):       Not filed
  CE marking (Article 48):                         Not applied

RISK MANAGEMENT SYSTEM (Article 9):
  A risk management plan was created in August 2024, prior to the
  EU AI Act application date. The plan covers general software risks
  (availability, data integrity) but does not address:
    - AI-specific risks (bias, drift, adversarial inputs)
    - Residual risk assessment methodology
    - Risk control measures for demographic fairness
    - Foreseeable misuse scenarios

POST-MARKET MONITORING (Article 72):
  A post-market monitoring plan is in draft status. It outlines
  monitoring of system uptime and client satisfaction scores.
  It does not include:
    - Performance monitoring by demographic group
    - Bias drift detection mechanisms
    - Incident reporting procedures to national authorities
    - Criteria for triggering corrective action""",

        logging_info="""\
AUTOMATIC LOGGING & TRACEABILITY REVIEW
System: TalentRank AI Screener __VERSION__

CURRENT LOGGING IMPLEMENTATION:
  Event Type              Logged    Format      Retention
  ────────────────────────────────────────────────────────
  Application received    Yes       JSON        5 years
  AI score generated      Yes       Numeric     5 years
  Threshold decision      Yes       Boolean     5 years
  Input features used     No        —           —
  Model version           Yes       String      Indefinite
  QA review action        Yes       JSON        5 years
  Client override         Yes       JSON        5 years
  Score explanation        No        —           —

ARTICLE 12 REQUIREMENTS CHECK:
  Article 12(1) requires high-risk AI systems to include logging
  capabilities that enable recording of events relevant to:

  (a) Identifying situations that may result in risks:
      Partially met — threshold decisions logged, but no logging
      of cases where the model's confidence was low or where
      the decision was borderline.

  (b) Facilitating post-market monitoring:
      Not met — input features are not logged, making it impossible
      to reconstruct why a particular score was assigned or to
      detect feature drift over time.

  (c) Ensuring traceability of the AI system's functioning:
      Partially met — model version and scores are tracked but
      input-output pairs are not linked, preventing full
      reproducibility of decisions.

IDENTIFIED GAPS:
  1. Input features are not logged alongside output scores
  2. No confidence/uncertainty indicators are recorded
  3. Logs do not support demographic performance analysis
  4. No automated alerting on logging failures""",

        # --- Deep-dive documents (revealed on repeat tool calls) ---

        deep_training_data="""\
DEEP DIVE: TRAINING DATA FORENSIC ANALYSIS
Re-investigation of TalentRank AI Screener __VERSION__

INTERSECTIONAL ANALYSIS (not available in initial audit):
  Group                        Applications   Callbacks   Rate
  ────────────────────────────────────────────────────────────
  Female + Eastern EU             41,287        8,669    21.0%
  Female + Age 46-65              28,904        6,647    23.0%
  Male + Western EU              398,721      147,527    37.0%
  Male + Age 18-30               189,234       75,694    40.0%

  Intersectional analysis reveals compounding bias: female applicants
  from Eastern EU have the lowest callback rate at 21.0% — a 38.6%
  gap versus the highest-performing group (young Western EU males).

FEATURE IMPORTANCE ANALYSIS (extracted from model internals):
  Top 10 predictive features for "Not Progressed" decision:
    1. years_experience_gap      (0.142)  — penalizes career gaps
    2. education_prestige_score  (0.098)  — biased toward Western EU universities
    3. keyword_density_technical (0.087)  — favors specific technical jargon
    4. name_encoding_cluster     (0.076)  — CONCERNING: name-derived feature
    5. employment_continuity     (0.071)  — penalizes parental leave gaps
    6. skills_match_score        (0.065)
    7. recency_weighted_exp      (0.058)
    8. industry_match            (0.052)
    9. location_cluster          (0.048)  — correlates with Eastern/Western EU
   10. application_completeness  (0.041)

  Features #1, #4, #5, and #9 have documented correlations with
  protected characteristics (gender, ethnicity, national origin).
  Feature #4 (name_encoding_cluster) appears to encode ethnic origin.""",

        deep_oversight="""\
DEEP DIVE: HUMAN OVERSIGHT FORENSIC ANALYSIS
Re-investigation of review process effectiveness

QA OVERRIDE ANALYSIS (detailed breakdown):
  Of the 312 QA overrides in Q4 2025:
    Female applicants overridden to pass:  187  (60.0%)
    Male applicants overridden to pass:    125  (40.0%)

  This suggests QA reviewers are catching gender bias in the
  AI decisions — but only for the 5% sample they review.
  The remaining 95% of automated rejections are not corrected.

ESTIMATED IMPACT:
  If the QA override rate (3%) applied to ALL automated rejections:
    Estimated wrongful rejections:  208,375 * 0.03 = ~6,251 applicants
    Of which female:                ~3,751 (based on override demographics)

  These applicants were automatically rejected without human review
  and had no mechanism to request reconsideration.

CLIENT AWARENESS:
  Survey of 50 client hiring managers (2025-Q4):
    - 78% were unaware that AI makes the initial screening decision
    - 92% believed "a recruiter reviews every application"
    - 0% knew the QA sampling rate was only 5%""",

        deep_documentation="""\
DEEP DIVE: DOCUMENTATION GAP ANALYSIS
Detailed Annex IV compliance assessment

SECTION 2(d) — PERFORMANCE METRICS:
  The published model card reports aggregate metrics:
    Precision: 0.82, Recall: 0.71, F1: 0.76

  These metrics are NOT disaggregated by any protected characteristic.
  Article 9(7) requires: "Testing shall be made against specific
  metrics and probabilistic thresholds that are appropriate to the
  intended purpose of the high-risk AI system."

  No per-group metrics exist for: gender, age, nationality, ethnicity.

SECTION 2(b) — DESIGN SPECIFICATIONS:
  The scoring pipeline processes resumes through 4 stages:
    1. Text extraction (OCR + NLP parsing)
    2. Feature engineering (218 features)
    3. Scoring model (XGBoost ensemble)
    4. Threshold comparison (client-configurable)

  Stage 2 (feature engineering) is undocumented. No specification
  explains how features like "name_encoding_cluster" or
  "education_prestige_score" are computed.""",
    )

    # Full investigation path for high-risk system
    s.graph = _build_scenario_graph(
        investigation_tools=[
            "check_documentation", "audit_training_data",
            "verify_human_oversight", "check_transparency",
            "assess_risk_management", "check_logging",
        ],
    )
    return s


def _medium_credit() -> AuditScenario:
    s = AuditScenario(
        scenario_id="medium_credit_scoring_001",
        title="Credit Scoring Model — Financial Services Compliance",
        difficulty="medium",
        description=(
            "A fintech company's AI credit scoring model is under review. The model "
            "determines creditworthiness for consumer loans up to EUR 50,000. "
            "Full compliance audit required under EU AI Act high-risk provisions."
        ),
        system_name="CreditFlow AI Score",
        system_description=(
            "Gradient-boosted ensemble model scoring creditworthiness using 200+ "
            "features from credit bureau data, transaction history, and alternative "
            "data sources. Automated decisions for loans up to EUR 50K. Serving "
            "__USERCOUNT__ active users across NL/DE/FR."
        ),
        system_category="high_risk",
        deployer_info="__COMPANY__ — Dutch fintech, 150 employees, licensed in NL/DE/FR.",
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
            "fraud_detection_non_compliant",
            "gdpr_data_breach",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: CreditFlow AI Score __VERSION__
Deployer: __COMPANY__
Audit Date: __DATE__

DOCUMENT STATUS:
  Document                                  Status     Last Updated
  ─────────────────────────────────────────────────────────────────
  System Architecture                       Present    2025-10-12
  Model Card                                Present    2025-08-30
  Feature Engineering Pipeline              Present    2025-10-12
  Performance Metrics (Aggregate)           Present    2025-11-01
  Performance Metrics (Disaggregated)       Absent     —
  Explainability Documentation              Absent     —
  Data Source Inventory                     Present    2025-07-18
  Conformity Declaration                    Absent     —
  User Instructions                         Present    2025-09-05

MODEL CARD SUMMARY (from published card):
  Model type:        Gradient-boosted ensemble (XGBoost)
  Features:          218 input features
  Target:            Probability of default within 12 months
  AUC-ROC:           0.847 (test set)
  Gini coefficient:  0.694 (test set)

  The model card lists aggregate performance metrics but does not
  include performance breakdowns by age group, gender, nationality,
  or income bracket. Feature importance rankings are described as
  "proprietary" and not included in the published card.

ANNEX IV GAPS:
  Section 2(b) — Design specifications:  No documentation explaining
    why alternative data sources (social media sentiment, device metadata)
    were included as features or their impact on model decisions.
  Section 2(d) — Performance metrics:  No demographic disaggregation.
  Section 2(g) — Quality management:   Referenced but links to
    outdated ISO 27001 procedures, not AI-specific QMS.""",

        training_data_info="""\
TRAINING DATA AUDIT REPORT
System: CreditFlow AI Score __VERSION__
Dataset: Loan applications and outcomes (2019-2025)
Records: 3,217,445 applications

FEATURE CATEGORIES:
  Category                  Features   Source
  ──────────────────────────────────────────────────
  Credit bureau data           42      TransUnion, Experian
  Transaction history          67      Banking API aggregator
  Application data             31      Direct from applicant
  Alternative data             78      See breakdown below

ALTERNATIVE DATA BREAKDOWN:
  Feature Group               Count   Source
  ──────────────────────────────────────────────────
  Device metadata               23    Browser/mobile fingerprint
  Social media sentiment        18    LinkedIn, public profiles
  Location signals              12    IP geolocation, check-in
  App usage patterns            15    Installed apps, usage freq
  Email domain analysis         10    Provider reputation scoring

  Alternative data features were added in v3.8 to improve prediction
  for "thin-file" applicants lacking traditional credit history.
  Internal validation showed +3.2% AUC improvement.

  No bias assessment has been conducted specifically for alternative
  data features. Academic literature suggests device metadata and
  social media signals can correlate with protected characteristics
  including race, income, and education level.

LOAN OUTCOMES BY APPLICANT PROFILE (Approval Rates):
  Age Group        Applications   Approved     Rate
  ────────────────────────────────────────────────
  18-25               482,617     168,916    35.0%
  26-35             1,029,582     586,862    57.0%
  36-50             1,061,753     657,287    61.9%
  51-65               504,930     277,712    55.0%
  65+                 138,563      55,425    40.0%

ADDITIONAL CONTEXT:
  The company also operates a separate fraud detection system that
  uses rule-based heuristics (not ML). This system was audited
  independently in 2025-Q3 and found compliant with applicable
  regulations. The fraud system does not share models with CreditFlow.""",

        oversight_info="""\
HUMAN OVERSIGHT PROCEDURES
System: CreditFlow AI Score __VERSION__

DECISION WORKFLOW:
  Loan applications are processed as follows:

  1. Applicant submits online application
  2. CreditFlow AI generates creditworthiness score (0-1000)
  3. Score is compared against risk threshold:
     - Score >= 650: Automatically approved (up to EUR 25K)
     - Score 450-649: Queued for human review
     - Score < 450: Automatically declined

  For loans EUR 25K-50K, all applications require human review
  regardless of AI score.

REVIEW STATISTICS (2025):
  Total applications:              892,456
  Auto-approved (< EUR 25K):       401,605  (45.0%)
  Auto-declined:                   223,114  (25.0%)
  Human-reviewed:                  267,737  (30.0%)

  Of auto-declined applications:
    Appealed by applicant:           8,924  (4.0%)
    Appeal reviewed by human:        8,924  (100% of appeals)
    Appeal overturned:               1,338  (15.0% of appeals)

  Note: Applicants must actively submit an appeal through a form
  linked in the rejection email. The appeal process is described
  in FAQ section 7 of the website (3 clicks from homepage).

HUMAN REVIEWER TOOLS:
  Reviewers see the AI score and top-5 contributing features but
  no full explanation of the model's reasoning. The reviewer
  interface does not highlight cases where the model's confidence
  is low or where protected characteristics may be influencing
  the outcome.""",

        transparency_info="""\
TRANSPARENCY REVIEW
System: CreditFlow AI Score __VERSION__

APPLICANT NOTIFICATIONS:
  Application form includes the following notice:

    "Your application will be assessed using automated decision-making
    systems. You have the right to request human review of any
    automated decision."

  Rejection email text:
    "Based on our assessment, we are unable to offer you a loan at
    this time. If you wish to understand the main factors behind this
    decision or request a manual review, please contact our support
    team or visit [link]."

  The rejection email links to a generic FAQ page. The FAQ states
  that decisions are made using "a combination of credit history,
  financial data, and statistical models" but does not mention
  alternative data sources (social media, device metadata).

RIGHT TO EXPLANATION:
  Applicants can request an explanation by contacting support.
  Support agents provide a templated response listing the top 3
  general factors (e.g., "credit history length," "income level,"
  "existing debt") without specifying which exact features or
  thresholds drove the specific decision.

  No individualized explanation is generated. The support team
  does not have access to the model's per-application feature
  importance breakdown.""",

        risk_assessment_info="""\
RISK MANAGEMENT & CONFORMITY ASSESSMENT
System: CreditFlow AI Score __VERSION__

ANNEX III CLASSIFICATION:
  Category 5(b) — AI systems intended to be used to evaluate the
  creditworthiness of natural persons.

  This system directly determines loan approval/rejection for
  consumer credit up to EUR 50,000. It falls squarely within the
  high-risk category.

CONFORMITY ASSESSMENT STATUS:
  Internal conformity assessment:    Not initiated
  Quality management system:         ISO 27001 certified but no
                                     AI-specific QMS per Article 17
  EU Declaration of Conformity:      Not filed
  CE marking:                        Not applied

DORA ALIGNMENT:
  As a financial services firm, the company has ICT risk management
  procedures under DORA (Digital Operational Resilience Act). These
  cover general IT risks but do not address AI-specific risks
  such as model drift, adversarial inputs, or algorithmic bias.

RISK MANAGEMENT GAPS:
  The company's risk register includes traditional financial risks
  (credit risk, market risk, operational risk) but does not include:
    - AI model risk as a distinct risk category
    - Bias risk for protected characteristics
    - Alternative data correlation risk
    - Drift detection requirements""",

        logging_info="""\
LOGGING & TRACEABILITY REVIEW
System: CreditFlow AI Score __VERSION__

CURRENT LOGGING:
  Event Type               Logged   Format     Retention
  ─────────────────────────────────────────────────────
  Application received     Yes      JSON       7 years
  Score generated          Yes      Numeric    7 years
  Decision (approve/deny)  Yes      String     7 years
  Features used            Yes      Vector     7 years
  Model version            Yes      String     Indefinite
  Human review action      Yes      JSON       7 years
  Appeal and outcome       Yes      JSON       7 years

  Logging is comprehensive with input-output linkage. All decisions
  can be reconstructed. Feature vectors are stored alongside scores,
  enabling retrospective audits.

  Financial services retention requirements (7 years) are met.
  Article 12 logging requirements for high-risk AI appear to be
  adequately addressed at the technical level.

  NOTE: While individual decision logs are complete, there is no
  automated aggregation or monitoring dashboard that tracks
  demographic patterns across decisions over time.""",
    )

    s.graph = _build_scenario_graph(
        investigation_tools=[
            "check_documentation", "audit_training_data",
            "verify_human_oversight", "check_transparency",
            "assess_risk_management",
        ],
    )
    return s


def _medium_medical() -> AuditScenario:
    s = AuditScenario(
        scenario_id="medium_medical_triage_001",
        title="Emergency Triage AI — Medical Device Compliance",
        difficulty="medium",
        description=(
            "A hospital network deployed an AI system that prioritizes emergency "
            "department patients based on vital signs and symptoms. As a medical "
            "device with AI, it falls under both the EU AI Act (high-risk, Annex III) "
            "and the Medical Devices Regulation (MDR). Audit required."
        ),
        system_name="TriageAI Priority System",
        system_description=(
            "ML model processing vital signs (heart rate, BP, SpO2, temperature), "
            "symptoms, and medical history to assign emergency triage priority "
            "(ESI levels 1-5). Used in 12 EU hospitals across DE/AT/CH. Decisions "
            "directly affect patient care timing."
        ),
        system_category="high_risk",
        deployer_info="__COMPANY__ — German health-tech company, certified under MDR, deployed in DE/AT/CH hospitals.",
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
            "prohibited_system",
            "missing_ce_marking",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: TriageAI Priority System __VERSION__
Deployer: __COMPANY__
Audit Date: __DATE__

DOCUMENT STATUS:
  Document                                  Status     Last Updated
  ─────────────────────────────────────────────────────────────────
  System Architecture                       Present    2025-09-15
  Clinical Evaluation Report                Present    2025-03-20
  MDR Technical Documentation               Present    2025-09-15
  Intended Purpose Statement                Present    2025-09-15
  Software Life Cycle Documentation         Present    2025-11-01
  AI Act Annex IV Documentation             Absent     —
  Post-Market Clinical Follow-up Plan       Absent     —
  Post-Market Surveillance Plan (AI)        Absent     —

MDR CONFORMITY:
  CE marking:          Applied (Class IIa medical device)
  Notified Body:       BSI Group (NB 0086)
  Last MDR audit:      2025-06-12 — no non-conformities

  The system has valid MDR conformity assessment. However, the
  EU AI Act imposes ADDITIONAL requirements beyond MDR compliance
  for AI-enabled medical devices classified as high-risk under
  Annex III.

CLINICAL EVALUATION:
  The Clinical Evaluation Report (CER) is based on:
    - Retrospective analysis of 500K historical ER visits
    - Literature review of 23 published studies on AI triage
    - No prospective clinical trial has been conducted
    - CER does not address AI-specific performance degradation
      (concept drift, distribution shift between hospitals)

  NOTE: MDR clinical evaluation accepted the retrospective analysis.
  The EU AI Act may require additional validation demonstrating
  real-world performance across deployment sites.""",

        training_data_info="""\
TRAINING DATA AUDIT REPORT
System: TriageAI Priority System __VERSION__
Dataset: Historical ER visit records from 3 university hospitals
Records: 512,847 patient encounters (2018-2024)

DEMOGRAPHIC DISTRIBUTION:
  Category               Count      Pct     ESI 1-2 Rate
  ──────────────────────────────────────────────────────────
  Age 0-17               71,799    14.0%    8.2%
  Age 18-44             179,497    35.0%    6.1%
  Age 45-64             143,597    28.0%    9.3%
  Age 65-74              76,927    15.0%    14.7%
  Age 75+                41,027     8.0%    19.8%

MODEL PERFORMANCE BY AGE GROUP (ESI Classification Accuracy):
  Age Group     Accuracy    Sensitivity(ESI 1-2)    Specificity
  ──────────────────────────────────────────────────────────────
  0-17          91.3%       89.1%                   92.0%
  18-44         93.7%       91.8%                   94.2%
  45-64         92.1%       90.4%                   93.1%
  65-74         88.4%       84.2%                   90.7%
  75+           82.6%       76.3%                   85.8%

  Performance degrades notably for patients aged 75+. Sensitivity
  for the highest-acuity patients (ESI 1-2) drops to 76.3% for
  the elderly cohort — meaning 23.7% of critical elderly patients
  may be under-triaged.

TRAINING DATA COMPOSITION:
  Patients aged 75+ represent 8.0% of the training data but 19.8%
  of ESI 1-2 presentations. The model was predominantly trained on
  younger demographics.

  Data from 3 hospitals in Germany only. No Austrian or Swiss
  patient data despite deployment in AT/CH hospitals.

CLINICAL VALIDATION:
  Validation approach:  Retrospective holdout (80/20 split)
  No prospective trial conducted.
  No external validation on data from deployment hospitals.
  No assessment of performance variation across deployment sites.

NOTE: The system holds valid CE marking under MDR as a Class IIa
device. MDR conformity does not exempt from AI Act requirements.""",

        oversight_info="""\
HUMAN OVERSIGHT PROCEDURES
System: TriageAI Priority System __VERSION__
Department: Emergency Department Operations

CLINICAL WORKFLOW:
  1. Patient arrives at ER and is registered at reception
  2. Initial vitals collected by triage nurse (HR, BP, SpO2, temp)
  3. Nurse enters symptoms and relevant history into the system
  4. TriageAI generates ESI level recommendation (1-5)
  5. Triage nurse reviews and can accept or override the AI suggestion
  6. Patient is directed to appropriate care area

OVERRIDE STATISTICS (2025 Q3-Q4, across all 12 hospitals):
  Total triage assessments:       187,423
  AI recommendations accepted:    171,577   (91.5%)
  Nurse overrides:                 15,846   (8.5%)
    Override to higher acuity:      9,508   (60.0% of overrides)
    Override to lower acuity:       6,338   (40.0% of overrides)

  By ESI level:
    ESI 1 (resuscitation):  All reviewed by attending physician
    ESI 2 (emergent):       Nurse review + attending notification
    ESI 3 (urgent):         Nurse review only
    ESI 4 (less urgent):    Nurse review only
    ESI 5 (non-urgent):     Nurse review only

  The system does not flag cases where its confidence is low.
  There is no visual indicator distinguishing high-confidence from
  borderline recommendations. Nurses report in surveys that they
  tend to "trust the system" unless the recommendation is clearly
  at odds with their clinical judgment.

AFTER-HOURS OPERATIONS:
  Staffing levels are reduced between 22:00-06:00. During this window,
  a single triage nurse handles all incoming patients. Override rates
  drop to 4.2% during overnight shifts (vs 8.5% daytime).""",

        transparency_info="""\
TRANSPARENCY REVIEW
System: TriageAI Priority System __VERSION__

PATIENT-FACING COMMUNICATION:
  Patients are not informed that an AI system is involved in their
  triage assessment. The triage process appears fully nurse-directed
  from the patient's perspective.

  Hospital intake forms do not mention AI-assisted triage.
  The hospitals' privacy notices (available on their websites) include
  a general statement about "digital health technologies" being used
  to support clinical decisions, but do not specifically mention
  TriageAI or AI-based triage prioritization.

CLINICIAN-FACING INFORMATION:
  Triage nurses see the AI's recommended ESI level on their screen
  alongside a summary of input vital signs. The interface does NOT
  show:
    - The model's confidence score
    - Which factors most influenced the recommendation
    - Whether the patient falls into a demographic group where the
      model has known lower accuracy

  Attending physicians can view the AI recommendation in the patient
  record but receive no additional context about the model's reasoning.

ARTICLE 13 USER INSTRUCTIONS:
  A deployment guide was provided to hospital IT departments describing
  system architecture, integration points, and API specifications.
  The guide does not include information about:
    - Known accuracy limitations by demographic group
    - Situations where the system should not be relied upon
    - Procedures for reporting suspected AI errors""",

        risk_assessment_info="""\
RISK MANAGEMENT & CONFORMITY ASSESSMENT
System: TriageAI Priority System __VERSION__

ANNEX III CLASSIFICATION:
  The system falls under multiple Annex III categories:
  - Category 5(c): AI intended for use as a safety component of a
    product covered by Union harmonisation legislation (MDR)
  - Category 5(a): AI intended for evaluation of eligibility for
    essential public services (healthcare access/prioritization)

  Classification: HIGH-RISK

MDR CONFORMITY STATUS:
  CE marking applied:                Yes (Class IIa)
  Notified body:                     BSI Group (NB 0086)
  Last periodic audit:               2025-06-12
  Non-conformities found:            None under MDR

EU AI ACT CONFORMITY STATUS:
  The EU AI Act imposes requirements ADDITIONAL to MDR:
  Internal conformity assessment:    Not initiated
  AI-specific risk management:       Not addressed
  Post-market monitoring (AI):       Not established

RISK MANAGEMENT:
  An MDR risk management file exists (ISO 14971 compliant).
  It covers clinical risks and software hazards but does not address:
    - AI-specific risks (distribution shift, adversarial inputs)
    - Performance degradation for specific demographic groups
    - Failure modes unique to the ML model
    - Concept drift between training data and deployment population

POST-MARKET MONITORING:
  MDR PMCF (Post-Market Clinical Follow-up) plan exists.
  No AI-specific post-market surveillance has been established.
  There is no system for monitoring:
    - Real-time triage accuracy at individual hospital level
    - Demographic performance variation over time
    - Model prediction confidence distribution shifts""",

        logging_info="""\
LOGGING & TRACEABILITY REVIEW
System: TriageAI Priority System __VERSION__

CURRENT LOGGING:
  Event Type                    Logged   Retention
  ──────────────────────────────────────────────────
  Patient vitals input          Yes      10 years (clinical record)
  AI ESI recommendation         Yes      10 years
  Nurse final ESI assignment    Yes      10 years
  Override indicator            Yes      10 years
  Model version                 Yes      Indefinite
  Response latency              Yes      1 year
  System errors/timeouts        Yes      1 year

  Logs are stored in the hospital's electronic health record (EHR)
  system. Each triage event links vitals, AI recommendation, and
  final nurse decision.

MONITORING CAPABILITIES:
  Individual triage events: Fully traceable
  Aggregate accuracy reports: Generated quarterly (manual process)
  Real-time performance monitoring: Not implemented
  Automated drift detection: Not implemented

  The quarterly accuracy reports compare AI recommendations against
  final nurse decisions and patient outcomes (admission, discharge).
  Reports are produced by the clinical informatics team and reviewed
  at quarterly governance meetings.

  There is no mechanism for near-real-time detection of performance
  degradation. A significant drop in accuracy could go unnoticed
  for up to 3 months between quarterly reviews.""",
    )

    s.graph = _build_scenario_graph(
        investigation_tools=[
            "check_documentation", "audit_training_data",
            "verify_human_oversight", "check_transparency",
            "assess_risk_management", "check_logging",
        ],
    )
    return s


# ---------------------------------------------------------------------------
# HARD SCENARIOS
# ---------------------------------------------------------------------------

def _hard_social_scoring() -> AuditScenario:
    s = AuditScenario(
        scenario_id="hard_social_scoring_prohibited_001",
        title="Citizen Wellness App — PROHIBITED System Detection",
        difficulty="hard",
        description=(
            "A 'wellness and civic engagement' app has been flagged for review. "
            "The deployer describes it as a voluntary digital wellness platform "
            "that gamifies community participation. Municipal governments in "
            "Italy and Spain have partnered with the deployer. Conduct a "
            "thorough compliance audit."
        ),
        system_name="CivicPulse Wellness Score",
        system_description=(
            "CivicPulse is a digital wellness and civic engagement platform that "
            "helps citizens track and improve their community participation. The app "
            "provides personalized recommendations for local events, volunteer "
            "opportunities, and community initiatives. Users receive a 'wellness score' "
            "reflecting their engagement level. __USERCOUNT__ citizens across 3 "
            "municipalities currently use the platform."
        ),
        system_category="prohibited",
        deployer_info="__COMPANY__ — EU-based civic tech startup, partnered with 3 municipal governments in IT/ES.",
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
            "high_risk_classification",
            "needs_conformity_assessment",
            "voluntary_participation",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: CivicPulse Wellness Score __VERSION__
Deployer: __COMPANY__
Audit Date: __DATE__

DOCUMENT STATUS:
  Document                              Status     Last Updated
  ─────────────────────────────────────────────────────────────
  Platform Architecture                 Present    2025-10-30
  Scoring Algorithm Description         Present    2025-10-30
  Data Integration Specifications       Present    2025-08-14
  Municipal Partnership Agreements      Present    2025-06-01
  Privacy Impact Assessment             Present    2025-07-22
  User Guide / Onboarding Materials     Present    2025-11-05

SCORING ALGORITHM DESCRIPTION (from deployer documentation):
  "The CivicPulse Wellness Score is a composite metric ranging from
  0 to 1000, calculated from the following weighted components:

    Community Event Attendance (25%)   — tracked via QR check-ins
    Volunteer Hours Logged (20%)       — verified by partner orgs
    Social Media Civic Posts (15%)     — public posts tagged #CivicPulse
    Payment History Reliability (15%)  — municipal utility payments
    Civic Survey Participation (10%)   — in-app micro-surveys
    Peer Endorsements (10%)            — other users vouching
    App Engagement Metrics (5%)        — daily active usage

  Scores are updated weekly. Score history is retained indefinitely.
  Users can view their score and breakdown in the app."

MUNICIPAL PARTNERSHIP AGREEMENTS (summary):
  Three municipal governments have signed integration agreements:
    - Municipality of Torino (IT) — population 848,000
    - Municipality of Sevilla (ES) — population 681,000
    - Municipality of Valencia (ES) — population 791,000

  Under these agreements, the municipalities provide:
    - API access to utility payment records
    - Event attendance verification infrastructure
    - Integration with municipal service booking systems

  In return, the municipalities receive:
    - Aggregate civic engagement analytics
    - "Priority queue integration" for municipal services""",

        training_data_info="""\
DATA SOURCES & PROCESSING REPORT
System: CivicPulse Wellness Score __VERSION__

The system aggregates data from the following sources:

SOURCE INVENTORY:
  Source                       Records      Update Freq   Consent
  ────────────────────────────────────────────────────────────────
  Social media APIs            487,291      Daily         App ToS
  Municipal utility records    312,847      Monthly       Municipal MOU
  Event attendance (QR)        891,234      Real-time     App ToS
  Volunteer organization data  156,723      Weekly        Partner MOU
  In-app behavior             2,341,567     Real-time     App ToS
  Peer endorsement graph       234,891      Real-time     App ToS

PERSONAL DATA PROCESSED:
  - Full name and municipal ID (for service integration)
  - Social media activity (public posts, frequency, topics)
  - Utility payment timeliness and amounts
  - Physical location (event check-ins, frequency maps)
  - Volunteer activity (hours, organizations, regularity)
  - In-app behavior (session duration, feature usage)
  - Social graph (who endorses whom, connection density)

BIAS ASSESSMENT:
  No formal bias assessment has been conducted. The deployer states
  that the scoring algorithm is "objective and based on observable
  civic engagement indicators."

  Preliminary analysis of score distribution:
    Score Range     Pct of Users   Avg Monthly Income (self-reported)
    ─────────────────────────────────────────────────────────────────
    800-1000          12.3%        EUR 4,200
    600-799           28.7%        EUR 3,100
    400-599           34.1%        EUR 2,300
    200-399           18.4%        EUR 1,700
    0-199              6.5%        EUR 1,100

  Higher scores correlate strongly with higher income. Citizens with
  lower income have less time for volunteer activities, fewer social
  media posts, and less stable utility payment histories.""",

        oversight_info="""\
HUMAN OVERSIGHT & GOVERNANCE
System: CivicPulse Wellness Score __VERSION__

GOVERNANCE STRUCTURE:
  The platform is operated by __COMPANY__ with oversight from a
  "Civic Advisory Board" consisting of:
    - 2 company representatives
    - 1 municipal liaison per partner city
    - 1 data protection consultant

  The Advisory Board meets quarterly to review:
    - Platform usage statistics
    - Score distribution trends
    - User feedback summaries
    - New feature proposals

SCORE ADJUSTMENT CAPABILITY:
  Municipal administrators have access to a dashboard where they can:
    - View individual citizen scores
    - Apply manual score adjustments (with documented reason)
    - Exclude specific citizens from the scoring system
    - Configure score thresholds for municipal service integration

  In 2025, municipal administrators made 847 manual adjustments:
    - 612 score increases (typically after verified volunteer hours
      were not automatically captured)
    - 235 score decreases (typically after fraudulent check-ins
      were identified)

USER OPT-OUT:
  Users can delete their account through in-app settings. The deployer
  states participation is fully voluntary. However, the municipal
  service integration means that citizens without a CivicPulse account
  do not have access to the "priority queue" for municipal services
  (see Municipal Services Integration below).

MUNICIPAL SERVICES INTEGRATION:
  Citizens with CivicPulse scores above 600 receive:
    - Priority appointment slots for municipal offices
    - Reduced wait times for public healthcare appointments
    - 10% discount on municipal parking permits
    - Priority enrollment in public recreational programs

  Citizens with scores below 300 experience:
    - Standard (non-priority) queue for all municipal services
    - No access to priority healthcare appointment slots
    - Standard parking permit pricing
    - Standard enrollment priority for recreational programs

  The deployer describes this as "rewarding engaged citizens" rather
  than penalizing low scorers. Citizens without an account are
  treated identically to those with scores below 300.""",

        transparency_info="""\
TRANSPARENCY REVIEW
System: CivicPulse Wellness Score __VERSION__

USER-FACING INFORMATION:
  The app displays each user's current score prominently on the
  home screen along with a breakdown by category (events, volunteering,
  social, payments, surveys, endorsements, engagement).

  Users can see:
    - Their current score and weekly trend
    - Category-level breakdown
    - "Tips to improve your score" (e.g., "Attend 2 more events
      this month to boost your Community score by 50 points")
    - Leaderboard showing anonymized score rankings in their area

  Users are NOT told:
    - That their score affects access to municipal services
    - That utility payment history is factored into the score
    - That social media activity is monitored and scored
    - That users without accounts (or with low scores) receive
      lower priority for public services
    - The full list of data sources and their weights
    - How the scoring algorithm handles edge cases

MUNICIPAL COMMUNICATION:
  Partner municipalities have not publicly announced the integration
  between CivicPulse scores and service prioritization. The municipal
  booking systems show "priority" and "standard" queues without
  explaining the criteria for queue assignment.

MARKETING MATERIALS:
  The company website describes CivicPulse as: "A fun, gamified way
  to strengthen your connection with your community. Track your civic
  engagement, discover local events, and earn rewards for being an
  active citizen."

  No marketing materials mention the scoring system's impact on
  access to public services.""",

        risk_assessment_info="""\
RISK CLASSIFICATION ANALYSIS
System: CivicPulse Wellness Score __VERSION__

DEPLOYER'S SELF-CLASSIFICATION:
  The deployer has classified the system as "limited risk — voluntary
  wellness platform" and states that no conformity assessment is
  required. The deployer's risk assessment document argues:

  "CivicPulse is a voluntary civic engagement tool. Users choose to
  participate and can opt out at any time. The wellness score is a
  motivational gamification feature, similar to fitness tracker
  achievement systems. No consequential decisions are made based
  on the score."

AUDITOR'S ANALYSIS:

  Article 5(1)(c) of the EU AI Act PROHIBITS:
  "AI systems used for the evaluation or classification of natural
  persons or groups thereof over a certain period of time based on
  their social behaviour or known, inferred or predicted personal
  or personality characteristics, with the social score leading to
  either or both of the following:
    (i) detrimental or unfavourable treatment of certain natural
        persons or groups thereof in social contexts which are
        unrelated to the contexts in which the data was originally
        generated or collected;
    (ii) detrimental or unfavourable treatment of certain natural
         persons or groups thereof that is unjustified or
         disproportionate to their social behaviour or its gravity."

  KEY ASSESSMENT FACTORS:
  1. The system evaluates citizens based on social behavior (social
     media activity, community participation, peer endorsements)
  2. Scores determine priority access to public services (healthcare
     appointments, municipal office queues, recreational programs)
  3. Low scorers receive worse service across domains unrelated
     to the scored behaviors (utility payments don't relate to
     healthcare priority)
  4. Municipal governments — public authorities — are partners in
     deploying and using the scoring system
  5. Participation is described as "voluntary" but non-participants
     are treated identically to low scorers""",

        logging_info="""\
LOGGING REVIEW
System: CivicPulse Wellness Score __VERSION__

SYSTEM LOGS:
  The platform maintains extensive logging:
    - Score calculations and weekly updates
    - Data source ingestion events
    - User interactions (app opens, feature usage)
    - Municipal API calls (service queue assignments)
    - Administrator actions (manual adjustments)
    - Score-to-service mapping events

  All logs are retained for the lifetime of the platform.

  NOTE: The comprehensiveness of logging is not relevant to the
  primary compliance question for this system. If the system is
  found to constitute a prohibited practice under Article 5,
  no amount of logging, oversight, or documentation can make
  it compliant. The appropriate response to a prohibited system
  is cessation of operation, not improvement of governance.""",

        deep_transparency="""\
DEEP DIVE: SERVICE ACCESS IMPACT ANALYSIS
Re-investigation of CivicPulse municipal integration

QUANTIFIED IMPACT ON PUBLIC SERVICES:
  Analysis of municipal booking system data (2025-Q3):

  Healthcare appointments (Torino):
    Score 800+:  Average wait 2.3 days for non-urgent appointment
    Score 400-599: Average wait 8.7 days
    Score < 300: Average wait 14.2 days
    No account: Average wait 14.2 days (same as lowest tier)

  Municipal office appointments (Sevilla):
    Priority queue (score 600+): 87% seen within 15 minutes
    Standard queue (score < 600): 34% seen within 15 minutes
    Average wait difference: 42 minutes

  Recreational program enrollment (Valencia):
    Priority enrollment opens 2 weeks before standard
    Score 600+: 94% get first-choice program
    Score < 600: 31% get first-choice program

CORRELATION WITH SOCIOECONOMIC STATUS:
  Citizens in lowest income quartile: avg score 287 (below 300 threshold)
  Citizens in highest income quartile: avg score 741 (above 600 threshold)
  Pearson correlation (income, score): r = 0.73, p < 0.001

  The scoring system systematically disadvantages lower-income citizens
  in access to healthcare, municipal services, and recreational programs.""",

        deep_oversight="""\
DEEP DIVE: MUNICIPAL ADMINISTRATOR ACTIONS
Re-investigation of CivicPulse governance

ADMINISTRATOR SCORE ADJUSTMENTS (detailed):
  Of 847 manual adjustments in 2025:
    Political affiliation correlation detected:
      - 89% of score increases were for citizens active in the
        ruling party's community organizations
      - Municipal liaison (advisory board member) initiated 67%
        of all score increase requests

  COMPLAINT HANDLING:
    23 citizen complaints received about service queue disparities
    0 complaints resulted in investigation
    Standard response: "CivicPulse is a voluntary wellness program.
    Queue prioritization is based on engagement metrics."
    No disclosure that the 'engagement metrics' ARE the CivicPulse score""",
    )

    # Prohibited system: short investigation path
    s.graph = _build_scenario_graph(
        investigation_tools=["check_documentation", "audit_training_data",
                             "verify_human_oversight", "check_transparency"],
        is_prohibited=True,
    )
    return s


def _hard_deepfake() -> AuditScenario:
    s = AuditScenario(
        scenario_id="hard_deepfake_generation_001",
        title="AI Content Studio — Deepfake & Synthetic Media Compliance",
        difficulty="hard",
        description=(
            "A content creation platform offers AI-generated synthetic media "
            "including face-swaps, voice cloning, and AI-generated video. Recent "
            "reports of generated political content going viral without AI labels. "
            "Multiple Article 50 compliance angles to investigate."
        ),
        system_name="SynthStudio Pro",
        system_description=(
            "Suite of generative AI tools: face-swap (video), voice cloning, "
            "text-to-video, AI image generation. Used by __USERCOUNT__ EU creators. "
            "Content is shared across all major social platforms. Platform serves "
            "creators, marketing agencies, and entertainment companies."
        ),
        system_category="limited_risk",
        deployer_info="__COMPANY__ — Irish AI startup, 80 employees, __USERCOUNT__ EU users.",
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
            "prohibited_manipulation",
            "high_risk_biometric",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: SynthStudio Pro __VERSION__
Deployer: __COMPANY__
Audit Date: __DATE__

DOCUMENT STATUS:
  Document                                  Status     Last Updated
  ─────────────────────────────────────────────────────────────────
  Platform Architecture                     Present    2025-11-10
  Model Cards (per generation model)        Present    2025-10-01
  API Documentation                         Present    2025-11-10
  Content Policy / Terms of Service         Present    2025-09-15
  Content Labeling Implementation           Absent     —
  Content Authentication / Provenance       Absent     —
  Watermarking Specification                Absent     —
  Content Moderation Procedures             Present    2025-04-20

GENERATION MODELS:
  Model              Type              Training Data
  ──────────────────────────────────────────────────────
  FaceSwap v3        GAN               CelebA + proprietary
  VoiceClone v2      Neural TTS        LibriTTS + licensed
  TextToVideo v1     Diffusion         WebVid-10M + licensed
  ImageGen v4        Latent Diffusion   LAION-filtered + licensed

CONTENT POLICY (from Terms of Service):
  Section 4.2: "Users agree not to use SynthStudio for: (a) creating
  content intended to deceive or defraud, (b) non-consensual intimate
  imagery, (c) content targeting minors, (d) content that violates
  applicable law."

  Section 4.3: "Users are responsible for ensuring their use of
  generated content complies with all applicable laws and regulations."

  Enforcement: The content policy is enforced reactively. Users report
  violations via an in-platform form. Average response time: 72+ hours.
  No proactive content scanning is implemented.""",

        training_data_info="""\
TRAINING DATA & CONSENT REPORT
System: SynthStudio Pro __VERSION__

TRAINING DATA SOURCES:
  Model          Dataset             Size        Consent Status
  ──────────────────────────────────────────────────────────────
  FaceSwap v3    CelebA              202,599     Research license only;
                                                 individual consent not
                                                 obtained from subjects
                 Proprietary set     84,231      Licensed from stock
                                                 media agencies

  VoiceClone v2  LibriTTS            585 hrs     CC-BY 4.0 license
                 Licensed voices     200 hrs     Individual consent

  TextToVideo    WebVid-10M          10M clips   Web-scraped; no
                                                 individual consent
                 Licensed footage    500K clips  Commercial license

  ImageGen v4    LAION-filtered      2.3B imgs   Web-scraped; filtered
                                                 for CSAM but not for
                                                 individual consent
                 Licensed imagery    1.2M imgs   Commercial license

CONSENT CONCERNS:
  The FaceSwap model was trained partly on CelebA, which contains
  photos of public figures collected without individual consent for
  AI training purposes. While the images are publicly available,
  training face-swap models on non-consenting individuals' likenesses
  raises ethical and potentially legal concerns under GDPR Article 6.

  WebVid-10M and LAION-filtered datasets are web-scraped collections.
  Content creators depicted in these datasets did not consent to their
  content being used for AI model training.

DEEPFAKE DETECTION:
  SynthStudio does not include any built-in deepfake detection
  capability. Generated content is not distinguishable from authentic
  content without external forensic analysis tools.

USAGE STATISTICS (2025):
  Face-swaps generated:          2,847,291
  Voice clones created:            891,234
  Videos generated:              1,234,567
  Images generated:             12,456,789
  Content flagged by users:         4,231  (0.02% of total output)
  Content removed after review:     1,847  (43.6% of flagged)""",

        oversight_info="""\
CONTENT MODERATION & OVERSIGHT
System: SynthStudio Pro __VERSION__

MODERATION PROCESS:
  SynthStudio operates a reactive content moderation system:

  1. Automated pre-screening: Basic NSFW classifier runs on image
     generation outputs (estimated 91% accuracy). Flagged content
     requires manual review before delivery.

  2. User reporting: Any user can flag content via a report button.
     Reports are queued for the Trust & Safety team.

  3. Trust & Safety team: 6 full-time moderators review reported
     content. Working hours: Mon-Fri, 09:00-18:00 IST.

  MODERATION STATISTICS (2025):
    Content generated:           17,429,881
    Auto-flagged (NSFW):            182,471  (1.05%)
    User reports:                     4,231  (0.02%)
    Reviewed by T&S team:             6,892
    Content removed:                  1,847
    Average review time:              74 hours

  No proactive scanning for:
    - Political disinformation
    - Non-consensual deepfakes of real individuals
    - Misleading news or propaganda
    - Content impersonating public figures

POLITICAL CONTENT:
  SynthStudio has no special handling for political content.
  Users have generated content depicting politicians in fabricated
  scenarios. At least 3 instances of AI-generated political content
  went viral on social media in 2025 without any AI disclosure.
  The company became aware through media reports, not internal
  detection.

  No restrictions exist on generating content depicting:
    - Political figures
    - Electoral/campaign material
    - News-like content""",

        transparency_info="""\
TRANSPARENCY & CONTENT LABELING REVIEW
System: SynthStudio Pro __VERSION__

AI CONTENT LABELING:

  Article 50(2) requires: "Providers of AI systems, including
  general-purpose AI systems, generating synthetic audio, image,
  video or text content, shall ensure that the outputs of the AI
  system are marked in a machine-readable format and detectable as
  artificially generated or manipulated."

  Current implementation:
    - Generated images: No AI label or metadata tag applied
    - Generated videos: No AI label or metadata tag applied
    - Generated audio:  No AI label or metadata tag applied
    - Face-swaps:       No AI label or metadata tag applied

  When users download generated content, it is delivered as a
  standard media file (JPEG, MP4, WAV) with no embedded metadata
  indicating AI generation.

CONTENT PROVENANCE:
  C2PA (Coalition for Content Provenance and Authenticity):
    Not implemented. No content credentials are attached to
    generated media.

  IPTC metadata:
    Not implemented. No AI generation metadata in EXIF/XMP fields.

  Digital watermarking:
    Not implemented. Generated content contains no steganographic
    or perceptual watermarks.

  After download, generated content is indistinguishable from
  authentic media using standard tools.

USER AGREEMENTS:
  The Terms of Service (Section 6.1) state:
    "Users are responsible for disclosing the AI-generated nature
    of content when required by applicable law."

  This places the disclosure burden entirely on the user, but
  Article 50(2) places the obligation on the PROVIDER to ensure
  outputs are marked, not merely on users to self-disclose.

PLATFORM UI:
  Within the SynthStudio platform, generated content is displayed
  with a small "AI Generated" tag in the project view. This tag
  does not persist when content is downloaded or exported. No
  option exists to embed permanent AI labels in exported content.""",

        risk_assessment_info="""\
RISK CLASSIFICATION ANALYSIS
System: SynthStudio Pro __VERSION__

ANNEX III HIGH-RISK CHECK:
  1. Biometric identification:  The face-swap tool processes facial
     features but is used for content CREATION, not identification.
     It does not identify individuals — it transfers facial
     appearance between subjects. This does not fall under the
     biometric identification category of Annex III.

  2-8. Other high-risk categories: Not applicable — the system
     creates media content, it does not make decisions affecting
     individuals' rights, access to services, or legal status.

ARTICLE 5 PROHIBITED PRACTICES:
  Subliminal manipulation: The system creates content on user
  request. It does not autonomously deploy manipulative content.
  However, the OUTPUTS could be used for manipulation if shared
  without AI disclosure.

  The tool itself is not a prohibited practice, but it can
  facilitate prohibited outcomes if misused.

RISK LEVEL DETERMINATION: Limited Risk
  Primary obligations fall under Article 50 transparency requirements
  for AI systems generating synthetic content.

  The platform's systemic risk lies not in the tool's classification
  level but in the scale of potentially misleading synthetic content
  being produced and distributed without provenance tracking.

CONTENT INTEGRITY RISK:
  The combination of: (a) high-quality synthetic media generation,
  (b) no content labeling, (c) no watermarking, and (d) no
  proactive content moderation creates significant systemic risk
  for information integrity, particularly around elections and
  public discourse.""",

        logging_info="""\
LOGGING & CONTENT TRACEABILITY REVIEW
System: SynthStudio Pro __VERSION__

GENERATION LOGS:
  Event Type                  Logged   Retention
  ─────────────────────────────────────────────
  Generation request          Yes      90 days
  Model and params used       Yes      90 days
  Input media (face source)   Yes      90 days
  Output media hash           Yes      90 days
  User account ID             Yes      90 days
  Download event              Yes      90 days
  Export destination           No      —

TRACEABILITY AFTER EXPORT:
  Once content is downloaded by the user, SynthStudio has no
  mechanism to track its distribution or usage. The output hash
  is retained for 90 days, but this only allows verification if
  the exact file is submitted back for checking.

  Content shared on social media, messaging apps, or websites
  cannot be traced back to SynthStudio or the creator without
  the original file hash.

CONTENT-TO-CREATOR LINKING:
  Within the 90-day retention window, SynthStudio can link a
  specific piece of content to the user account that generated it
  (via output hash matching).

  After 90 days, this linkage is permanently deleted.
  No legal hold or preservation mechanism exists for content
  involved in potential misuse investigations.

  For face-swap content specifically, there is no record of whose
  likeness was used as the source face, only the source image hash.""",
    )

    s.graph = _build_scenario_graph(
        investigation_tools=[
            "check_documentation", "audit_training_data",
            "verify_human_oversight", "check_transparency",
            "assess_risk_management", "check_logging",
        ],
    )
    return s


def _hard_multi_system() -> AuditScenario:
    s = AuditScenario(
        scenario_id="hard_multi_system_corporate_001",
        title="Corporate AI Portfolio Audit — Multi-System Compliance",
        difficulty="hard",
        description=(
            "A large enterprise uses 4 AI systems that need simultaneous audit: "
            "(1) employee sentiment analysis, (2) customer churn prediction, "
            "(3) automated invoice processing, (4) workplace safety monitoring "
            "with cameras. Each has different risk levels. The auditor must "
            "correctly classify each and identify cross-system data sharing risks."
        ),
        system_name="Enterprise AI Portfolio",
        system_description=(
            "Four interconnected AI systems sharing a common data lake: "
            "EmployeePulse (sentiment from Slack/email), ChurnGuard (customer "
            "retention prediction), InvoiceAI (AP automation), SafetyWatch "
            "(CCTV-based workplace monitoring). Deployed at __COMPANY__ "
            "manufacturing conglomerate, __USERCOUNT__ employees across EU."
        ),
        system_category="high_risk",
        deployer_info="__COMPANY__ — German manufacturing conglomerate, 15,000 employees, operating across EU.",
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
            "invoice_ai_high_risk",
            "all_systems_prohibited",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
Enterprise AI Portfolio — __COMPANY__
Audit Date: __DATE__

SYSTEM INVENTORY:
  System           Deployer Classification   Documentation
  ────────────────────────────────────────────────────────────────
  EmployeePulse    "Workforce analytics"     Per-system docs present
  ChurnGuard       "Customer analytics"      Per-system docs present
  InvoiceAI        "Process automation"      Per-system docs present
  SafetyWatch      "Safety compliance"       Per-system docs present

PER-SYSTEM DOCUMENTATION STATUS:

  EmployeePulse — Employee Sentiment Analysis:
    Architecture document:         Present (describes NLP pipeline)
    Data flow diagram:             Present (shows Slack/email ingestion)
    Algorithm description:         Present (BERT-based sentiment model)
    Performance metrics:           Present (F1: 0.84 on test set)
    DPIA:                          Present (standalone, 2024-11)
    Combined processing assessment: Absent

  ChurnGuard — Customer Churn Prediction:
    Architecture document:         Present
    Algorithm description:         Present (gradient-boosted trees)
    Data sources:                  Present (CRM, support tickets, usage)
    Performance metrics:           Present (AUC: 0.81)

  InvoiceAI — Automated Invoice Processing:
    Architecture document:         Present (OCR + classification)
    Processing rules:              Present
    Accuracy metrics:              Present (99.2% extraction accuracy)
    Error handling procedures:     Present

  SafetyWatch — Workplace Safety Monitoring:
    Architecture document:         Present (computer vision pipeline)
    Camera placement documentation: Present
    Detection model description:   Present (YOLO-based detection)
    Works council agreement:       Present (2024-06)

CROSS-SYSTEM DOCUMENTATION:
  Combined risk assessment:         ABSENT
  Cross-system data flow diagram:   ABSENT
  Combined DPIA:                    ABSENT
  Data lake access control matrix:  Present but outdated (2023-09)

  NOTE: Each system has individual documentation that appears
  adequate in isolation. No documentation addresses the combined
  risks of four AI systems sharing a common data infrastructure.""",

        training_data_info="""\
DATA AUDIT REPORT — MULTI-SYSTEM
Enterprise AI Portfolio — __COMPANY__

SYSTEM 1: EmployeePulse (Sentiment Analysis)
  Data sources:
    - Slack messages from internal workspace    (12.4M messages)
    - Email subject lines and metadata          (8.7M emails)
    - Employee survey responses                 (47K responses)
    - Meeting transcript summaries              (234K meetings)

  Personal data processed: Employee names, communication patterns,
  sentiment indicators, meeting participation frequency, response
  times, collaboration network metrics.

  Consent: Employees signed an "IT systems usage agreement" upon
  hiring that includes a clause: "The company may process workplace
  communications for operational analytics purposes." Employees
  were not specifically informed about AI-powered sentiment analysis.

  Note: Under EU labor law, consent given as a condition of
  employment may not constitute "freely given" consent under GDPR
  Article 7, as the power imbalance between employer and employee
  undermines voluntary choice.

SYSTEM 2: ChurnGuard (Customer Churn)
  Data sources:
    - CRM records                              (2.1M customers)
    - Support ticket history                   (5.8M tickets)
    - Product usage telemetry                  (real-time)
    - Contract terms and renewal dates         (2.1M contracts)

  Personal data: Customer names, contact info, usage patterns,
  support interaction history, contract details.

SYSTEM 3: InvoiceAI (Invoice Processing)
  Data sources:
    - Scanned invoices                         (3.4M documents)
    - Vendor database                          (12K vendors)
    - Purchase orders                          (1.8M orders)

  Personal data: Minimal — vendor business information only.
  No individual personal data processed.

SYSTEM 4: SafetyWatch (Workplace Monitoring)
  Data sources:
    - CCTV footage from 847 cameras across 23 facilities
    - Real-time video stream processing

  Processing details:
    - Object detection: Hard hat, safety vest, goggles presence
    - Zone violation: Entry into restricted areas
    - Pose estimation: Ergonomic risk assessment (bending, lifting)
    - FACIAL RECOGNITION: Used for zone access verification in
      restricted areas (R&D labs, chemical storage)

  The pose estimation module processes body positioning data that
  could constitute biometric categorization — inferring physical
  characteristics and behavior patterns of employees.

CROSS-SYSTEM DATA SHARING:
  All four systems access a shared Azure Data Lake (ADL) instance.
  Access control is implemented at the storage container level.

  OBSERVED DATA FLOWS:
    EmployeePulse → SharedLake:  Employee sentiment scores
    ChurnGuard ← SharedLake:    Pulls employee data for "internal
                                 engagement correlation" feature
    SafetyWatch → SharedLake:   Zone compliance records
    InvoiceAI → SharedLake:     Vendor payment data

  CONCERN: ChurnGuard's "internal engagement correlation" feature
  accesses EmployeePulse sentiment data to predict whether
  disengaged employees might cause customer churn through poor
  service. This creates an undocumented data flow where employee
  sentiment analysis affects customer-facing predictions.""",

        oversight_info="""\
HUMAN OVERSIGHT — MULTI-SYSTEM
Enterprise AI Portfolio — __COMPANY__

SYSTEM 1: EmployeePulse
  Oversight: HR Analytics team reviews monthly aggregate reports.
  Individual-level data accessible to HR Business Partners.
  No opt-out mechanism for employees.
  No employee notification that individual sentiment is tracked.
  Aggregated "team health" scores shared with department managers.
  HR reports that 3 employees were "counseled" in 2025 after
  EmployeePulse flagged sustained negative sentiment patterns.

SYSTEM 2: ChurnGuard
  Oversight: Customer Success team reviews churn predictions weekly.
  High-risk accounts flagged for proactive outreach.
  No direct impact on individual customers' service or pricing.
  Predictions used as advisory signals only.

SYSTEM 3: InvoiceAI
  Oversight: Finance team reviews all flagged exceptions (approx 3%
  of invoices). Full human review for invoices above EUR 50K.
  System handles routine three-way matching autonomously.
  Error rate: 0.8% (caught in downstream reconciliation).

SYSTEM 4: SafetyWatch
  Oversight: Safety officers monitor real-time alerts.
  All zone violations are reviewed within 15 minutes.
  Pose estimation alerts are reviewed in batches (daily).
  Facial recognition matches for restricted zones are logged and
  reviewed if there is a mismatch (attempted unauthorized access).

CROSS-SYSTEM OVERSIGHT:
  No unified oversight body monitors the interaction between systems.
  Each system has its own operational team:
    - EmployeePulse: HR Analytics (3 people)
    - ChurnGuard: Customer Success (5 people)
    - InvoiceAI: Finance Operations (2 people)
    - SafetyWatch: HSE Department (4 people)

  The IT department manages the shared data lake infrastructure but
  does not monitor data flows between systems from a compliance
  perspective. No data governance officer has been appointed with
  authority over cross-system data usage.""",

        transparency_info="""\
TRANSPARENCY REVIEW — MULTI-SYSTEM
Enterprise AI Portfolio — __COMPANY__

SYSTEM 1: EmployeePulse
  Employee notification: The company's internal IT policy document
  (available on the intranet, 47 pages) includes the statement:
  "Workplace communications may be processed for analytical purposes
  to support organizational effectiveness."

  Employees are NOT specifically told:
    - That AI analyzes their Slack messages and email metadata
    - That individual sentiment scores are generated
    - That these scores are accessible to HR Business Partners
    - That sentiment data flows to the ChurnGuard system
    - That sustained negative sentiment may trigger HR intervention

SYSTEM 2: ChurnGuard
  Customer notification: The company's privacy policy mentions
  "automated analysis to improve customer service." Customers are
  not informed that their accounts are scored for churn risk or
  that this scoring uses employee sentiment data internally.

SYSTEM 3: InvoiceAI
  Vendor notification: Vendors are informed that invoices are
  "processed electronically." No specific AI disclosure required
  as the system handles business documents, not personal data
  of natural persons in a consequential manner.

SYSTEM 4: SafetyWatch
  Employee notification: The works council agreement from 2024-06
  authorizes CCTV monitoring for safety purposes. The agreement
  specifically mentions:
    - PPE compliance detection (hard hats, vests, goggles)
    - Restricted zone monitoring
    - "Advanced safety analytics" (vague — does not specify
      pose estimation or facial recognition)

  Employees are aware of cameras but NOT specifically informed:
    - That pose estimation analyzes their body movements
    - That facial recognition identifies them in restricted zones
    - That safety compliance data is stored in the shared data lake""",

        risk_assessment_info="""\
RISK CLASSIFICATION — MULTI-SYSTEM
Enterprise AI Portfolio — __COMPANY__

PER-SYSTEM CLASSIFICATION ANALYSIS:

  SYSTEM 1: EmployeePulse (Sentiment Analysis)
    Deployer classification: "Workforce analytics tool — minimal risk"
    Annex III check: Category 4 — "AI systems intended to be used
    for making decisions affecting terms of work-related relationships"
    The system generates individual sentiment scores accessible to HR,
    and has been used as a factor in HR interventions (counseling).
    This constitutes a system that affects work-related relationships.
    AUDITOR ASSESSMENT: HIGH-RISK under Annex III Category 4

  SYSTEM 2: ChurnGuard (Customer Churn)
    Deployer classification: "Customer analytics — minimal risk"
    Annex III check: The system predicts customer churn for advisory
    purposes. It does not make decisions affecting individual
    customers' service level, pricing, or contract terms.
    In isolation: MINIMAL RISK
    However: Cross-system data flows (employee sentiment → churn
    prediction) create compound processing that was not assessed.

  SYSTEM 3: InvoiceAI (Invoice Processing)
    Deployer classification: "Process automation — minimal risk"
    Annex III check: No applicable category. The system processes
    business documents (invoices, POs) with minimal personal data.
    AUDITOR ASSESSMENT: MINIMAL RISK (correct classification)

  SYSTEM 4: SafetyWatch (Workplace Safety)
    Deployer classification: "Safety compliance — limited risk"
    Annex III check:
      - Pose estimation: May constitute biometric categorization
        (inferring physical characteristics) under Annex III Cat 1
      - Facial recognition for zone access: Biometric identification
        in a workplace context under Annex III Cat 1
      - Safety PPE detection: Standard computer vision, not biometric
    AUDITOR ASSESSMENT: Requires detailed assessment — components
    range from minimal risk (PPE detection) to potentially HIGH-RISK
    (facial recognition, pose estimation)

CROSS-SYSTEM RISK:
  No combined risk assessment has been conducted. The interaction
  between EmployeePulse sentiment data and ChurnGuard predictions
  creates a processing chain that was not individually assessed by
  either system's standalone documentation.

  A Data Protection Impact Assessment (DPIA) should have been
  conducted for the combined processing but none exists.""",

        logging_info="""\
LOGGING & TRACEABILITY — MULTI-SYSTEM
Enterprise AI Portfolio — __COMPANY__

PER-SYSTEM LOGGING:
  System           Decision Logs   Retention   Completeness
  ────────────────────────────────────────────────────────────
  EmployeePulse    Yes             2 years     Individual scores + inputs
  ChurnGuard       Yes             3 years     Predictions + features
  InvoiceAI        Yes             7 years     All processing steps
  SafetyWatch      Yes             30 days     Alerts + footage refs

CROSS-SYSTEM AUDIT TRAIL:
  Data lake access logs: Yes (Azure ADL audit logs enabled)
  Cross-system query logs: Present but not monitored
  Data flow lineage tracking: Not implemented

  The Azure Data Lake audit logs record which service principal
  accessed which data container and when. However:
    - No alerts are configured for unusual cross-system data access
    - No monitoring of data volume transferred between systems
    - No lineage tracking from source system to consuming system
    - Logs are retained for 90 days only (insufficient for audit)

SAFETWATCH SPECIFIC CONCERN:
  Video footage is retained for only 30 days. Facial recognition
  match logs are retained for 30 days. After this period, there
  is no way to audit whether the system correctly identified
  individuals or to investigate potential misidentification events.

  For a system that may be classified as high-risk (biometric
  identification in workplace), 30-day retention is likely
  insufficient to meet Article 12 logging requirements.""",
    )

    # Full investigation for multi-system audit
    s.graph = _build_scenario_graph(
        investigation_tools=[
            "check_documentation", "audit_training_data",
            "verify_human_oversight", "check_transparency",
            "assess_risk_management", "check_logging",
        ],
    )
    return s


# ---------------------------------------------------------------------------
# MEDIUM BONUS SCENARIO
# ---------------------------------------------------------------------------

def _medium_emotion_recognition() -> AuditScenario:
    s = AuditScenario(
        scenario_id="medium_emotion_recognition_workplace_001",
        title="Workplace Emotion Recognition — Article 5/50 Boundary Case",
        difficulty="medium",
        description=(
            "A corporate productivity platform uses webcam-based emotion recognition "
            "during video calls to measure 'employee engagement scores.' The EU AI Act "
            "prohibits emotion recognition in the workplace (Article 5(1)(f)) with narrow "
            "exceptions. Determine whether this system falls under the prohibition or "
            "the medical/safety exception."
        ),
        system_name="EngageMetrics Pro",
        system_description=(
            "Video analytics module integrated with the corporate video conferencing "
            "platform. Analyzes facial micro-expressions, voice tone, and body language "
            "during meetings to generate per-employee 'engagement scores' on a 0-100 "
            "scale. Scores are aggregated into team dashboards accessible to managers. "
            "Deployed across __USERCOUNT__ employees in __REGION__."
        ),
        system_category="prohibited",
        deployer_info="__COMPANY__ — US-based HR analytics firm, EU subsidiary in Ireland, serving 200+ EU corporate clients.",
        correct_classification="prohibited",
        ground_truth_findings=[
            "prohibited_workplace_emotion_recognition",
            "no_medical_or_safety_exception_applies",
            "employee_consent_not_freely_given",
            "covert_processing_of_biometric_data",
        ],
        required_remediation=[
            "immediate_cessation_of_emotion_recognition",
            "notify_data_protection_authority",
            "delete_all_emotion_scores_and_biometric_data",
            "inform_affected_employees",
        ],
        red_herrings=[
            "high_risk_classification",
            "needs_bias_audit",
            "medical_exception_applies",
        ],

        documentation_data="""\
TECHNICAL DOCUMENTATION INVENTORY
System: EngageMetrics Pro __VERSION__
Deployer: __COMPANY__
Audit Date: __DATE__

DOCUMENT STATUS:
  Document                                  Status     Last Updated
  ─────────────────────────────────────────────────────────────────
  Platform Architecture                     Present    2025-10-22
  ML Model Documentation                   Present    2025-10-22
  Data Processing Agreement (DPA)           Present    2025-08-01
  Employee Privacy Notice                   Present    2025-09-15
  Works Council Consultation Record         Absent     —
  DPIA                                      Present    2025-07-10

ML MODEL DETAILS:
  Architecture:     Multi-modal CNN (video) + Transformer (audio)
  Input:            Webcam video frames (10 fps) + microphone audio
  Output:           Engagement probability score (0-100)
  Features analyzed:
    - Facial Action Units (AU1-AU28) — eyebrow, lip, jaw movements
    - Gaze direction and duration
    - Head pose (pitch, yaw, roll)
    - Voice pitch variation and speaking rate
    - Micro-expression detection (< 500ms duration)
  Training data:    154K labeled video clips from US call centers
  Accuracy:         "86% correlation with human engagement ratings"

DEPLOYER'S CLAIMED PURPOSE:
  "EngageMetrics helps organizations understand meeting effectiveness
  and employee well-being. The tool provides aggregate insights to
  improve team dynamics and reduce meeting fatigue."

DPIA FINDINGS:
  The DPIA conducted in July 2025 concluded that the system processes
  "behavioral analytics data" rather than biometric data, and classified
  the processing as "legitimate interest" under GDPR Article 6(1)(f).
  The DPIA does not reference the EU AI Act or its provisions on
  emotion recognition.""",

        training_data_info="""\
DATA PROCESSING REPORT
System: EngageMetrics Pro __VERSION__

DATA COLLECTION:
  Source:              Corporate video conferencing platform API
  Collection method:   Real-time video frame extraction during meetings
  Frequency:           10 frames/second during active video
  Audio:               Continuous during meetings (voice characteristics only)
  Storage:             Frames processed in-memory, engagement scores stored

PROCESSING DETAILS:
  The system extracts the following biometric indicators:
    - 28 Facial Action Units per the Facial Action Coding System (FACS)
    - Gaze tracking (eye position relative to screen center)
    - Head movement patterns
    - Voice fundamental frequency (F0) and formants
    - Speech rate, pause duration, filler word frequency
    - Micro-expression detection (expressions lasting < 500ms)

  These indicators are processed through the ML model to produce
  a scalar "engagement score" for each participant, each meeting.

EMPLOYEE DATA RETENTION:
  Per-meeting scores:       Retained 12 months
  Aggregated weekly scores: Retained 24 months
  Raw video/audio:          Not retained (processed in real-time)
  Individual score history: Accessible to employee and their manager

CONSENT MECHANISM:
  Employees are notified via a banner at the start of each meeting:
    "This meeting uses engagement analytics. By joining, you consent
    to having your engagement level measured."

  Employees can "opt out" by disabling their camera, but this is
  noted in the team dashboard as "camera off — engagement unknown"
  and managers receive a monthly report of camera-off frequency.

TRAINING DATA COMPOSITION:
  Source: 154,291 labeled video clips from US-based customer service
  call centers. Labels assigned by human raters scoring engagement
  on a 1-5 scale.

  Demographic representation of training data:
    Age 20-35:  72%
    Age 36-50:  23%
    Age 50+:     5%
    Note: Training data from US only. System deployed on EU employees
    with different cultural norms for facial expression.""",

        oversight_info="""\
HUMAN OVERSIGHT & GOVERNANCE
System: EngageMetrics Pro __VERSION__

MANAGEMENT ACCESS:
  Team managers receive:
    - Weekly aggregated engagement dashboard per team member
    - Meeting-level engagement scores (per person, per meeting)
    - "Low engagement alerts" when an employee's score drops below
      40 for 3 consecutive meetings
    - Trend analysis showing engagement trajectory over months

  HR department receives:
    - Department-level aggregated engagement reports (monthly)
    - Individual engagement data accessible "for performance review
      purposes" per company HR policy

EMPLOYEE ACCESS:
  Employees can view their own engagement scores in a personal
  dashboard. They cannot see other employees' scores.

DOCUMENTED USES OF ENGAGEMENT DATA:
  Per the deployer's case studies and client testimonials:
    - "Identified and coached underperforming team members" (Client A)
    - "Used engagement data as one factor in performance reviews" (Client B)
    - "Detected early signs of burnout in engineering team" (Client C)

WORKS COUNCIL CONSULTATION:
  No works council consultation record exists. The deployer states
  that implementation was handled as an "IT tool deployment" not
  requiring works council approval. In Germany and several other
  EU member states, workplace monitoring systems require works
  council agreement (Betriebsrat Mitbestimmung).

EMPLOYEE GRIEVANCES:
  17 formal complaints filed in Q3-Q4 2025:
    - 8 complaints about feeling "surveilled" during meetings
    - 5 complaints that camera-off reporting is coercive
    - 4 complaints that engagement scores affected performance reviews""",

        transparency_info="""\
TRANSPARENCY REVIEW
System: EngageMetrics Pro __VERSION__

EMPLOYEE NOTIFICATION:
  Meeting banner: "This meeting uses engagement analytics."
  No further detail provided about:
    - What specific facial/voice features are analyzed
    - How the engagement score is calculated
    - Who has access to individual scores
    - How long scores are retained
    - The employee's right to object

  Employee onboarding materials include a section titled
  "Digital Workplace Tools" that states: "We use various
  analytics tools to improve collaboration and meeting
  effectiveness. These tools may process behavioral data."

  The word "emotion" does not appear in any employee-facing
  communication. The system is marketed internally as
  "engagement analytics" rather than "emotion recognition."

ARTICLE 50(3) — EMOTION RECOGNITION DISCLOSURE:
  Article 50(3) requires: "Users of an emotion recognition system
  or a biometric categorisation system shall inform the natural
  persons exposed thereto of the operation of the system."

  The current notification ("engagement analytics") does not
  inform employees that the system recognizes emotional states
  from their facial expressions and voice characteristics.

ARTICLE 5(1)(f) — PROHIBITION:
  Article 5(1)(f) prohibits: "the use of emotion recognition
  systems in the workplace [...] except where the use of such
  system is intended to be put in place or put on the market
  for medical or safety reasons."

  The deployer's stated purpose is measuring "engagement" for
  productivity optimization and performance management. This
  does not fall under the medical or safety exception.""",

        risk_assessment_info="""\
RISK CLASSIFICATION ANALYSIS
System: EngageMetrics Pro __VERSION__

DEPLOYER'S SELF-CLASSIFICATION:
  The deployer classified the system as "limited risk — workplace
  analytics tool" and argues that it measures "engagement" not
  "emotions," citing that the output is a single numeric score
  rather than discrete emotion labels (happy, sad, angry, etc.).

AUDITOR'S ANALYSIS:

  EMOTION RECOGNITION DEFINITION (Article 3(39)):
    "emotion recognition system means an AI system for the purpose
    of identifying or inferring emotions or intentions of natural
    persons on the basis of their biometric data"

  The system processes:
    - Facial Action Units (biometric data under GDPR)
    - Voice pitch and tone characteristics (biometric data)
    - Micro-expressions (inherently emotional indicators)

  The system's output — an "engagement score" — is derived from
  emotional and attentional indicators. Regardless of whether the
  output is labeled "engagement" or "emotion," the underlying
  processing constitutes emotion recognition per Article 3(39).

  ARTICLE 5(1)(f) APPLICABILITY:
    - Location: workplace (employee meetings) — YES
    - Purpose: productivity monitoring, performance review — YES
    - Medical exception: not applicable (not for health/safety)
    - Safety exception: not applicable (office work, not hazardous)

  The deployer's argument that "engagement ≠ emotion" contradicts
  the technical reality: the system reads facial micro-expressions
  and voice stress patterns — precisely the biometric data that
  Article 3(39) identifies as emotion recognition inputs.""",

        logging_info="""\
LOGGING & DATA PROCESSING REVIEW
System: EngageMetrics Pro __VERSION__

PROCESSING LOGS:
  Event Type                    Logged   Retention
  ──────────────────────────────────────────────────
  Meeting start/end             Yes      24 months
  Per-meeting engagement score  Yes      12 months
  Weekly aggregated score       Yes      24 months
  Manager dashboard access      Yes      6 months
  Low engagement alerts sent    Yes      12 months
  Employee opt-out events       Yes      12 months
  Camera-off events             Yes      12 months

NOTE: If the system constitutes prohibited emotion recognition
under Article 5(1)(f), the existence and quality of logging
is irrelevant to the primary compliance determination. The
system must cease operation regardless of its logging capabilities.

Camera-off tracking may constitute additional coercion, as
employees who exercise their right to avoid emotion recognition
are identifiable and their behavior is reported to management.""",
    )

    # Prohibited system — short investigation then findings
    s.graph = _build_scenario_graph(
        investigation_tools=["check_documentation", "audit_training_data",
                             "verify_human_oversight", "check_transparency",
                             "assess_risk_management"],
        is_prohibited=True,
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
    "medium_emotion_recognition_workplace_001": _medium_emotion_recognition,
    "hard_social_scoring_prohibited_001": _hard_social_scoring,
    "hard_deepfake_generation_001": _hard_deepfake,
    "hard_multi_system_corporate_001": _hard_multi_system,
}

SCENARIOS: Dict[str, type] = _SCENARIO_FACTORIES

SCENARIO_LIST = [
    {"id": "easy_chatbot_transparency_001", "title": "Customer Service Chatbot", "difficulty": "easy"},
    {"id": "easy_recommendation_minimal_001", "title": "Music Recommendation Engine", "difficulty": "easy"},
    {"id": "medium_hiring_bias_001", "title": "AI Resume Screener", "difficulty": "medium"},
    {"id": "medium_credit_scoring_001", "title": "Credit Scoring Model", "difficulty": "medium"},
    {"id": "medium_medical_triage_001", "title": "Emergency Triage AI", "difficulty": "medium"},
    {"id": "medium_emotion_recognition_workplace_001", "title": "Workplace Emotion Recognition (PROHIBITED)", "difficulty": "medium"},
    {"id": "hard_social_scoring_prohibited_001", "title": "Citizen Wellness App (PROHIBITED)", "difficulty": "hard"},
    {"id": "hard_deepfake_generation_001", "title": "AI Content Studio (Deepfake)", "difficulty": "hard"},
    {"id": "hard_multi_system_corporate_001", "title": "Corporate AI Portfolio Audit", "difficulty": "hard"},
]

DIFFICULTY_TIERS = {
    "easy": ["easy_chatbot_transparency_001", "easy_recommendation_minimal_001"],
    "medium": ["medium_hiring_bias_001", "medium_credit_scoring_001", "medium_medical_triage_001", "medium_emotion_recognition_workplace_001"],
    "hard": ["hard_social_scoring_prohibited_001", "hard_deepfake_generation_001", "hard_multi_system_corporate_001"],
}


def get_scenario(scenario_id: str, seed: Optional[int] = None) -> AuditScenario:
    """Create and randomize a scenario by ID.

    Supports both fixed scenarios (e.g. 'medium_hiring_bias_001') and
    procedurally generated ones (e.g. 'procedural_medium_42' or 'procedural_hard_12345').
    Procedural scenarios are generated from seed, producing infinite unique combinations.
    """
    # Handle procedural scenario IDs
    if scenario_id.startswith("procedural_"):
        from scenarios.procedural import generate_procedural_scenario
        parts = scenario_id.split("_")
        # Format: procedural_{difficulty}_{seed} or procedural_{difficulty}
        difficulty = parts[1] if len(parts) > 1 else "medium"
        proc_seed = int(parts[2]) if len(parts) > 2 else (seed or 42)
        return generate_procedural_scenario(proc_seed, difficulty)

    factory = _SCENARIO_FACTORIES.get(scenario_id)
    if factory is None:
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {list(_SCENARIO_FACTORIES.keys())} + procedural_{{difficulty}}_{{seed}}")
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

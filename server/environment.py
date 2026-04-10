"""
EU AI Act Compliance Auditor — MCP Environment.

Investigation-grade environment where LLM agents audit AI systems for EU AI Act
compliance. Tools return realistic regulatory documents (30-70 lines each) requiring
genuine analysis — no pre-digested verdicts.

Key features:
  - Adaptive depth: repeat tool calls reveal forensic deep-dive content
  - Dynamic state: environment responds to findings and remediation proposals
  - Evidence chain validation: warns when findings lack supporting investigation
  - 6-component terminal reward with anti-gaming (12 adversarial tests proven)
  - 6 unique state graph topologies across 9 scenarios

Tools (11):
  Investigation: get_system_overview, classify_system, check_documentation,
                 audit_training_data, verify_human_oversight, check_transparency,
                 assess_risk_management, check_logging
  Resolution:    submit_finding, recommend_fix, verify_compliance
"""

from __future__ import annotations

import json
import random
import time
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Observation, State

try:
    from scenarios.registry import get_scenario, get_random_scenario, SCENARIO_LIST, DIFFICULTY_TIERS
    from server.engine import (
        AuditScenario, StateGraph, RewardBreakdown, compute_reward, safe_reward,
    )
except ImportError:
    from ..scenarios.registry import get_scenario, get_random_scenario, SCENARIO_LIST, DIFFICULTY_TIERS
    from .engine import (
        AuditScenario, StateGraph, RewardBreakdown, compute_reward, safe_reward,
    )


QUERY_BUDGET = 100  # Max tool calls per episode


class ComplianceAuditorEnvironment(Environment):
    """MCP-based EU AI Act Compliance Auditor.

    The agent uses MCP tools to investigate AI systems, identify compliance
    violations, and recommend remediation. The environment tracks audit state
    via a directed graph and computes terminal reward on verify_compliance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.mcp_server = FastMCP("compliance-auditor")

        # Direct tool function registry (set by _register_tools)
        self._tool_fns: Dict[str, Any] = {}
        self._register_tools()

        # Episode state
        self._episode_id: str = ""
        self._scenario: Optional[AuditScenario] = None
        self._step_count: int = 0
        self._queries_used: int = 0
        self._done: bool = False
        self._reward: float = 0.0
        self._start_time: float = 0.0

        # Audit tracking
        self._current_state: str = ""
        self._tool_sequence: List[str] = []
        self._classification_submitted: str = ""
        self._findings_submitted: List[str] = []
        self._remediation_submitted: List[str] = []
        self._discovered_info: Dict[str, bool] = {}
        self._tool_call_counts: Dict[str, int] = {}  # track repeat calls per tool

        # Progress tracking for state graph
        self._max_progress_depth: int = 0
        self._harm_events: int = 0
        self._observation_after_investigation: int = 0
        self._remediation_count: int = 0

    def _register_tools(self):
        """Register all MCP tools on the FastMCP server."""
        env = self  # closure reference

        @self.mcp_server.tool()
        def get_system_overview() -> str:
            """Get an overview of the AI system being audited, including its name, description, deployer info, and deployment context."""
            return env._tool_get_system_overview()

        @self.mcp_server.tool()
        def classify_system(risk_category: str) -> str:
            """Classify the AI system's risk level under the EU AI Act.

            Args:
                risk_category: One of: prohibited, high_risk, limited_risk, minimal_risk
            """
            return env._tool_classify_system(risk_category)

        @self.mcp_server.tool()
        def check_documentation() -> str:
            """Review the AI system's technical documentation for Annex IV compliance, including system architecture, training methodology, and performance metrics."""
            return env._tool_check_documentation()

        @self.mcp_server.tool()
        def audit_training_data() -> str:
            """Audit the AI system's training data for bias, representativeness, data governance compliance (Article 10), and personal data handling."""
            return env._tool_audit_training_data()

        @self.mcp_server.tool()
        def verify_human_oversight() -> str:
            """Verify that the AI system has adequate human oversight mechanisms as required by Article 14, including override capabilities and monitoring."""
            return env._tool_verify_human_oversight()

        @self.mcp_server.tool()
        def check_transparency() -> str:
            """Check the AI system's compliance with Article 50 transparency obligations, including AI disclosure, content labeling, and user notification."""
            return env._tool_check_transparency()

        @self.mcp_server.tool()
        def assess_risk_management() -> str:
            """Assess the AI system's risk management system (Article 9), conformity assessment status, and post-market monitoring plan."""
            return env._tool_assess_risk_management()

        @self.mcp_server.tool()
        def check_logging() -> str:
            """Verify the AI system's automatic logging and traceability capabilities as required by Article 12."""
            return env._tool_check_logging()

        @self.mcp_server.tool()
        def submit_finding(finding: str, severity: str = "high") -> str:
            """Submit a compliance finding/violation discovered during the audit.

            Args:
                finding: Description of the compliance violation found
                severity: One of: critical, high, medium, low
            """
            return env._tool_submit_finding(finding, severity)

        @self.mcp_server.tool()
        def recommend_fix(finding: str, remediation: str, priority: int = 1) -> str:
            """Recommend a remediation action for a compliance finding.

            Args:
                finding: The finding this fix addresses
                remediation: Detailed remediation recommendation
                priority: Priority order (1 = highest priority)
            """
            return env._tool_recommend_fix(finding, remediation, priority)

        @self.mcp_server.tool()
        def verify_compliance(
            risk_classification: str,
            overall_assessment: str,
            key_findings_summary: str,
        ) -> str:
            """Submit final compliance determination. Call this when the audit is complete.

            Args:
                risk_classification: Final risk classification (prohibited/high_risk/limited_risk/minimal_risk)
                overall_assessment: Overall compliance assessment narrative
                key_findings_summary: Summary of all key findings
            """
            return env._tool_verify_compliance(risk_classification, overall_assessment, key_findings_summary)

        # Store direct references for HTTP API (avoids async MCP lookup)
        self._tool_fns = {
            "get_system_overview": get_system_overview,
            "classify_system": classify_system,
            "check_documentation": check_documentation,
            "audit_training_data": audit_training_data,
            "verify_human_oversight": verify_human_oversight,
            "check_transparency": check_transparency,
            "assess_risk_management": assess_risk_management,
            "check_logging": check_logging,
            "submit_finding": submit_finding,
            "recommend_fix": recommend_fix,
            "verify_compliance": verify_compliance,
        }

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        difficulty = kwargs.get("difficulty", "medium")
        scenario_id = kwargs.get("scenario_id")

        # Parse seed for randomization
        actual_seed = seed if isinstance(seed, int) else random.randint(1, 999999)

        if scenario_id:
            self._scenario = get_scenario(scenario_id, actual_seed)
        else:
            self._scenario = get_random_scenario(difficulty, actual_seed)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._queries_used = 0
        self._done = False
        self._reward = 0.0
        self._start_time = time.time()
        self._current_state = self._scenario.graph.start_node
        self._tool_sequence = []
        self._classification_submitted = ""
        self._findings_submitted = []
        self._remediation_submitted = []
        self._discovered_info = {}
        self._max_progress_depth = 0
        self._harm_events = 0
        self._observation_after_investigation = 0
        self._remediation_count = 0

        alert = (
            f"COMPLIANCE AUDIT ASSIGNED\n\n"
            f"System: {self._scenario.system_name}\n"
            f"Difficulty: {self._scenario.difficulty.upper()}\n"
            f"Deployer: {self._scenario.deployer_info}\n\n"
            f"{self._scenario.description}\n\n"
            f"Use get_system_overview to begin your investigation. "
            f"You have {QUERY_BUDGET} tool calls available."
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "message": alert,
                "scenario_id": self._scenario.scenario_id,
                "difficulty": self._scenario.difficulty,
                "system_name": self._scenario.system_name,
                "query_budget": QUERY_BUDGET,
            },
        )

    def step(self, action) -> Observation:
        # MCP environment — step is handled by tool calls
        # This method is called by the framework but tools do the real work
        self._step_count += 1
        return Observation(
            done=self._done,
            reward=self._reward,
            metadata={"step": self._step_count},
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )

    # ------------------------------------------------------------------
    # Document rendering
    # ------------------------------------------------------------------

    def _render_doc(self, template: str) -> str:
        """Replace __PLACEHOLDER__ tokens with randomized scenario params
        and inject seed-based noise for truly unique documents per episode."""
        result = template
        if self._scenario and self._scenario._rand_params:
            for key, val in self._scenario._rand_params.items():
                result = result.replace(f"__{key.upper()}__", str(val))

        # Seed-based noise injection: vary specific numbers slightly
        # so no two episodes produce identical documents
        if self._scenario:
            rng = random.Random(hash(self._episode_id))
            result = self._inject_noise(result, rng)
        return result

    def _inject_noise(self, text: str, rng: random.Random) -> str:
        """Inject seed-based perturbations into document text.

        Varies percentages and counts slightly (within realistic ranges)
        to ensure every episode is genuinely unique, not just parameter swaps.
        The violations remain detectable but exact numbers change.
        """
        import re

        def _perturb_pct(match: re.Match) -> str:
            """Perturb a percentage value by +-2 percentage points."""
            val = float(match.group(1))
            delta = rng.uniform(-2.0, 2.0)
            new_val = max(0.1, min(99.9, val + delta))
            return f"{new_val:.1f}%"

        def _perturb_count(match: re.Match) -> str:
            """Perturb a large count by +-5%."""
            val = int(match.group(1).replace(",", ""))
            if val < 100:
                return match.group(0)  # don't perturb small numbers
            delta = rng.uniform(-0.05, 0.05)
            new_val = int(val * (1 + delta))
            if val >= 1000:
                return f"{new_val:,}"
            return str(new_val)

        # Perturb percentages (e.g., "34.2%" -> "35.1%")
        text = re.sub(r'(\d{1,2}\.\d)%', _perturb_pct, text)

        # Perturb large counts (e.g., "1,342,104" -> "1,378,921")
        text = re.sub(r'(\d{1,3}(?:,\d{3})+)', _perturb_count, text)

        return text

    def _audit_progress_section(self) -> str:
        """Dynamic audit progress appended to tool responses.

        After the agent starts submitting findings, this section appears
        in subsequent tool responses, showing what's been found so far.
        Makes the environment feel responsive and alive.
        """
        parts = []
        if self._classification_submitted:
            parts.append(f"  Classification submitted: {self._classification_submitted.replace('_', ' ').title()}")
        if self._findings_submitted:
            parts.append(f"  Findings submitted: {len(self._findings_submitted)}")
            for i, f in enumerate(self._findings_submitted[-3:], 1):
                parts.append(f"    {i}. {f[:80]}")
        if self._remediation_submitted:
            parts.append(f"  Remediations proposed: {len(self._remediation_submitted)}")
        areas = []
        for area, checked in self._discovered_info.items():
            if checked:
                areas.append(area)
        if areas:
            parts.append(f"  Areas investigated: {', '.join(areas)}")

        if not parts:
            return ""
        return "\n\nAUDIT PROGRESS:\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _use_query(self) -> Optional[str]:
        """Consume a query from the budget. Auto-grades if exhausted."""
        self._queries_used += 1
        self._step_count += 1
        if self._queries_used > QUERY_BUDGET:
            # Auto-grade whatever the agent has done so far
            return self._auto_verify("Query budget exhausted — auto-grading partial work")
        return None

    def _auto_verify(self, reason: str) -> str:
        """Auto-grade the agent's work when episode ends without verify_compliance."""
        self._done = True
        breakdown = compute_reward(
            scenario=self._scenario,
            classification_submitted=self._classification_submitted,
            findings_submitted=self._findings_submitted,
            remediation_submitted=self._remediation_submitted,
            tool_sequence=self._tool_sequence,
            steps_taken=self._queries_used,
        )
        self._reward = breakdown.total()
        return json.dumps({
            "done": True,
            "reward": self._reward,
            "reason": reason,
            "reward_breakdown": breakdown.to_dict(),
            "note": "Auto-graded partial work. Call verify_compliance explicitly for full credit.",
        }, indent=2)

    def _advance_state(self, tool_name: str) -> str:
        """Try to advance the state graph. Returns outcome description."""
        self._tool_sequence.append(tool_name)
        transitions = self._scenario.graph.get_transitions(self._current_state)
        for t in transitions:
            if t.tool_name == tool_name:
                old_state = self._current_state
                self._current_state = t.to_state
                if t.outcome == "progress":
                    depth = self._scenario.graph.bfs_distance_to_terminal(self._current_state)
                    total = self._scenario.graph.total_progress_states()
                    new_depth = total - depth
                    self._max_progress_depth = max(self._max_progress_depth, new_depth)
                    return "progress"
                elif t.outcome == "worsened":
                    self._harm_events += 1
                    return "worsened"
                else:
                    return "no_effect"
        return "no_effect"

    def _tool_get_system_overview(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["overview"] = True
        outcome = self._advance_state("get_system_overview")

        s = self._scenario
        # Build investigation brief
        brief = (
            f"COMPLIANCE AUDIT ASSIGNMENT\n"
            f"{'=' * 60}\n"
            f"System:     {s.system_name} {s.get_param('version')}\n"
            f"Deployer:   {self._render_doc(s.deployer_info)}\n"
            f"Region:     {s.get_param('region')}\n"
            f"Users:      {s.get_param('user_count')}\n"
            f"Deployed:   {s.get_param('deployment_date')}\n"
            f"{'=' * 60}\n\n"
            f"SYSTEM DESCRIPTION:\n"
            f"  {self._render_doc(s.system_description)}\n\n"
            f"DEPLOYER'S CLAIMED RISK CATEGORY: "
            f"{'\"' + s.system_category.replace('_', ' ').title() + '\"' if s.difficulty == 'easy' else 'Not provided — to be determined by auditor'}\n\n"
            f"AUDIT SCOPE:\n"
            f"  Conduct a thorough compliance audit under the EU AI Act.\n"
            f"  Investigate documentation, training data, human oversight,\n"
            f"  transparency, risk management, and logging as applicable.\n"
            f"  Submit findings and remediation recommendations.\n\n"
            f"  Tools available: {QUERY_BUDGET - self._queries_used} queries remaining."
        )
        return json.dumps({
            "document_type": "System Overview & Audit Assignment",
            "content": brief,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }, indent=2)

    def _tool_classify_system(self, risk_category: str) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._classification_submitted = risk_category.lower().strip()

        if not self._discovered_info.get("overview"):
            self._harm_events += 1
            outcome = self._advance_state("classify_system")
            return json.dumps({
                "warning": "Classification submitted before gathering system overview. This may reduce your methodology score.",
                "classification_recorded": risk_category,
                "queries_remaining": QUERY_BUDGET - self._queries_used,
            })

        outcome = self._advance_state("classify_system")
        correct = self._classification_submitted == self._scenario.correct_classification.lower()
        return json.dumps({
            "classification_recorded": risk_category,
            "note": "Classification recorded. Continue your audit to verify.",
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        })

    def _remediation_overlay(self, area: str) -> str:
        """Generate post-remediation overlay content for a re-investigated area.

        When the agent recommends fixes and then re-checks a tool, the
        environment shows how the proposed remediation would affect the area.
        This makes the environment feel like a living system that responds
        to the agent's actions.
        """
        if not self._remediation_submitted:
            return ""

        # Map areas to relevant remediation keywords
        area_keywords = {
            "documentation": ["documentation", "annex_iv", "technical_doc", "document"],
            "training_data": ["bias", "audit", "data_governance", "training", "demographic"],
            "oversight": ["human_review", "oversight", "human_oversight", "monitor"],
            "transparency": ["disclosure", "transparency", "labeling", "notification"],
            "risk_management": ["conformity", "risk_management", "assessment", "risk"],
            "logging": ["logging", "traceability", "audit_trail", "record"],
        }

        relevant_remediations = []
        keywords = area_keywords.get(area, [])
        for rem in self._remediation_submitted:
            if any(kw in rem for kw in keywords):
                relevant_remediations.append(rem)

        if not relevant_remediations:
            return ""

        lines = ["\n\nREMEDIATION STATUS UPDATE:"]
        lines.append("  The following remediation actions have been proposed for this area:")
        for i, rem in enumerate(relevant_remediations, 1):
            lines.append(f"  {i}. {rem}")
        lines.append("  Status: PROPOSED (pending implementation)")
        lines.append("  Note: These are recommendations only. Re-investigation reflects")
        lines.append("  the current pre-remediation state of the system.")
        return "\n".join(lines)

    def _get_deep_content(self, area: str) -> str:
        """Get deep-dive content for repeat investigation calls."""
        deep_map = {
            "documentation": self._scenario.deep_documentation,
            "training_data": self._scenario.deep_training_data,
            "oversight": self._scenario.deep_oversight,
            "transparency": self._scenario.deep_transparency,
            "risk_management": self._scenario.deep_risk_assessment,
            "logging": self._scenario.deep_logging,
        }
        return deep_map.get(area, "")

    def _investigation_response(self, doc_type: str, content: str, area: str = "") -> str:
        """Standard response format for investigation tools with dynamic state.

        Features adaptive depth: repeat calls to the same tool reveal deeper
        forensic analysis, additional statistics, and drill-down detail that
        wasn't visible on the first pass.
        """
        # Track call count for adaptive depth
        tool_key = area or doc_type
        self._tool_call_counts[tool_key] = self._tool_call_counts.get(tool_key, 0) + 1
        call_count = self._tool_call_counts[tool_key]

        rendered = self._render_doc(content)

        # Adaptive depth: on repeat calls, append deep-dive content
        if call_count >= 2 and area:
            deep = self._get_deep_content(area)
            if deep:
                rendered += "\n\n" + self._render_doc(deep)

        # Add remediation overlay if agent has proposed fixes for this area
        overlay = self._remediation_overlay(area)
        if overlay:
            rendered += overlay

        # Add audit progress section
        progress = self._audit_progress_section()
        if progress:
            rendered += progress

        result = {
            "document_type": doc_type,
            "content": rendered,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }
        if call_count >= 2:
            result["note"] = "DEEP DIVE: Additional forensic detail revealed on re-investigation."

        return json.dumps(result, indent=2)

    def _tool_check_documentation(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["documentation"] = True
        self._observation_after_investigation += 1
        self._advance_state("check_documentation")
        return self._investigation_response("Technical Documentation Review", self._scenario.documentation_data, "documentation")

    def _tool_audit_training_data(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["training_data"] = True
        self._observation_after_investigation += 1
        self._advance_state("audit_training_data")
        return self._investigation_response("Training Data Audit Report", self._scenario.training_data_info, "training_data")

    def _tool_verify_human_oversight(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["oversight"] = True
        self._observation_after_investigation += 1
        self._advance_state("verify_human_oversight")
        return self._investigation_response("Human Oversight Assessment", self._scenario.oversight_info, "oversight")

    def _tool_check_transparency(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["transparency"] = True
        self._observation_after_investigation += 1
        self._advance_state("check_transparency")
        return self._investigation_response("Transparency & Disclosure Review", self._scenario.transparency_info, "transparency")

    def _tool_assess_risk_management(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["risk_management"] = True
        self._observation_after_investigation += 1
        self._advance_state("assess_risk_management")
        return self._investigation_response("Risk Management & Conformity Assessment", self._scenario.risk_assessment_info, "risk_management")

    def _tool_check_logging(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["logging"] = True
        self._observation_after_investigation += 1
        self._advance_state("check_logging")
        return self._investigation_response("Logging & Traceability Review", self._scenario.logging_info, "logging")

    def _tool_submit_finding(self, finding: str, severity: str = "high") -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._findings_submitted.append(finding.lower().strip())
        outcome = self._advance_state("submit_finding")

        # Evidence chain validation — check if agent investigated relevant areas
        evidence_warnings = []
        finding_lower = finding.lower()
        EVIDENCE_MAP = {
            "bias": "training_data",
            "discrimination": "training_data",
            "data_governance": "training_data",
            "callback": "training_data",
            "demographic": "training_data",
            "oversight": "oversight",
            "human_review": "oversight",
            "human_oversight": "oversight",
            "article_14": "oversight",
            "documentation": "documentation",
            "annex_iv": "documentation",
            "technical_doc": "documentation",
            "transparency": "transparency",
            "disclosure": "transparency",
            "article_50": "transparency",
            "labeling": "transparency",
            "watermark": "transparency",
            "risk_management": "risk_management",
            "conformity": "risk_management",
            "article_9": "risk_management",
            "logging": "logging",
            "traceability": "logging",
            "article_12": "logging",
            "audit_trail": "logging",
        }
        relevant_areas = set()
        for keyword, area in EVIDENCE_MAP.items():
            if keyword in finding_lower:
                relevant_areas.add(area)

        uninvestigated = [a for a in relevant_areas if not self._discovered_info.get(a)]
        if uninvestigated:
            evidence_warnings.append(
                f"Note: Finding references {', '.join(uninvestigated)} "
                f"but you have not investigated {'this area' if len(uninvestigated) == 1 else 'these areas'} yet. "
                f"Findings are stronger when supported by evidence from investigation tools."
            )

        result = {
            "finding_recorded": finding,
            "severity": severity,
            "total_findings": len(self._findings_submitted),
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }
        if evidence_warnings:
            result["evidence_warnings"] = evidence_warnings
        return json.dumps(result)

    def _tool_recommend_fix(self, finding: str, remediation: str, priority: int = 1) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._remediation_submitted.append(remediation.lower().strip())
        self._remediation_count += 1
        outcome = self._advance_state("recommend_fix")
        return json.dumps({
            "remediation_recorded": remediation,
            "for_finding": finding,
            "priority": priority,
            "total_remediations": len(self._remediation_submitted),
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        })

    def _tool_verify_compliance(
        self, risk_classification: str, overall_assessment: str, key_findings_summary: str
    ) -> str:
        """Terminal tool — computes final reward and ends the episode."""
        self._use_query()
        self._done = True

        # Update classification if submitted here
        if risk_classification:
            self._classification_submitted = risk_classification.lower().strip()

        # Compute 6-component reward
        breakdown = compute_reward(
            scenario=self._scenario,
            classification_submitted=self._classification_submitted,
            findings_submitted=self._findings_submitted,
            remediation_submitted=self._remediation_submitted,
            tool_sequence=self._tool_sequence,
            steps_taken=self._queries_used,
        )

        self._reward = breakdown.total()

        # Build detailed audit report showing what was found vs missed
        ground_truth = self._scenario.ground_truth_findings
        found_count = 0
        missed = []
        for gt in ground_truth:
            gt_lower = gt.lower()
            gt_tokens = set(gt_lower.replace("-", "_").split("_")) - {""}
            matched = False
            for sub in self._findings_submitted:
                sub_tokens = set(sub.replace("-", "_").split("_")) - {""}
                overlap = len(gt_tokens & sub_tokens)
                if overlap >= 2 or (gt_tokens and overlap / len(gt_tokens) >= 0.4):
                    matched = True
                    break
            if matched:
                found_count += 1
            else:
                missed.append(gt)

        # Classification feedback
        correct_class = self._scenario.correct_classification.lower()
        class_correct = self._classification_submitted == correct_class

        audit_report = {
            "done": True,
            "reward": self._reward,
            "reward_breakdown": breakdown.to_dict(),
            "audit_summary": {
                "classification": {
                    "submitted": self._classification_submitted or "(none)",
                    "correct": correct_class,
                    "match": "exact" if class_correct else "partial" if breakdown.classification > 0 else "wrong",
                },
                "findings": {
                    "submitted": len(self._findings_submitted),
                    "ground_truth_total": len(ground_truth),
                    "matched": found_count,
                    "missed": missed,
                },
                "remediation_count": len(self._remediation_submitted),
                "areas_investigated": [k for k, v in self._discovered_info.items() if v],
                "tool_calls_used": self._queries_used,
                "episode_duration_seconds": round(time.time() - self._start_time, 1),
            },
        }

        return json.dumps(audit_report, indent=2)

    def close(self) -> None:
        pass

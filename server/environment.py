"""
EU AI Act Compliance Auditor — MCP Environment.

Registers 10 MCP tools that the agent uses to audit AI systems for EU AI Act
compliance. State-graph tracks audit progress. Terminal reward computed on
verify_compliance with 6-component scoring.

Tools:
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
    # Tool implementations
    # ------------------------------------------------------------------

    def _use_query(self) -> Optional[str]:
        """Consume a query from the budget. Returns error string if exhausted."""
        self._queries_used += 1
        self._step_count += 1
        if self._queries_used > QUERY_BUDGET:
            self._done = True
            self._reward = safe_reward(0.0)
            return json.dumps({
                "error": "Query budget exhausted",
                "done": True,
                "reward": self._reward,
            })
        return None

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
        result = {
            "system_name": s.system_name,
            "description": s.system_description,
            "deployer": s.deployer_info,
            "category_claim": s.system_category if s.difficulty == "easy" else "To be determined by auditor",
            "deployment_date": s.get_param("deployment_date"),
            "region": s.get_param("region"),
            "user_count": s.get_param("user_count"),
            "company": s.get_param("company"),
            "version": s.get_param("version"),
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }
        return json.dumps(result, indent=2)

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

    def _tool_check_documentation(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["documentation"] = True
        self._observation_after_investigation += 1
        outcome = self._advance_state("check_documentation")
        return json.dumps({
            "documentation_review": self._scenario.documentation_data,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }, indent=2)

    def _tool_audit_training_data(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["training_data"] = True
        self._observation_after_investigation += 1
        outcome = self._advance_state("audit_training_data")
        return json.dumps({
            "training_data_audit": self._scenario.training_data_info,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }, indent=2)

    def _tool_verify_human_oversight(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["oversight"] = True
        self._observation_after_investigation += 1
        outcome = self._advance_state("verify_human_oversight")
        return json.dumps({
            "oversight_assessment": self._scenario.oversight_info,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }, indent=2)

    def _tool_check_transparency(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["transparency"] = True
        self._observation_after_investigation += 1
        outcome = self._advance_state("check_transparency")
        return json.dumps({
            "transparency_assessment": self._scenario.transparency_info,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }, indent=2)

    def _tool_assess_risk_management(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["risk_management"] = True
        self._observation_after_investigation += 1
        outcome = self._advance_state("assess_risk_management")
        return json.dumps({
            "risk_assessment": self._scenario.risk_assessment_info,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }, indent=2)

    def _tool_check_logging(self) -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._discovered_info["logging"] = True
        self._observation_after_investigation += 1
        outcome = self._advance_state("check_logging")
        return json.dumps({
            "logging_assessment": self._scenario.logging_info,
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        }, indent=2)

    def _tool_submit_finding(self, finding: str, severity: str = "high") -> str:
        budget_err = self._use_query()
        if budget_err:
            return budget_err
        self._findings_submitted.append(finding.lower().strip())
        outcome = self._advance_state("submit_finding")
        return json.dumps({
            "finding_recorded": finding,
            "severity": severity,
            "total_findings": len(self._findings_submitted),
            "queries_remaining": QUERY_BUDGET - self._queries_used,
        })

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

        return json.dumps({
            "done": True,
            "reward": self._reward,
            "assessment_recorded": overall_assessment[:200],
            "reward_breakdown": breakdown.to_dict(),
            "findings_submitted": len(self._findings_submitted),
            "remediations_submitted": len(self._remediation_submitted),
            "queries_used": self._queries_used,
            "episode_duration_seconds": round(time.time() - self._start_time, 1),
        }, indent=2)

    def close(self) -> None:
        pass

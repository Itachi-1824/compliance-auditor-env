"""
State-graph engine for the EU AI Act Compliance Auditor.

Each scenario is a directed graph where:
  - Nodes represent audit states (e.g., INITIAL, CLASSIFYING, AUDITING_DATA)
  - Edges represent tool calls with outcomes: progress / no_effect / worsened
  - BFS depth from current node to RESOLVED gives partial credit
  - Wrong actions can push the audit backward (worsened transitions)
  - Parameter randomization prevents memorization

Reward is computed from 6 components:
  1. Classification accuracy (20%) — correct risk category
  2. Finding completeness (25%) — found X of Y violations
  3. Finding precision (15%) — penalty for false positives
  4. Remediation quality (15%) — correct priority ordering
  5. Process methodology (15%) — followed correct audit sequence
  6. Efficiency (10%) — steps vs optimal path
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# State graph primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateNode:
    """A node in the audit state graph."""
    id: str
    label: str
    is_terminal: bool = False
    is_start: bool = False


@dataclass(frozen=True)
class Transition:
    """An edge in the audit state graph."""
    from_state: str
    to_state: str
    tool_name: str
    outcome: str  # "progress" | "no_effect" | "worsened"
    required_args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class StateGraph:
    """Directed graph of audit states with BFS-based partial credit."""

    def __init__(self):
        self.nodes: Dict[str, StateNode] = {}
        self.transitions: List[Transition] = []
        self._adjacency: Dict[str, List[Transition]] = {}
        self._start_node: Optional[str] = None
        self._terminal_nodes: Set[str] = set()

    def add_node(self, node: StateNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []
        if node.is_start:
            self._start_node = node.id
        if node.is_terminal:
            self._terminal_nodes.add(node.id)

    def add_transition(self, t: Transition) -> None:
        self.transitions.append(t)
        if t.from_state not in self._adjacency:
            self._adjacency[t.from_state] = []
        self._adjacency[t.from_state].append(t)

    @property
    def start_node(self) -> str:
        if self._start_node is None:
            raise ValueError("No start node defined")
        return self._start_node

    def get_transitions(self, state_id: str) -> List[Transition]:
        return self._adjacency.get(state_id, [])

    def get_progress_transitions(self, state_id: str) -> List[Transition]:
        return [t for t in self.get_transitions(state_id) if t.outcome == "progress"]

    def bfs_distance_to_terminal(self, state_id: str) -> int:
        """BFS shortest path from state_id to any terminal node."""
        if state_id in self._terminal_nodes:
            return 0
        visited = {state_id}
        queue = deque([(state_id, 0)])
        while queue:
            current, dist = queue.popleft()
            for t in self.get_transitions(current):
                if t.outcome == "progress" and t.to_state not in visited:
                    if t.to_state in self._terminal_nodes:
                        return dist + 1
                    visited.add(t.to_state)
                    queue.append((t.to_state, dist + 1))
        return 999  # unreachable

    def optimal_path_length(self) -> int:
        """Minimum steps from start to any terminal."""
        return self.bfs_distance_to_terminal(self.start_node)

    def total_progress_states(self) -> int:
        """Total number of non-terminal states reachable via progress transitions."""
        visited = set()
        queue = deque([self.start_node])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for t in self.get_transitions(current):
                if t.outcome == "progress":
                    queue.append(t.to_state)
        return len(visited)


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class AuditScenario:
    """A complete compliance audit scenario with state graph and ground truth."""

    scenario_id: str
    title: str
    difficulty: str  # easy | medium | hard
    description: str  # initial alert/assignment text

    # The AI system being audited
    system_name: str
    system_description: str
    system_category: str  # prohibited | high_risk | limited_risk | minimal_risk
    deployer_info: str

    # State graph
    graph: StateGraph = field(default_factory=StateGraph)

    # Ground truth for grading
    correct_classification: str = ""  # prohibited | high_risk | limited_risk | minimal_risk
    ground_truth_findings: List[str] = field(default_factory=list)
    required_remediation: List[str] = field(default_factory=list)
    red_herrings: List[str] = field(default_factory=list)

    # Tool-specific data (returned when agent calls tools)
    documentation_data: Dict[str, Any] = field(default_factory=dict)
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    oversight_info: Dict[str, Any] = field(default_factory=dict)
    transparency_info: Dict[str, Any] = field(default_factory=dict)
    risk_assessment_info: Dict[str, Any] = field(default_factory=dict)
    logging_info: Dict[str, Any] = field(default_factory=dict)

    # Randomization parameters (re-rolled on each reset)
    _rand_params: Dict[str, str] = field(default_factory=dict)

    def randomize(self, seed: Optional[int] = None) -> None:
        """Re-roll randomizable parameters to prevent memorization."""
        rng = random.Random(seed)
        company_names = [
            "TechNova Solutions", "QuantumLeap AI", "NeuralPath Inc",
            "DataForge Systems", "CogniTech Labs", "AlphaWave AI",
            "SynthMind Corp", "PrismAI Technologies", "Vertex Analytics",
            "OmniSense AI", "DeepCurrent Inc", "StrataLogic Systems",
        ]
        regions = ["EU-West", "EU-Central", "EU-North", "EU-South", "EU-East"]
        versions = ["v2.1", "v3.0", "v4.2", "v5.1", "v1.8", "v6.0"]

        self._rand_params = {
            "company": rng.choice(company_names),
            "region": rng.choice(regions),
            "version": rng.choice(versions),
            "deployment_date": f"2026-{rng.randint(1,3):02d}-{rng.randint(1,28):02d}",
            "user_count": str(rng.randint(10000, 5000000)),
        }

    def get_param(self, key: str) -> str:
        return self._rand_params.get(key, "Unknown")


# ---------------------------------------------------------------------------
# Reward computation (6 components)
# ---------------------------------------------------------------------------

def safe_reward(score: float) -> float:
    """Clamp reward to (0, 1) exclusive — required by OpenEnv validator."""
    return max(0.01, min(0.99, score))


@dataclass
class RewardBreakdown:
    classification: float = 0.0   # 20%
    finding_completeness: float = 0.0  # 25%
    finding_precision: float = 0.0  # 15%
    remediation: float = 0.0  # 15%
    methodology: float = 0.0  # 15%
    efficiency: float = 0.0  # 10%

    def total(self) -> float:
        raw = (
            self.classification * 0.20
            + self.finding_completeness * 0.25
            + self.finding_precision * 0.15
            + self.remediation * 0.15
            + self.methodology * 0.15
            + self.efficiency * 0.10
        )
        return safe_reward(raw)

    def to_dict(self) -> Dict[str, float]:
        return {
            "classification": round(self.classification, 3),
            "finding_completeness": round(self.finding_completeness, 3),
            "finding_precision": round(self.finding_precision, 3),
            "remediation": round(self.remediation, 3),
            "methodology": round(self.methodology, 3),
            "efficiency": round(self.efficiency, 3),
            "total": round(self.total(), 4),
        }


def compute_reward(
    scenario: AuditScenario,
    classification_submitted: str,
    findings_submitted: List[str],
    remediation_submitted: List[str],
    tool_sequence: List[str],
    steps_taken: int,
) -> RewardBreakdown:
    """Compute the 6-component reward for a completed audit."""

    breakdown = RewardBreakdown()

    # 1. Classification accuracy (20%)
    if classification_submitted.lower().strip() == scenario.correct_classification.lower():
        breakdown.classification = 1.0
    elif _partial_classification_match(classification_submitted, scenario.correct_classification):
        breakdown.classification = 0.4
    else:
        breakdown.classification = 0.0

    # 2. Finding completeness (25%) — recall of ground truth findings
    if scenario.ground_truth_findings:
        found = set(f.lower().strip() for f in findings_submitted)
        truth = set(f.lower() for f in scenario.ground_truth_findings)
        matches = sum(1 for t in truth if any(t in f or f in t for f in found))
        breakdown.finding_completeness = matches / len(truth)
    else:
        breakdown.finding_completeness = 1.0  # no findings expected

    # 3. Finding precision (15%) — penalize false positives
    if findings_submitted:
        found = set(f.lower().strip() for f in findings_submitted)
        truth = set(f.lower() for f in scenario.ground_truth_findings)
        red = set(r.lower() for r in scenario.red_herrings)
        true_positives = sum(1 for f in found if any(t in f or f in t for t in truth))
        false_positives = sum(1 for f in found if any(r in f or f in r for r in red))
        total = len(found)
        if total > 0:
            precision = true_positives / total
            red_herring_penalty = false_positives * 0.15
            breakdown.finding_precision = max(0.0, precision - red_herring_penalty)
        else:
            breakdown.finding_precision = 0.0
    else:
        breakdown.finding_precision = 0.0

    # 4. Remediation quality (15%) — correct fixes in priority order
    if scenario.required_remediation:
        rem_lower = [r.lower().strip() for r in remediation_submitted]
        req_lower = [r.lower() for r in scenario.required_remediation]
        # Check presence
        matches = sum(1 for req in req_lower if any(req in r or r in req for r in rem_lower))
        presence_score = matches / len(req_lower)
        # Check ordering (bonus if in correct priority)
        order_score = _check_ordering(rem_lower, req_lower)
        breakdown.remediation = presence_score * 0.7 + order_score * 0.3
    else:
        breakdown.remediation = 1.0

    # 5. Process methodology (15%) — correct audit sequence
    expected_sequence = [
        "classify_system", "check_documentation", "audit_training_data",
        "verify_human_oversight", "check_transparency", "assess_risk_management",
    ]
    actual_tools = [t for t in tool_sequence if t in expected_sequence]
    if actual_tools:
        # Score based on how many tools were used in the expected order
        order_violations = 0
        for i in range(len(actual_tools) - 1):
            if actual_tools[i] in expected_sequence and actual_tools[i + 1] in expected_sequence:
                idx_a = expected_sequence.index(actual_tools[i])
                idx_b = expected_sequence.index(actual_tools[i + 1])
                if idx_b < idx_a:
                    order_violations += 1
        coverage = len(set(actual_tools)) / len(expected_sequence)
        order_penalty = min(order_violations * 0.15, 0.5)
        breakdown.methodology = max(0.0, coverage - order_penalty)
    else:
        breakdown.methodology = 0.0

    # 6. Efficiency (10%) — steps vs optimal
    optimal = scenario.graph.optimal_path_length()
    if optimal > 0 and steps_taken > 0:
        ratio = optimal / steps_taken
        breakdown.efficiency = min(ratio, 1.0)
    else:
        breakdown.efficiency = 0.5

    return breakdown


def _partial_classification_match(submitted: str, correct: str) -> bool:
    """Check if classification is partially correct (e.g., high_risk vs limited_risk)."""
    risk_levels = ["prohibited", "high_risk", "limited_risk", "minimal_risk"]
    sub = submitted.lower().strip().replace("-", "_").replace(" ", "_")
    cor = correct.lower().strip()
    if sub not in risk_levels or cor not in risk_levels:
        return False
    return abs(risk_levels.index(sub) - risk_levels.index(cor)) == 1


def _check_ordering(submitted: List[str], required: List[str]) -> float:
    """Score how well submitted items match the required priority order."""
    if not submitted or not required:
        return 0.0
    matched_indices = []
    for req in required:
        for i, sub in enumerate(submitted):
            if req in sub or sub in req:
                matched_indices.append(i)
                break
    if len(matched_indices) < 2:
        return 0.5
    # Check if matched items are in increasing order
    in_order = sum(1 for i in range(len(matched_indices) - 1) if matched_indices[i] < matched_indices[i + 1])
    return in_order / (len(matched_indices) - 1)

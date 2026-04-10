"""
Data models for the EU AI Act Compliance Auditor Environment.

MCP pattern: no custom Action class needed — the framework provides CallToolAction.
We define Observation and State for typed environment responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ComplianceObservation(BaseModel):
    """Observation returned after each environment interaction."""

    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Audit-specific fields
    action_type: str = ""
    message: str = ""
    system_info: Optional[Dict[str, Any]] = None
    documentation_report: Optional[Dict[str, Any]] = None
    data_audit_report: Optional[Dict[str, Any]] = None
    oversight_report: Optional[Dict[str, Any]] = None
    transparency_report: Optional[Dict[str, Any]] = None
    risk_report: Optional[Dict[str, Any]] = None
    logging_report: Optional[Dict[str, Any]] = None
    findings: Optional[List[Dict[str, str]]] = None
    queries_remaining: int = 0
    episode_elapsed_seconds: float = 0.0


class ComplianceState(BaseModel):
    """Episode-level metadata exposed via GET /state."""

    episode_id: str = ""
    step_count: int = 0
    scenario_id: str = ""
    difficulty: str = ""
    system_name: str = ""
    queries_used: int = 0
    query_budget: int = 0
    classification_submitted: bool = False
    findings_count: int = 0
    compliance_verified: bool = False
    current_reward: float = 0.0

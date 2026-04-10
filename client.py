"""
EU AI Act Compliance Auditor — HTTP Client.

HTTP-based client connecting to the HF Space via short HTTP round-trips.
API: reset(), list_tools(), call_tool(). No WebSocket needed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class ComplianceAuditorHTTP:
    """HTTP-based client for the Compliance Auditor environment.

    Each call is a short HTTP round-trip — no WebSocket needed.
    Works reliably through HF Space reverse proxy.
    """

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        if self.base_url.startswith("ws://"):
            self.base_url = self.base_url.replace("ws://", "http://", 1)
        elif self.base_url.startswith("wss://"):
            self.base_url = self.base_url.replace("wss://", "https://", 1)
        self.timeout = timeout
        self._client = None
        self._session_id: Optional[str] = None
        self._tools_cache: Optional[List[Dict]] = None
        self._last_done: bool = False
        self._last_reward: float = 0.0

    async def __aenter__(self):
        import httpx
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._session_id and self._client:
            try:
                await self._client.post(
                    f"{self.base_url}/api/close",
                    json={"session_id": self._session_id},
                )
            except Exception:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None

    async def reset(self, **kwargs) -> Dict[str, Any]:
        """Reset environment and start a new audit session."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=self.timeout)

        resp = await self._client.post(
            f"{self.base_url}/api/reset",
            json={
                "difficulty": kwargs.get("difficulty", "medium"),
                "scenario_id": kwargs.get("scenario_id"),
                "seed": kwargs.get("seed"),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data["session_id"]
        self._tools_cache = data.get("tools", [])
        self._last_done = False
        self._last_reward = 0.0
        return data["observation"]

    async def list_tools(self) -> List[Dict]:
        """Return cached tool list from last reset()."""
        return self._tools_cache or []

    async def call_tool(self, name: str, **kwargs) -> Any:
        """Call an audit tool. Returns the tool's result (JSON string)."""
        if not self._session_id:
            raise RuntimeError("No active session. Call reset() first.")

        resp = await self._client.post(
            f"{self.base_url}/api/call_tool",
            json={
                "session_id": self._session_id,
                "tool_name": name,
                "arguments": kwargs,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._last_done = data.get("done", False)
        self._last_reward = data.get("reward", 0.0)
        return data.get("result", "")

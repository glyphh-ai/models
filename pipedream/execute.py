"""
Pipedream Connect Execution Bridge

Executes matched actions via the Pipedream Connect API.
Called by the runtime when a query routes to DONE with high confidence.

Usage (from runtime):
    from execute import PipedreamExecutor

    executor = PipedreamExecutor()
    result = executor.run_action(
        action_key="slack-send-message-to-channel",
        external_user_id="user-123",
        configured_props={"channel": "#general", "text": "Hello!"},
    )

Requires env vars:
    PIPEDREAM_CLIENT_ID
    PIPEDREAM_CLIENT_SECRET
    PIPEDREAM_PROJECT_ID
    PIPEDREAM_ENVIRONMENT (default: development)
"""

import os
import sys
from typing import Any

import requests

BASE_URL = "https://api.pipedream.com/v1"


class PipedreamExecutor:
    """Executes Pipedream actions via the Connect API."""

    def __init__(self):
        self.client_id = os.environ.get("PIPEDREAM_CLIENT_ID")
        self.client_secret = os.environ.get("PIPEDREAM_CLIENT_SECRET")
        self.project_id = os.environ.get("PIPEDREAM_PROJECT_ID")
        self.environment = os.environ.get("PIPEDREAM_ENVIRONMENT", "development")

        if not all([self.client_id, self.client_secret, self.project_id]):
            raise RuntimeError(
                "Missing Pipedream env vars. "
                "Required: PIPEDREAM_CLIENT_ID, PIPEDREAM_CLIENT_SECRET, PIPEDREAM_PROJECT_ID"
            )

        self._access_token: str | None = None

    def _get_token(self) -> str:
        """Obtain OAuth access token via client credentials."""
        if self._access_token:
            return self._access_token

        resp = requests.post(
            f"{BASE_URL}/oauth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        return self._access_token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "X-PD-Environment": self.environment,
            "Content-Type": "application/json",
        }

    def run_action(
        self,
        action_key: str,
        external_user_id: str,
        configured_props: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict:
        """Execute a Pipedream action via Connect API.

        Args:
            action_key: The Pipedream action key (e.g., "slack-send-message-to-channel")
            external_user_id: The end user's ID in your system
            configured_props: Action parameters (channel, text, etc.)
            timeout: Request timeout in seconds

        Returns:
            {success: bool, data: dict | None, error: str | None}
        """
        payload = {
            "id": action_key,
            "external_user_id": external_user_id,
            "configured_props": configured_props or {},
        }

        try:
            resp = requests.post(
                f"{BASE_URL}/connect/{self.project_id}/actions/run",
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )

            if resp.status_code == 200:
                return {
                    "success": True,
                    "data": resp.json(),
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "data": None,
                    "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
                }
        except requests.Timeout:
            return {
                "success": False,
                "data": None,
                "error": f"Timeout after {timeout}s",
            }
        except requests.RequestException as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }

    def get_action_props(self, action_key: str) -> dict:
        """Get the configurable properties for an action.

        Returns the prop schema so the model can determine what
        parameters are required vs optional for ASK/DONE decisions.

        Returns:
            {props: list[dict], error: str | None}
        """
        try:
            resp = requests.get(
                f"{BASE_URL}/connect/{self.project_id}/actions/{action_key}",
                headers=self._headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "props": data.get("configurable_props", []),
                    "error": None,
                }
            else:
                return {
                    "props": [],
                    "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
                }
        except requests.RequestException as e:
            return {"props": [], "error": str(e)}

    def list_user_accounts(self, external_user_id: str) -> list[dict]:
        """List connected accounts for a user.

        Returns accounts the user has authenticated, so the model
        can determine which apps the user actually has access to.
        """
        try:
            resp = requests.get(
                f"{BASE_URL}/connect/{self.project_id}/accounts",
                headers=self._headers(),
                params={"external_user_id": external_user_id},
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
        except requests.RequestException:
            return []


def validate_props(
    required_props: list[dict],
    provided_props: dict[str, Any],
) -> dict:
    """Validate that all required props are provided.

    Used by the runtime to decide DONE vs ASK:
    - All required props present → DONE (execute)
    - Missing required props → ASK (list what's missing)

    Args:
        required_props: List of prop schemas from get_action_props()
        provided_props: Props extracted from the user's query

    Returns:
        {complete: bool, missing: list[str], provided: list[str]}
    """
    missing = []
    provided = []

    for prop in required_props:
        prop_name = prop.get("name", "")
        is_required = prop.get("required", False) or prop.get("optional") is not True

        if prop_name in provided_props and provided_props[prop_name] is not None:
            provided.append(prop_name)
        elif is_required:
            missing.append(prop_name)

    return {
        "complete": len(missing) == 0,
        "missing": missing,
        "provided": provided,
    }

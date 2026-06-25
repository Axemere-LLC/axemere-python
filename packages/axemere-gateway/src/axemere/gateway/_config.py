"""AiGatewayConfig: connection and attribution settings for the Axemere AI Gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _gateway_token_from_env() -> Optional[str]:
    """Resolve the gateway token from the environment.

    ``AXEMERE_GATEWAY_TOKEN`` is the canonical variable. ``AXEMERE_WORKLOAD_TOKEN``
    is accepted as a fallback so users who configured the older name keep working.
    """
    return (
        os.environ.get("AXEMERE_GATEWAY_TOKEN")
        or os.environ.get("AXEMERE_WORKLOAD_TOKEN")
        or None
    )


@dataclass
class AiGatewayConfig:
    """Connection and attribution settings for the Axemere AI Gateway.

    Every field defaults to its corresponding environment variable, so a bare
    ``AiGatewayConfig()`` is equivalent to :meth:`from_env`.

    Attributes:
        gateway_url: Base URL of the gateway (e.g. ``http://localhost:7080``).
        gateway_token: Bearer token for the gateway. Read from
            ``AXEMERE_GATEWAY_TOKEN``, falling back to ``AXEMERE_WORKLOAD_TOKEN``.
        default_provider: Provider to use when none is specified.
        default_model: Model to use when none is specified.
        workload_id: Workload identifier for attribution.
        project_id: Project identifier for attribution.
        account_id: Account identifier for attribution.
        customer_id: Customer identifier for attribution.
        labels: Arbitrary key-value labels attached to every request.
        provider_api_key: Upstream provider API key (passed through when not
            using a gateway-managed credential).
        timeout: HTTP request timeout in seconds.
    """

    gateway_url: str = field(
        default_factory=lambda: os.environ.get("AXEMERE_GATEWAY_URL", "http://localhost:7080")
    )
    gateway_token: Optional[str] = field(default_factory=_gateway_token_from_env)
    default_provider: Optional[str] = field(
        default_factory=lambda: os.environ.get("AXEMERE_DEFAULT_PROVIDER")
    )
    default_model: Optional[str] = field(
        default_factory=lambda: os.environ.get("AXEMERE_DEFAULT_MODEL")
    )
    workload_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("AXEMERE_WORKLOAD_ID")
    )
    project_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("AXEMERE_PROJECT_ID")
    )
    account_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("AXEMERE_ACCOUNT_ID")
    )
    customer_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("AXEMERE_CUSTOMER_ID")
    )
    labels: Dict[str, str] = field(default_factory=dict)
    provider_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("AXEMERE_PROVIDER_API_KEY")
    )
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("AXEMERE_TIMEOUT", "120"))
    )

    @classmethod
    def from_env(cls) -> "AiGatewayConfig":
        """Build a config from environment variables.

        Equivalent to ``AiGatewayConfig()`` — every field already defaults to its
        environment variable. Retained for explicitness and backwards compatibility.

        Environment variables:
            AXEMERE_GATEWAY_URL: Gateway base URL.
            AXEMERE_GATEWAY_TOKEN: Bearer token (falls back to AXEMERE_WORKLOAD_TOKEN).
            AXEMERE_DEFAULT_PROVIDER: Default provider name.
            AXEMERE_DEFAULT_MODEL: Default model name.
            AXEMERE_WORKLOAD_ID: Workload ID for attribution.
            AXEMERE_PROJECT_ID: Project ID for attribution.
            AXEMERE_ACCOUNT_ID: Account ID for attribution.
            AXEMERE_CUSTOMER_ID: Customer ID for attribution.
            AXEMERE_PROVIDER_API_KEY: Upstream provider API key.
            AXEMERE_TIMEOUT: Request timeout in seconds.
        """
        return cls()

    def set_defaults(self, provider: str, model: str) -> None:
        """Set default provider and model in-place."""
        self.default_provider = provider
        self.default_model = model

    def proxy_url(self, provider: str) -> str:
        """Build the proxy URL for a given provider.

        The URL embeds attribution path segments so that the gateway can
        attribute the request without requiring an explicit action payload.

        .. warning::
            When a gateway token is set it is embedded directly in the URL path
            (``/k/<token>``). URLs are routinely recorded in server access logs,
            reverse-proxy logs, and browser history, so the token can leak to
            anyone with access to those logs. Prefer the header-based flow
            (:meth:`AiGatewayClient.execute` / :meth:`AiGatewayClient.execute_sync`,
            which send the token in the ``Authorization`` header) whenever the
            client supports it, and treat tokens used with ``proxy_url`` as
            sensitive credentials that should be rotated if logs are exposed.
        """
        url = f"{self.gateway_url}/proxy/{provider}"
        if self.gateway_token:
            url += f"/k/{self.gateway_token}"
        if self.workload_id:
            url += f"/w/{self.workload_id}"
        if self.project_id:
            url += f"/p/{self.project_id}"
        if self.account_id:
            url += f"/a/{self.account_id}"
        if self.customer_id:
            url += f"/c/{self.customer_id}"
        return url + "/"

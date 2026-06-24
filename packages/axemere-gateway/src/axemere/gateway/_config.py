"""AiGatewayConfig: connection and attribution settings for the Axemere AI Gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AiGatewayConfig:
    """Connection and attribution settings for the Axemere AI Gateway.

    Attributes:
        gateway_url: Base URL of the gateway (e.g. ``http://localhost:7080``).
        gateway_token: Bearer token for the gateway.
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

    gateway_url: str = "http://localhost:7080"
    gateway_token: Optional[str] = None
    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    workload_id: Optional[str] = None
    project_id: Optional[str] = None
    account_id: Optional[str] = None
    customer_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    provider_api_key: Optional[str] = None
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "AiGatewayConfig":
        """Build a config from environment variables.

        Environment variables:
            AXEMERE_GATEWAY_URL: Gateway base URL.
            AXEMERE_GATEWAY_TOKEN: Bearer token.
            AXEMERE_DEFAULT_PROVIDER: Default provider name.
            AXEMERE_DEFAULT_MODEL: Default model name.
            AXEMERE_WORKLOAD_ID: Workload ID for attribution.
            AXEMERE_PROJECT_ID: Project ID for attribution.
            AXEMERE_ACCOUNT_ID: Account ID for attribution.
            AXEMERE_CUSTOMER_ID: Customer ID for attribution.
            AXEMERE_PROVIDER_API_KEY: Upstream provider API key.
            AXEMERE_TIMEOUT: Request timeout in seconds.
        """
        return cls(
            gateway_url=os.environ.get("AXEMERE_GATEWAY_URL", "http://localhost:7080"),
            gateway_token=os.environ.get("AXEMERE_GATEWAY_TOKEN"),
            default_provider=os.environ.get("AXEMERE_DEFAULT_PROVIDER"),
            default_model=os.environ.get("AXEMERE_DEFAULT_MODEL"),
            workload_id=os.environ.get("AXEMERE_WORKLOAD_ID"),
            project_id=os.environ.get("AXEMERE_PROJECT_ID"),
            account_id=os.environ.get("AXEMERE_ACCOUNT_ID"),
            customer_id=os.environ.get("AXEMERE_CUSTOMER_ID"),
            provider_api_key=os.environ.get("AXEMERE_PROVIDER_API_KEY"),
            timeout=int(os.environ.get("AXEMERE_TIMEOUT", "120")),
        )

    def set_defaults(self, provider: str, model: str) -> None:
        """Set default provider and model in-place."""
        self.default_provider = provider
        self.default_model = model

    def proxy_url(self, provider: str) -> str:
        """Build the proxy URL for a given provider.

        The URL embeds attribution path segments so that the gateway can
        attribute the request without requiring an explicit action payload.
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

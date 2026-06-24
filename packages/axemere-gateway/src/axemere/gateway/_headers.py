"""Gateway request header helpers."""

from __future__ import annotations

from typing import Dict, Optional

from axemere.gateway._config import AiGatewayConfig

PLACEHOLDER_API_KEY = "axemere-proxy-managed"


def ai_gateway_headers(
    config: AiGatewayConfig,
    *,
    target_host: Optional[str] = None,
) -> Dict[str, str]:
    """Build the standard Axemere AI Gateway request headers.

    Args:
        config: Gateway connection and attribution settings.
        target_host: Upstream provider hostname (e.g. ``api.openai.com``).
            Required when using proxy mode so the gateway knows where to
            forward the request.

    Returns:
        A dict of HTTP headers to merge into the outgoing request.
    """
    headers: Dict[str, str] = {}

    if config.gateway_token:
        headers["Authorization"] = f"Bearer {config.gateway_token}"

    if target_host:
        headers["X-MVGC-Target-Host"] = target_host

    if config.workload_id:
        headers["X-MVGC-Workload-ID"] = config.workload_id

    if config.project_id:
        headers["X-MVGC-Project-ID"] = config.project_id

    if config.account_id:
        headers["X-MVGC-Account-ID"] = config.account_id

    if config.customer_id:
        headers["X-MVGC-Customer-ID"] = config.customer_id

    if config.labels:
        import json
        headers["X-MVGC-Labels"] = json.dumps(config.labels)

    if config.provider_api_key:
        headers["X-MVGC-Provider-API-Key"] = config.provider_api_key

    return headers

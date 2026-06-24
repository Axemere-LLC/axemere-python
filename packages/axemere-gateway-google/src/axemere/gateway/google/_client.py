"""Factory function wrapping google.genai.Client to route through the Axemere AI Gateway."""

from __future__ import annotations

from typing import Any

from axemere.gateway import AiGatewayConfig, PLACEHOLDER_API_KEY, ai_gateway_headers


def genai_client(
    *,
    config: AiGatewayConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Create a google.genai.Client routed through the Axemere AI Gateway.

    Args:
        config: Gateway connection and attribution settings.
        **kwargs: Additional keyword arguments forwarded to ``genai.Client``.

    Returns:
        A configured ``google.genai.Client`` instance.
    """
    import google.genai as genai

    cfg = config or AiGatewayConfig.from_env()
    headers = ai_gateway_headers(cfg, target_host="generativelanguage.googleapis.com")

    http_options = kwargs.pop("http_options", {}) or {}
    http_options.setdefault("base_url", cfg.gateway_url)
    existing_headers = http_options.get("headers", {}) or {}
    http_options["headers"] = {**headers, **existing_headers}

    # Pass a placeholder so the SDK initialises without complaining about missing
    # credentials, then strip x-goog-api-key from its internal headers.
    # The gateway injects the real Gemini credential; forwarding the placeholder
    # would trigger the BYOK policy rule and send it as the actual API key.
    kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)

    client = genai.Client(http_options=http_options, **kwargs)

    # Remove x-goog-api-key that the SDK set from api_key above.
    try:
        client._api_client._http_options.headers.pop("x-goog-api-key", None)
    except AttributeError:
        pass

    return client

"""Factory helpers that return provider SDK clients pre-configured for the gateway."""

from __future__ import annotations

from typing import Any

from axemere.gateway import AiGatewayConfig, ai_gateway_headers, PLACEHOLDER_API_KEY


def ai_gateway_openai_client(
    config: AiGatewayConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Return an ``openai.OpenAI`` client routed through the Axemere AI Gateway.

    Args:
        config: Gateway connection and attribution settings.
        **kwargs: Additional keyword arguments forwarded to ``openai.OpenAI``.

    Returns:
        A configured ``openai.OpenAI`` instance.
    """
    import openai

    cfg = config or AiGatewayConfig.from_env()
    headers = ai_gateway_headers(cfg, target_host="api.openai.com")
    kwargs.setdefault("base_url", f"{cfg.gateway_url}/v1")
    kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
    return openai.OpenAI(default_headers=headers, **kwargs)


def ai_gateway_anthropic_client(
    config: AiGatewayConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Return an ``anthropic.Anthropic`` client routed through the Axemere AI Gateway.

    Args:
        config: Gateway connection and attribution settings.
        **kwargs: Additional keyword arguments forwarded to ``anthropic.Anthropic``.

    Returns:
        A configured ``anthropic.Anthropic`` instance.
    """
    import anthropic

    cfg = config or AiGatewayConfig.from_env()
    headers = ai_gateway_headers(cfg, target_host="api.anthropic.com")
    kwargs.setdefault("base_url", cfg.gateway_url)
    kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
    return anthropic.Anthropic(default_headers=headers, **kwargs)


def ai_gateway_mistral_client(
    config: AiGatewayConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Return a ``mistralai.Mistral`` client routed through the Axemere AI Gateway.

    Args:
        config: Gateway connection and attribution settings.
        **kwargs: Additional keyword arguments forwarded to ``mistralai.Mistral``.

    Returns:
        A configured ``mistralai.Mistral`` instance.
    """
    import mistralai

    cfg = config or AiGatewayConfig.from_env()
    headers = ai_gateway_headers(cfg, target_host="api.mistral.ai")
    kwargs.setdefault("server_url", cfg.gateway_url)
    kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
    return mistralai.Mistral(default_headers=headers, **kwargs)

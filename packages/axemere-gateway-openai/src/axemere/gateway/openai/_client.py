"""Thin OpenAI client subclasses that route through the Axemere AI Gateway."""

from __future__ import annotations

import openai as _openai

from axemere.gateway import AiGatewayConfig, PLACEHOLDER_API_KEY, ai_gateway_headers


def _extract_azure_hostname(endpoint: str) -> str:
    if not endpoint:
        return ""
    if "://" in endpoint:
        endpoint = endpoint.split("://", 1)[1]
    endpoint = endpoint.split("/")[0]
    return endpoint


class OpenAI(_openai.OpenAI):
    """Drop-in replacement for ``openai.OpenAI`` that routes through the Axemere AI Gateway."""

    def __init__(self, *, config: AiGatewayConfig | None = None, **kwargs) -> None:
        cfg = config or AiGatewayConfig.from_env()
        caller_headers = kwargs.pop("default_headers", None) or {}
        merged_headers = {**ai_gateway_headers(cfg, target_host="api.openai.com"), **caller_headers}
        kwargs.setdefault("base_url", f"{cfg.gateway_url}/v1")
        kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
        super().__init__(default_headers=merged_headers, **kwargs)


class AsyncOpenAI(_openai.AsyncOpenAI):
    """Drop-in replacement for ``openai.AsyncOpenAI`` that routes through the Axemere AI Gateway."""

    def __init__(self, *, config: AiGatewayConfig | None = None, **kwargs) -> None:
        cfg = config or AiGatewayConfig.from_env()
        caller_headers = kwargs.pop("default_headers", None) or {}
        merged_headers = {**ai_gateway_headers(cfg, target_host="api.openai.com"), **caller_headers}
        kwargs.setdefault("base_url", f"{cfg.gateway_url}/v1")
        kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
        super().__init__(default_headers=merged_headers, **kwargs)


class AzureOpenAI(_openai.AzureOpenAI):
    """Drop-in replacement for ``openai.AzureOpenAI`` that routes through the Axemere AI Gateway."""

    def __init__(self, *, config: AiGatewayConfig | None = None, azure_target_host: str | None = None, **kwargs) -> None:
        import os
        cfg = config or AiGatewayConfig.from_env()
        target_host = azure_target_host or os.environ.get("AXEMERE_AZURE_ENDPOINT", "")
        target_host = _extract_azure_hostname(target_host)
        caller_headers = kwargs.pop("default_headers", None) or {}
        merged_headers = {**ai_gateway_headers(cfg, target_host=target_host or "openai.azure.com"), **caller_headers}
        kwargs.setdefault("azure_endpoint", cfg.gateway_url)
        kwargs.setdefault("api_version", "2024-06-01")
        kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
        super().__init__(default_headers=merged_headers, **kwargs)


class AsyncAzureOpenAI(_openai.AsyncAzureOpenAI):
    """Drop-in replacement for ``openai.AsyncAzureOpenAI`` that routes through the Axemere AI Gateway."""

    def __init__(self, *, config: AiGatewayConfig | None = None, azure_target_host: str | None = None, **kwargs) -> None:
        import os
        cfg = config or AiGatewayConfig.from_env()
        target_host = azure_target_host or os.environ.get("AXEMERE_AZURE_ENDPOINT", "")
        target_host = _extract_azure_hostname(target_host)
        caller_headers = kwargs.pop("default_headers", None) or {}
        merged_headers = {**ai_gateway_headers(cfg, target_host=target_host or "openai.azure.com"), **caller_headers}
        kwargs.setdefault("azure_endpoint", cfg.gateway_url)
        kwargs.setdefault("api_version", "2024-06-01")
        kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
        super().__init__(default_headers=merged_headers, **kwargs)

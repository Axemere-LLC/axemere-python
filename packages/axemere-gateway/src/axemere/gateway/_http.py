"""Thin HTTP clients that proxy requests through the Axemere AI Gateway."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import httpx

from axemere.gateway._config import AiGatewayConfig
from axemere.gateway._headers import ai_gateway_headers


class AiGatewayHttpClient:
    """Synchronous HTTP client that rewrites request URLs to route through the gateway.

    Use this when you need raw HTTP access to a provider API via the gateway
    without framework-specific tooling.

    Args:
        config: Gateway connection and attribution settings.
            Defaults to ``AiGatewayConfig.from_env()``.
        target_host: Upstream provider hostname (e.g. ``api.openai.com``).
    """

    def __init__(
        self,
        config: Optional[AiGatewayConfig] = None,
        *,
        target_host: str = "",
    ) -> None:
        self._config = config or AiGatewayConfig.from_env()
        self._target_host = target_host
        self._client = httpx.Client(timeout=float(self._config.timeout))

    def _gateway_headers(self) -> Dict[str, str]:
        return ai_gateway_headers(self._config, target_host=self._target_host)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST to *url* via the gateway.

        The URL is rewritten to point at the gateway; the original host is
        preserved in ``X-MVGC-Target-Host``.
        """
        headers = {**self._gateway_headers(), **kwargs.pop("headers", {})}
        rewritten = self._rewrite_url(url)
        return self._client.post(rewritten, headers=headers, **kwargs)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        headers = {**self._gateway_headers(), **kwargs.pop("headers", {})}
        rewritten = self._rewrite_url(url)
        return self._client.get(rewritten, headers=headers, **kwargs)

    def _rewrite_url(self, url: str) -> str:
        """Replace the host portion of *url* with the gateway base URL."""
        if "://" not in url:
            return f"{self._config.gateway_url}{url}"
        scheme, rest = url.split("://", 1)
        path = rest.split("/", 1)[1] if "/" in rest else ""
        return f"{self._config.gateway_url}/{path}"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AiGatewayHttpClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncAiGatewayHttpClient:
    """Asynchronous HTTP client that rewrites request URLs to route through the gateway.

    Args:
        config: Gateway connection and attribution settings.
            Defaults to ``AiGatewayConfig.from_env()``.
        target_host: Upstream provider hostname (e.g. ``api.openai.com``).
    """

    def __init__(
        self,
        config: Optional[AiGatewayConfig] = None,
        *,
        target_host: str = "",
    ) -> None:
        self._config = config or AiGatewayConfig.from_env()
        self._target_host = target_host
        self._client = httpx.AsyncClient(timeout=float(self._config.timeout))

    def _gateway_headers(self) -> Dict[str, str]:
        return ai_gateway_headers(self._config, target_host=self._target_host)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        headers = {**self._gateway_headers(), **kwargs.pop("headers", {})}
        rewritten = self._rewrite_url(url)
        return await self._client.post(rewritten, headers=headers, **kwargs)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        headers = {**self._gateway_headers(), **kwargs.pop("headers", {})}
        rewritten = self._rewrite_url(url)
        return await self._client.get(rewritten, headers=headers, **kwargs)

    def _rewrite_url(self, url: str) -> str:
        if "://" not in url:
            return f"{self._config.gateway_url}{url}"
        scheme, rest = url.split("://", 1)
        path = rest.split("/", 1)[1] if "/" in rest else ""
        return f"{self._config.gateway_url}/{path}"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncAiGatewayHttpClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

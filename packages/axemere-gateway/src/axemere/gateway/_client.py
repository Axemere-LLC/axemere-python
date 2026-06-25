"""AiGatewayClient: framework-independent async client for the Axemere AI Gateway."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from axemere.gateway._config import AiGatewayConfig
from axemere.gateway._errors import (
    GatewayError,
    GatewayTimeoutError,
    PolicyDeniedError,
    QuotaExceededError,
)
from axemere.gateway._types import (
    CostBreakdownItem,
    ExecuteResponse,
    Metering,
    StreamChunk,
)

# Canonical upstream hostname per provider
_PROVIDER_HOSTS: Dict[str, str] = {
    "openai": "api.openai.com",
    "anthropic": "api.anthropic.com",
    "google": "generativelanguage.googleapis.com",
    "gemini": "generativelanguage.googleapis.com",
    "cohere": "api.cohere.ai",
    "mistral": "api.mistral.ai",
    "groq": "api.groq.com",
    "together": "api.together.xyz",
    "fireworks": "api.fireworks.ai",
    "perplexity": "api.perplexity.ai",
    "deepseek": "api.deepseek.com",
    "xai": "api.x.ai",
    "openrouter": "openrouter.ai",
    "nvidia-nim": "integrate.api.nvidia.com",
    "upstage": "api.upstage.ai",
    "moonshot": "api.moonshot.ai",
    "minimax": "api.minimax.chat",
    "zhipu": "api.z.ai",
    "azure": "openai.azure.com",
    "azure_openai": "openai.azure.com",
    "bedrock": "bedrock.us-east-1.amazonaws.com",
    "vertex": "us-central1-aiplatform.googleapis.com",
    "replicate": "api.replicate.com",
    "huggingface": "api-inference.huggingface.co",
}


class AiGatewayClient:
    """Framework-independent async client for the Axemere AI Gateway.

    This client uses the gateway's explicit ``/v1/actions:execute`` endpoint.
    It is provider-agnostic: you specify the provider and model, and the client
    builds the correct request payload and target routing.

    Example::

        from axemere.gateway import AiGatewayClient, AiGatewayConfig

        config = AiGatewayConfig.from_env()
        client = AiGatewayClient(config)

        response = await client.execute(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.content)
    """

    def __init__(self, config: Optional[AiGatewayConfig] = None) -> None:
        self._config = config or AiGatewayConfig.from_env()

    async def execute(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **extra_params: Any,
    ) -> ExecuteResponse:
        """Send a chat completion request through the gateway.

        Args:
            provider: Provider name (e.g. ``"openai"``, ``"anthropic"``).
            model: Model identifier.
            messages: List of chat messages in OpenAI format
                (``[{"role": "user", "content": "..."}]``).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. Omitted if ``None``.
            system: System prompt (extracted automatically for Anthropic).
            **extra_params: Additional parameters forwarded to the provider.

        Returns:
            An :class:`ExecuteResponse` with ``content``, ``record_id``,
            ``metering``, and the raw ``provider_response``.

        Raises:
            PolicyDeniedError: The request was denied by policy.
            QuotaExceededError: A budget or quota was exceeded.
            GatewayTimeoutError: The connector timed out.
            GatewayError: Any other gateway or HTTP error.
        """
        payload = self._build_payload(
            provider=provider,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            extra_params=extra_params,
        )
        headers = self._build_headers()
        url = f"{self._config.gateway_url}/v1/actions:execute"

        async with httpx.AsyncClient(timeout=float(self._config.timeout)) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                raise GatewayTimeoutError() from exc
            except httpx.RequestError as exc:
                raise GatewayError(f"Gateway connection failed: {exc}") from exc

        return self._parse_response(resp, provider=provider)

    async def _stream(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **extra_params: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion from the gateway (SSE).

        Yields :class:`StreamChunk` objects. The final chunk has
        ``is_final=True`` and carries ``metering`` and ``record_id``.
        """
        payload = self._build_payload(
            provider=provider,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            extra_params={**extra_params, "stream": True},
        )
        headers = self._build_headers()
        url = f"{self._config.gateway_url}/v1/actions:execute"

        metering: Optional[Metering] = None
        record_id: str = ""

        async with httpx.AsyncClient(timeout=float(self._config.timeout)) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise GatewayError(
                        f"Gateway returned HTTP {resp.status_code}",
                        status_code=resp.status_code,
                        response_body=body,
                    )
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        yield StreamChunk(content="", is_final=True, record_id=record_id, metering=metering)
                        return
                    try:
                        chunk = json.loads(data)
                    except Exception:
                        continue

                    chunk_type = chunk.get("type", "")

                    if chunk_type == "mvgc_metering":
                        record_id = chunk.get("record_id", "")
                        metering = self._parse_metering(chunk.get("metering"))
                        continue

                    # Anthropic end-of-stream signal
                    if chunk_type == "message_stop":
                        yield StreamChunk(content="", is_final=True, record_id=record_id, metering=metering)
                        return

                    # Anthropic streaming delta
                    if chunk_type == "content_block_delta":
                        delta = chunk.get("delta", {})
                        if isinstance(delta, dict) and delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield StreamChunk(content=text)
                        continue

                    # OpenAI-compatible streaming delta
                    choices = chunk.get("choices")
                    if choices and isinstance(choices, list):
                        choice = choices[0] if choices else {}
                        if isinstance(choice, dict):
                            delta = choice.get("delta", {})
                            content = delta.get("content", "") if isinstance(delta, dict) else ""
                            finish_reason = choice.get("finish_reason")
                            if content:
                                yield StreamChunk(content=content)
                            if finish_reason and finish_reason != "null":
                                yield StreamChunk(content="", is_final=True, record_id=record_id, metering=metering)
                                return

        # Stream closed without an explicit terminator
        yield StreamChunk(content="", is_final=True, record_id=record_id, metering=metering)

    def execute_sync(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **extra_params: Any,
    ) -> ExecuteResponse:
        """Synchronous wrapper around :meth:`execute`.

        Runs the coroutine in a new event loop. Do not use inside an already-
        running async context — use ``await execute()`` there instead.
        """
        return asyncio.run(
            self.execute(
                provider=provider,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                **extra_params,
            )
        )

    async def stream(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **extra_params: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion through the gateway.

        Returns an async iterator of :class:`StreamChunk` objects. The final
        chunk has ``is_final=True`` and carries ``metering`` and ``record_id``.

        Example::

            async for chunk in await client.stream(provider="anthropic", model="...", messages=[...]):
                if not chunk.is_final:
                    print(chunk.content, end="", flush=True)

        Args:
            provider: Provider name (e.g. ``"anthropic"``, ``"openai"``).
            model: Model identifier.
            messages: List of chat messages in OpenAI format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. Omitted if ``None``.
            system: System prompt.
            **extra_params: Additional parameters forwarded to the provider.

        Returns:
            An async iterator of :class:`StreamChunk` objects.

        Raises:
            PolicyDeniedError: The request was denied by policy.
            GatewayError: On network failure or non-OK gateway status.
        """
        return self._stream(
            provider=provider,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **extra_params,
        )

    def stream_sync(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **extra_params: Any,
    ) -> List[StreamChunk]:
        """Synchronous wrapper around :meth:`stream`.

        Collects all chunks and returns them as a list. Do not call inside an
        already-running async context — use ``async for chunk in await stream()`` there.

        Returns:
            A list of :class:`StreamChunk` objects; the last entry has ``is_final=True``.

        Raises:
            PolicyDeniedError: The request was denied by policy.
            GatewayError: On network failure or non-OK gateway status.
        """
        async def _collect() -> List[StreamChunk]:
            return [chunk async for chunk in self._stream(
                provider=provider,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                **extra_params,
            )]

        return asyncio.run(_collect())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float],
        system: Optional[str],
        extra_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        target_host = _PROVIDER_HOSTS.get(provider, f"api.{provider}.com")

        if provider == "anthropic":
            # Anthropic separates system prompt from messages
            anthropic_messages = [m for m in messages if m.get("role") != "system"]
            if system is None:
                system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
                system = system_msgs[0] if system_msgs else None
            params: Dict[str, Any] = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
            }
            if system:
                params["system"] = system
            if temperature is not None:
                params["temperature"] = temperature
            params.update(extra_params)
            target_path = "/v1/messages"
        elif provider in ("google", "gemini"):
            gemini_msgs = []
            for m in messages:
                role = m.get("role", "user")
                gemini_role = "user" if role in ("user", "system") else "model"
                gemini_msgs.append({"role": gemini_role, "parts": [{"text": str(m.get("content", ""))}]})
            params = {"contents": gemini_msgs}
            if temperature is not None:
                params["generationConfig"] = {"temperature": temperature, "maxOutputTokens": max_tokens}
            params.update(extra_params)
            target_path = f"/v1beta/models/{model}:generateContent"
        else:
            # OpenAI-compatible format (OpenAI, Groq, Mistral, Together, etc.)
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                params["temperature"] = temperature
            params.update(extra_params)
            target_path = "/v1/chat/completions"

        cfg = self._config
        return {
            "schema": "mvgc.action_request.v2",
            "request_id": str(uuid.uuid4()),
            "org_id": "",
            "caller_id": "axemere-gateway-python",
            "workload_id": cfg.workload_id or None,
            "ingress_mode": "explicit_action_request",
            "action": {
                "type": "ai.infer",
                "method": "POST",
                "target_host": target_host,
                "target_path": target_path,
                "params": params,
            },
            "attribution": {
                "project_id": cfg.project_id,
                "customer_id": cfg.customer_id,
                "account_id": cfg.account_id,
                "labels": cfg.labels or {},
            },
        }

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._config.gateway_token:
            headers["Authorization"] = f"Bearer {self._config.gateway_token}"
        return headers

    @staticmethod
    def _parse_response(resp: httpx.Response, *, provider: str) -> ExecuteResponse:
        try:
            data = resp.json()
        except Exception as exc:
            raise GatewayError(
                f"Invalid JSON from gateway (status={resp.status_code}): {exc}",
                status_code=resp.status_code,
            ) from exc

        # Default to "allow" when the gateway omits a decision: an explicit
        # policy denial always sets decision="deny" (or returns HTTP 403), so a
        # missing field means a normal response, not a silent deny.
        decision = data.get("decision", "allow")
        if decision == "deny" or resp.status_code == 403:
            trace = data.get("decision_trace", {})
            reason = trace.get("reason") or data.get("reason") or "Policy denied the request"
            raise PolicyDeniedError(reason, reason=reason, trace=trace, record_id=data.get("record_id", ""))

        if resp.status_code == 429:
            raise QuotaExceededError(
                data.get("message", "Quota exceeded"),
                upgrade_url=data.get("upgrade_url", ""),
            )

        if resp.status_code == 504:
            raise GatewayTimeoutError()

        if resp.status_code >= 400:
            raise GatewayError(
                f"Gateway returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                response_body=data,
            )

        metering = AiGatewayClient._parse_metering(data.get("metering"))
        result_dto = data.get("result", {})
        body = result_dto.get("body", {})

        # Check for upstream errors surfaced inside result
        upstream_status = result_dto.get("status_code", 200)
        if isinstance(upstream_status, int) and upstream_status >= 400:
            err_msg = AiGatewayClient._extract_error_message(body)
            raise GatewayError(
                f"Upstream {provider} returned {upstream_status}: {err_msg}",
                status_code=upstream_status,
                response_body=body,
            )

        content = AiGatewayClient._extract_content(body, provider=provider)
        model = ""
        if isinstance(body, dict):
            model = body.get("model", "")

        return ExecuteResponse(
            content=content,
            record_id=data.get("record_id", ""),
            metering=metering,
            provider=provider,
            model=model,
            record_hash=data.get("record_hash", ""),
            provider_response=body if isinstance(body, dict) else None,
        )

    @staticmethod
    def _parse_metering(raw: Any) -> Optional[Metering]:
        if not isinstance(raw, dict):
            return None
        breakdowns = []
        for item in raw.get("cost_breakdown", []):
            breakdowns.append(CostBreakdownItem(
                label=item.get("label", ""),
                tokens=int(item.get("tokens", 0)),
                rate_per_million=float(item.get("rate_per_million", 0)),
                subtotal_usd=float(item.get("subtotal_usd", 0)),
            ))
        return Metering(
            cost_usd=float(raw.get("cost_usd", 0) or 0),
            tokens_in=int(raw.get("tokens_in", 0) or 0),
            tokens_out=int(raw.get("tokens_out", 0) or 0),
            bytes_in=int(raw.get("bytes_in", 0) or 0),
            bytes_out=int(raw.get("bytes_out", 0) or 0),
            cache_hit_tokens=int(raw.get("cache_hit_tokens", 0) or 0),
            cache_miss_tokens=int(raw.get("cache_miss_tokens", 0) or 0),
            cache_creation_tokens=int(raw.get("cache_creation_tokens", 0) or 0),
            reasoning_tokens=int(raw.get("reasoning_tokens", 0) or 0),
            pricing_config_version=str(raw.get("pricing_config_version", "")),
            org_pricing_config_version=str(raw.get("org_pricing_config_version", "")),
            markup_multiplier_applied=float(raw.get("markup_multiplier_applied", 1.0) or 1.0),
            cost_breakdown=breakdowns,
        )

    @staticmethod
    def _extract_content(body: Any, *, provider: str) -> str:
        if not isinstance(body, dict):
            return str(body) if body else ""
        if provider == "anthropic":
            blocks = body.get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        elif provider in ("google", "gemini"):
            candidates = body.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        else:
            choices = body.get("choices", [])
            if not choices:
                return ""
            msg = choices[0].get("message", {})
            return msg.get("content") or ""

    @staticmethod
    def _extract_error_message(body: Any) -> str:
        if not isinstance(body, dict):
            return str(body)
        err = body.get("error", {})
        if isinstance(err, dict):
            return err.get("message", str(err))
        return str(err) if err else str(body)

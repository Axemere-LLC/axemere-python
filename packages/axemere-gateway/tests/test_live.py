"""Live integration tests — require real gateway credentials.

These tests run against the managed Axemere AI Gateway and are skipped
automatically when AXEMERE_GATEWAY_TOKEN is not set. To run them:

    source tests/.env
    pytest python/packages/axemere-gateway/tests/test_live.py -v

The gateway URL defaults to https://us.gw.axemere.ai unless
AXEMERE_GATEWAY_URL is set.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AXEMERE_GATEWAY_TOKEN"),
    reason="AXEMERE_GATEWAY_TOKEN not set — skipping live gateway tests",
)

_GATEWAY_URL = os.environ.get("AXEMERE_GATEWAY_URL", "https://us.gw.axemere.ai")
_PROVIDER = "anthropic"
_MODEL = "claude-haiku-4-5-20251001"


def _make_client():
    from axemere.gateway import AiGatewayClient, AiGatewayConfig

    cfg = AiGatewayConfig(
        gateway_url=_GATEWAY_URL,
        gateway_token=os.environ["AXEMERE_GATEWAY_TOKEN"],
    )
    return AiGatewayClient(cfg)


# ---------------------------------------------------------------------------
# Basic completion
# ---------------------------------------------------------------------------


def test_live_basic_completion():
    """Execute a minimal completion and verify non-empty content is returned."""
    client = _make_client()
    result = client.execute_sync(
        provider=_PROVIDER,
        model=_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=10,
    )
    assert result.content, "Expected non-empty content from live gateway"
    assert result.record_id, "Expected a record_id from live gateway"
    assert result.metering is not None, "Expected metering data from live gateway"
    assert result.metering.tokens_out > 0, "Expected tokens_out > 0"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_live_streaming():
    """Stream a short response and verify chunks arrive with final metering."""
    import asyncio

    client = _make_client()

    async def _run():
        chunks = []
        async for chunk in await client.stream(
            provider=_PROVIDER,
            model=_MODEL,
            messages=[{"role": "user", "content": "Count to three: 1, 2, 3."}],
            max_tokens=20,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_run())
    assert chunks, "Expected at least one streaming chunk"

    content_chunks = [c for c in chunks if not c.is_final]
    final_chunks = [c for c in chunks if c.is_final]

    assert content_chunks, "Expected at least one content chunk before final"
    assembled = "".join(c.content for c in content_chunks)
    assert assembled, "Expected assembled content to be non-empty"

    assert final_chunks, "Expected a final chunk marking end of stream"
    final = final_chunks[-1]
    assert final.metering is not None, "Expected metering data on final streaming chunk"


# ---------------------------------------------------------------------------
# Error on bad token
# ---------------------------------------------------------------------------


def test_live_bad_token_raises_error():
    """A request with an invalid token must raise a GatewayError."""
    from axemere.gateway import AiGatewayClient, AiGatewayConfig, GatewayError

    cfg = AiGatewayConfig(
        gateway_url=_GATEWAY_URL,
        gateway_token="axm_k_invalid_token_for_testing",
    )
    client = AiGatewayClient(cfg)

    with pytest.raises(GatewayError):
        client.execute_sync(
            provider=_PROVIDER,
            model=_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )

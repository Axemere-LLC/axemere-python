"""Example 15: Cohere via Axemere AI Gateway

Demonstrates:
- Single chat via Cohere's /v2/chat REST API routed through AiGatewayHttpClient
- Multi-turn conversation with message history
- Streaming via explicit action request mode (SSE forwarded by the gateway)

The gateway manages the Cohere API key — set COHERE_API_KEY in the root .env
file (alongside OPENAI_API_KEY / ANTHROPIC_API_KEY) so the gateway's
cred-cohere credential can resolve it.  No provider API key is needed in the
Python process.

No axemere.gateway.cohere wrapper exists; AiGatewayHttpClient is the right tool for any HTTP
API that doesn't have a dedicated Axemere AI Gateway SDK wrapper.

Run:
    python 15_cohere_sdk.py
"""

import json
import sys

import httpx
from dotenv import load_dotenv

from axemere.gateway import AiGatewayConfig
from axemere.gateway import AiGatewayHttpClient

COHERE_CHAT_URL = "https://api.cohere.com/v2/chat"


def demo_single_chat(client: AiGatewayHttpClient) -> None:
    print("\n--- Cohere single chat ---")
    payload = {
        "model": "command-r-plus",
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer in one sentence."}],
        "max_tokens": 64,
    }
    # No Authorization header — the gateway injects the Cohere API key from cred-cohere.
    response = client.post(COHERE_CHAT_URL, json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data["message"]["content"][0]["text"]
        usage = data.get("usage", {})
        print(f"Response: {content}")
        print(f"Tokens in: {usage.get('billed_units', {}).get('input_tokens', 'n/a')}")
        print(f"Tokens out: {usage.get('billed_units', {}).get('output_tokens', 'n/a')}")
    else:
        print(f"Body: {response.text[:400]}")


def demo_multi_turn(client: AiGatewayHttpClient) -> None:
    print("\n--- Cohere multi-turn conversation ---")
    messages: list[dict] = [
        {"role": "user", "content": "Name three programming languages invented before 1990."},
    ]
    payload = {
        "model": "command-r-plus",
        "messages": messages,
        "max_tokens": 128,
    }
    first = client.post(COHERE_CHAT_URL, json=payload)
    if first.status_code != 200:
        print(f"Round 1 failed: HTTP {first.status_code}  {first.text[:200]}")
        return
    reply = first.json()["message"]["content"][0]["text"]
    print(f"Round 1 -- Assistant: {reply}")

    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": "Which of those is most commonly used today?"})

    second = client.post(COHERE_CHAT_URL, json={**payload, "messages": messages})
    if second.status_code != 200:
        print(f"Round 2 failed: HTTP {second.status_code}  {second.text[:200]}")
        return
    print(f"Round 2 -- Assistant: {second.json()['message']['content'][0]['text']}")


def demo_streaming(cfg: AiGatewayConfig) -> None:
    """Cohere SSE stream via explicit action request mode.

    When stream=True is set in action.params the gateway forwards the raw SSE
    response directly, identical to OpenAI streaming in example 14.
    Cohere SSE events are JSON objects; content deltas have type='content-delta'.
    """
    print("\n--- Cohere streaming (explicit action request) ---")
    action_request = {
        "schema": "mvgc.action_request.v2",
        "org_id": "",
        "workload_id": cfg.workload_id,
        "ingress_mode": "explicit_action_request",
        "action": {
            "type": "ai.infer",
            "method": "POST",
            "target_host": "api.cohere.com",
            "target_path": "/v2/chat",
            "params": {
                "model": "command-r-plus",
                "messages": [{"role": "user", "content": "Tell me a one-sentence fun fact about the ocean."}],
                "max_tokens": 128,
                "stream": True,
            },
        },
        "attribution": {
            "project_id": cfg.project_id,
            "customer_id": cfg.customer_id,
            "account_id": cfg.account_id,
            "labels": {"source": "cohere-example"},
        },
    }

    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{cfg.gateway_url}/v1/actions:execute",
            json=action_request,
        ) as response:
            if response.status_code != 200:
                response.read()
                print(f"HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
                return

            print(f"Content-Type: {response.headers.get('content-type')}")
            print("Stream: ", end="", flush=True)
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "content-delta":
                    text = event.get("delta", {}).get("message", {}).get("content", {}).get("text", "")
                    if text:
                        print(text, end="", flush=True)
                elif event_type == "message-end":
                    break
    print()  # newline after stream


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")
    print("(Cohere API key is managed by the gateway — set COHERE_API_KEY in the root .env)")

    try:
        with AiGatewayHttpClient(config=cfg) as client:
            demo_single_chat(client)
            demo_multi_turn(client)
        demo_streaming(cfg)
    except Exception as exc:
        msg = str(exc).lower()
        if "connection" in msg or "connect" in msg or "refused" in msg:
            print(f"\nConnectivity error: {exc}")
            print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        else:
            print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

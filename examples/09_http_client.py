"""Example 09: Generic HTTP Client

Demonstrates:
- Using AiGatewayHttpClient as a governed HTTP client for arbitrary API calls
- Context manager usage for connection pooling
- POST to OpenAI chat completions via raw HTTP (URL rewriting in action)
- GET health check against the gateway
- Error handling for connectivity errors

Run:
    python 09_http_client.py
"""

import json
import sys

from dotenv import load_dotenv

from axemere.gateway import AiGatewayConfig, AiGatewayHttpClient


def demo_health_check(client: AiGatewayHttpClient, gateway_url: str) -> None:
    print("\n--- Gateway health check ---")
    response = client.get(f"{gateway_url}/healthz")
    print(f"Status: {response.status_code}")
    if response.text:
        print(f"Body: {response.text}")


def demo_openai_post(client: AiGatewayHttpClient) -> None:
    print("\n--- POST to OpenAI via raw HTTP (URL rewritten by gateway) ---")
    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say 'hello from raw HTTP' in one sentence."}],
    }
    response = client.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload,
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['choices'][0]['message']['content']}")
        print(f"Model: {data['model']}")
    else:
        print(f"Body: {response.text[:200]}")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    try:
        with AiGatewayHttpClient(config=cfg) as client:
            demo_health_check(client, cfg.gateway_url)
            demo_openai_post(client)
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

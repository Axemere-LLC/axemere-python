"""Example 02: Proxy Mode with Standard SDK Clients

Demonstrates:
- Using ai_gateway_openai_client() and ai_gateway_anthropic_client() for proxy mode
- Client is pre-configured with gateway base_url and attribution headers
- Zero changes to calling patterns — only client construction differs
- The gateway routes to the upstream provider via X-MVGC-Target-Host header

Run:
    python 02_proxy_chat.py
"""

import sys

from axemere.gateway import AiGatewayConfig, GatewayError
from axemere.gateway.langchain import ai_gateway_anthropic_client, ai_gateway_openai_client


def demo_openai_proxy(cfg: AiGatewayConfig) -> None:
    print("\n--- OpenAI via Axemere proxy mode ---")
    client = ai_gateway_openai_client(cfg)
    print(f"Base URL: {cfg.gateway_url}/v1")
    print("Attribution headers injected automatically")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello from proxy mode! Reply in one sentence."}],
        )
        print(f"Response: {response.choices[0].message.content}")
    except GatewayError as exc:
        print(f"Gateway error: {exc}")
        print("Is the gateway running? Try: docker compose up -d")
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def demo_anthropic_proxy(cfg: AiGatewayConfig) -> None:
    print("\n--- Anthropic via Axemere proxy mode ---")
    client = ai_gateway_anthropic_client(cfg)
    print(f"Base URL: {cfg.gateway_url}")
    print("Attribution headers injected automatically")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello from proxy mode! Reply in one sentence."}],
        )
        print(f"Response: {response.content[0].text}")
    except GatewayError as exc:
        print(f"Gateway error: {exc}")
        print("Is the gateway running? Try: docker compose up -d")
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def main() -> None:
    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    demo_openai_proxy(cfg)
    demo_anthropic_proxy(cfg)


if __name__ == "__main__":
    main()

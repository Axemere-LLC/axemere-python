"""Example 01: Explicit Mode Hello World

Demonstrates:
- Sending a single message to OpenAI via the Axemere AI Gateway
- Sending a single message to Anthropic via the Axemere AI Gateway
- A successful allow response with decision_trace
- A policy denial (PolicyDeniedError) when project_id is missing

Run:
    python 01_basic_chat.py
"""

import sys

from axemere.gateway.langchain import ChatAiGateway
from axemere.gateway import AiGatewayConfig, PolicyDeniedError, GatewayError


def demo_openai(cfg: AiGatewayConfig) -> None:
    print("\n--- OpenAI via Axemere AI Gateway (explicit mode) ---")
    llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=128,
        config=cfg,
    )
    response = llm.invoke("What is 2 + 2? Answer in one sentence.")
    print(f"Response: {response.content}")


def demo_anthropic(cfg: AiGatewayConfig) -> None:
    print("\n--- Anthropic via Axemere AI Gateway (explicit mode) ---")
    llm = ChatAiGateway(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        config=cfg,
    )
    response = llm.invoke("What is the capital of France? Answer in one sentence.")
    print(f"Response: {response.content}")


def demo_policy_denial() -> None:
    """Omit project_id to trigger a policy deny from the gateway."""
    print("\n--- Policy denial: missing project_id ---")
    no_project_cfg = AiGatewayConfig(
        gateway_url=AiGatewayConfig.from_env().gateway_url,
        workload_id="wl-prod-app-1",
        project_id="",  # intentionally empty
        customer_id="cust-42",
        account_id="acct-12",
    )
    llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=64,
        config=no_project_cfg,
    )
    try:
        llm.invoke("Hello")
        print("ERROR: expected PolicyDeniedError but request succeeded")
    except PolicyDeniedError as exc:
        print(f"Gateway denied the request (expected): {exc}")
        if exc.trace:
            print(f"  Decision trace: {exc.trace}")
    except GatewayError as exc:
        print(f"Gateway error (check that the gateway is running): {exc}")


def main() -> None:
    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    try:
        demo_openai(cfg)
        demo_anthropic(cfg)
    except PolicyDeniedError as exc:
        print(f"\nPolicy denied: {exc}")
        print("Check that your gateway policies allow this workload/project.")
        sys.exit(1)
    except GatewayError as exc:
        print(f"\nGateway error: {exc}")
        print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        sys.exit(1)

    demo_policy_denial()


if __name__ == "__main__":
    main()

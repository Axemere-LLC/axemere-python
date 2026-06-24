"""Example 08: Anthropic SDK Drop-in Replacement

Demonstrates:
- Using axemere.gateway.anthropic.Anthropic as a drop-in for anthropic.Anthropic
- Single message creation routed through the Axemere AI Gateway
- Multi-turn conversation with message history
- Error handling for connectivity and policy errors

Run:
    python 08_anthropic_sdk.py
"""

import sys

from dotenv import load_dotenv

from axemere.gateway.anthropic import Anthropic
from axemere.gateway import AiGatewayConfig


def demo_single_message(client: Anthropic) -> None:
    print("\n--- Single message ---")
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    )
    print(f"Response: {response.content[0].text}")
    print(f"Model: {response.model}  Stop reason: {response.stop_reason}")


def demo_multi_turn(client: Anthropic) -> None:
    print("\n--- Multi-turn conversation ---")
    messages = [
        {"role": "user", "content": "Name three planets in the solar system."},
    ]
    first = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=messages,
    )
    assistant_reply = first.content[0].text
    print(f"Round 1 -- Assistant: {assistant_reply}")

    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "Which of those is closest to the Sun?"})

    second = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=messages,
    )
    print(f"Round 2 -- Assistant: {second.content[0].text}")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    client = Anthropic(config=cfg)

    try:
        demo_single_message(client)
        demo_multi_turn(client)
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

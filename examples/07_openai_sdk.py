"""Example 07: OpenAI SDK Drop-in Replacement

Demonstrates:
- Using axemere.gateway.openai.OpenAI as a drop-in for openai.OpenAI
- Single chat completion routed through the Axemere AI Gateway
- Multi-turn conversation with message history
- Error handling for connectivity and policy errors

Run:
    python 07_openai_sdk.py
"""

import sys

from dotenv import load_dotenv

from axemere.gateway.openai import OpenAI
from axemere.gateway import AiGatewayConfig


def demo_single_completion(client: OpenAI) -> None:
    print("\n--- Single chat completion ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=128,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer in one sentence."}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}  Tokens: {response.usage.total_tokens}")


def demo_multi_turn(client: OpenAI) -> None:
    print("\n--- Multi-turn conversation ---")
    messages = [
        {"role": "user", "content": "Name the three primary colors."},
    ]
    first = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=64,
        messages=messages,
    )
    assistant_reply = first.choices[0].message.content
    print(f"Round 1 -- Assistant: {assistant_reply}")

    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "Which of those do you get by mixing red and blue?"})

    second = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=64,
        messages=messages,
    )
    print(f"Round 2 -- Assistant: {second.choices[0].message.content}")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    client = OpenAI(config=cfg)

    try:
        demo_single_completion(client)
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

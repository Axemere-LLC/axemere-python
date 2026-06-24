"""Example 10: Google GenAI SDK Drop-in Replacement

Demonstrates:
- Using axemere.gateway.google.genai_client() routed through the Axemere AI Gateway
- Single generation call
- Multi-turn conversation via contents list
- Error handling for connectivity and policy errors

Run:
    python 10_gemini_sdk.py
"""

import sys

from dotenv import load_dotenv

from axemere.gateway.google import genai_client
from axemere.gateway import AiGatewayConfig


def demo_single_generation(client) -> None:
    print("\n--- Single generation ---")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is 2 + 2? Answer in one sentence.",
    )
    print(f"Response: {response.text}")


def demo_multi_turn(client) -> None:
    print("\n--- Multi-turn conversation ---")
    contents = [
        {"role": "user", "parts": [{"text": "Name the three primary colors."}]},
    ]
    first = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
    assistant_reply = first.text
    print(f"Round 1 -- Model: {assistant_reply}")

    contents.append({"role": "model", "parts": [{"text": assistant_reply}]})
    contents.append(
        {
            "role": "user",
            "parts": [{"text": "Which of those do you get by mixing red and blue?"}],
        }
    )
    second = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
    print(f"Round 2 -- Model: {second.text}")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    client = genai_client(config=cfg)

    try:
        demo_single_generation(client)
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

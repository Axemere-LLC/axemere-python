"""Example 12: LlamaIndex SDK Integration

Demonstrates:
- Using axemere.gateway.llamaindex.ChatAiGateway as a LlamaIndex CustomLLM
- Single completion via complete()
- Multi-turn chat via chat()
- Error handling for connectivity and policy errors

Run:
    python 12_llamaindex_sdk.py
"""

import sys

from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from axemere.gateway.llamaindex import ChatAiGateway
from axemere.gateway import AiGatewayConfig


def demo_complete(llm: ChatAiGateway) -> None:
    print("\n--- Single completion ---")
    response = llm.complete("What is 2 + 2? Answer in one sentence.")
    print(f"Response: {response.text}")


def demo_chat(llm: ChatAiGateway) -> None:
    print("\n--- Chat conversation ---")
    messages = [
        ChatMessage(role=MessageRole.USER, content="Name the three primary colors."),
    ]
    first = llm.chat(messages)
    assistant_reply = first.message.content
    print(f"Round 1 -- Assistant: {assistant_reply}")

    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_reply))
    messages.append(
        ChatMessage(
            role=MessageRole.USER,
            content="Which of those do you get by mixing red and blue?",
        )
    )
    second = llm.chat(messages)
    print(f"Round 2 -- Assistant: {second.message.content}")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    llm = ChatAiGateway(provider="openai", model="gpt-4o-mini", config=cfg)

    try:
        demo_complete(llm)
        demo_chat(llm)
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

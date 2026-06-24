"""Example 03: LCEL Chain Pipeline

Demonstrates:
- Building a LangChain Expression Language (LCEL) chain through the Axemere AI Gateway
- ChatPromptTemplate | ChatAiGateway | StrOutputParser composition
- Parameterized prompts with invoke() and batch()
- Governance is transparent to the chain — every LLM call is governed

Run:
    python 03_chain_pipeline.py
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from axemere.gateway.langchain import ChatAiGateway
from axemere.gateway import AiGatewayConfig, PolicyDeniedError, GatewayError


def build_chain(cfg: AiGatewayConfig):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise technical assistant. Answer in at most two sentences.",
            ),
            ("human", "Explain {concept} to a software engineer."),
        ]
    )

    llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=200,
        config=cfg,
    )

    return prompt | llm | StrOutputParser()


def main() -> None:
    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")

    chain = build_chain(cfg)

    print("\n--- Single invoke ---")
    try:
        result = chain.invoke({"concept": "idempotency"})
        print(f"idempotency: {result}")
    except PolicyDeniedError as exc:
        print(f"Policy denied: {exc}")
        sys.exit(1)
    except GatewayError as exc:
        print(f"Gateway error: {exc}")
        print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        sys.exit(1)

    print("\n--- Batch invoke (3 concepts) ---")
    concepts = [
        {"concept": "rate limiting"},
        {"concept": "circuit breakers"},
        {"concept": "eventual consistency"},
    ]

    try:
        results = chain.batch(concepts)
    except (PolicyDeniedError, GatewayError) as exc:
        print(f"Error during batch: {exc}")
        sys.exit(1)

    for item, answer in zip(concepts, results):
        print(f"\n{item['concept']}:\n  {answer}")


if __name__ == "__main__":
    main()

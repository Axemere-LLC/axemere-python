"""Example 05: Multi-Provider Parallel Routing

Demonstrates:
- Two ChatAiGateway instances (OpenAI + Anthropic) targeting different tasks
- RunnableParallel for concurrent execution through the gateway
- Unified attribution and audit trail across both providers
- The gateway records both calls with the same workload/project attribution

Run:
    python 05_multi_provider.py
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from axemere.gateway.langchain import ChatAiGateway
from axemere.gateway import AiGatewayConfig, PolicyDeniedError, GatewayError


def build_parallel_chain(cfg: AiGatewayConfig):
    openai_llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=200,
        config=cfg,
    )

    anthropic_llm = ChatAiGateway(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        config=cfg,
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "{question}")]
    )

    parser = StrOutputParser()

    openai_chain = prompt | openai_llm | parser
    anthropic_chain = prompt | anthropic_llm | parser

    return RunnableParallel(
        openai=openai_chain,
        anthropic=anthropic_chain,
    )


def main() -> None:
    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Both providers governed under workload={cfg.workload_id}, project={cfg.project_id}")

    chain = build_parallel_chain(cfg)

    questions = [
        "What are the three laws of thermodynamics? One sentence each.",
        "Name three design patterns from the Gang of Four book. One sentence each.",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)

        try:
            results = chain.invoke({"question": question})
        except PolicyDeniedError as exc:
            print(f"Policy denied: {exc}")
            sys.exit(1)
        except GatewayError as exc:
            print(f"Gateway error: {exc}")
            print("Is the Axemere AI Gateway running? Try: docker compose up -d")
            sys.exit(1)

        print(f"[OpenAI gpt-4o-mini]\n{results['openai']}")
        print(f"\n[Anthropic claude-haiku-4-5-20251001]\n{results['anthropic']}")

    print("\nBoth providers were governed. Check the gateway JSONL log for execution records:")
    print('  docker compose exec gateway tail -n 4 /data/records.jsonl | python -c "import sys,json; [print(json.dumps(json.loads(l),indent=2)) for l in sys.stdin if l.strip()]"')


if __name__ == "__main__":
    main()

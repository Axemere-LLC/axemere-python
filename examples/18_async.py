"""Example 18: Async and Concurrent Governed Requests

Demonstrates:
- asyncio.gather() for concurrent governed requests via httpx.AsyncClient
- LangChain .stream() for token-by-token output through the gateway
- LangChain .astream() (async streaming) through the gateway
- High-throughput pattern: fan-out N requests, collect results

Every request in this example is governed — the gateway enforces policy, records
execution, and tracks spend — regardless of whether it is sync or async.

Run:
    python 18_async.py
"""

import asyncio
import json
import sys
import time

import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from axemere.gateway.langchain import ChatAiGateway
from axemere.gateway import AiGatewayConfig, PolicyDeniedError, GatewayError


# ---------------------------------------------------------------------------
# LangChain synchronous streaming
# ---------------------------------------------------------------------------

def demo_langchain_stream(cfg: AiGatewayConfig) -> None:
    """LangChain .stream() emits chunks as they arrive from the gateway."""
    print("\n--- LangChain .stream() through gateway ---")
    llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=128,
        config=cfg,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a two-sentence description of {topic}.")]
    )
    chain = prompt | llm | StrOutputParser()

    print("Streaming: ", end="", flush=True)
    for chunk in chain.stream({"topic": "distributed systems"}):
        print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# LangChain async streaming
# ---------------------------------------------------------------------------

async def demo_langchain_astream(cfg: AiGatewayConfig) -> None:
    """LangChain .astream() is the async variant; useful inside async frameworks."""
    print("\n--- LangChain .astream() through gateway ---")
    llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=128,
        config=cfg,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Name three benefits of {topic} in one sentence each.")]
    )
    chain = prompt | llm | StrOutputParser()

    print("Async stream: ", end="", flush=True)
    async for chunk in chain.astream({"topic": "API governance"}):
        print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# asyncio.gather() fan-out
# ---------------------------------------------------------------------------

async def _single_governed_request(
    session: httpx.AsyncClient,
    gateway_url: str,
    workload_id: str,
    project_id: str,
    question: str,
    index: int,
) -> dict:
    """Send one explicit action request and return a result dict."""
    payload = {
        "schema": "mvgc.action_request.v2",
        "org_id": "",
        "workload_id": workload_id,
        "ingress_mode": "explicit_action_request",
        "action": {
            "type": "ai.infer",
            "method": "POST",
            "target_host": "api.openai.com",
            "target_path": "/v1/chat/completions",
            "params": {
                "model": "gpt-4o-mini",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": question}],
            },
        },
        "attribution": {
            "project_id": project_id,
            "labels": {"source": "async-example", "request_index": str(index)},
        },
    }
    t0 = time.monotonic()
    resp = await session.post(f"{gateway_url}/v1/actions:execute", json=payload)
    elapsed = time.monotonic() - t0

    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    answer = ""
    if resp.status_code == 200 and body.get("result"):
        result_body = body["result"].get("body", {})
        choices = result_body.get("choices", []) if isinstance(result_body, dict) else []
        if choices:
            answer = choices[0].get("message", {}).get("content", "")

    return {
        "index": index,
        "question": question,
        "status": resp.status_code,
        "decision": body.get("decision", "?"),
        "answer": answer,
        "elapsed_s": round(elapsed, 2),
    }


async def demo_concurrent_requests(cfg: AiGatewayConfig) -> None:
    """Fan out 5 governed requests concurrently using asyncio.gather()."""
    print("\n--- asyncio.gather(): 5 concurrent governed requests ---")
    questions = [
        "What is a Merkle tree? Answer in one sentence.",
        "What is rate limiting? Answer in one sentence.",
        "What is a policy bundle? Answer in one sentence.",
        "What is Ed25519? Answer in one sentence.",
        "What is JCS canonicalization? Answer in one sentence.",
    ]

    async with httpx.AsyncClient(timeout=60.0) as session:
        t0 = time.monotonic()
        results = await asyncio.gather(
            *[
                _single_governed_request(
                    session,
                    cfg.gateway_url,
                    cfg.workload_id,
                    cfg.project_id,
                    q,
                    i,
                )
                for i, q in enumerate(questions)
            ]
        )
        total = time.monotonic() - t0

    for r in sorted(results, key=lambda x: x["index"]):
        status_str = f"HTTP {r['status']} ({r['decision']})"
        answer_preview = r["answer"][:80].replace("\n", " ") if r["answer"] else "—"
        print(f"  [{r['index']}] {r['elapsed_s']}s  {status_str}")
        print(f"       Q: {r['question']}")
        print(f"       A: {answer_preview}")

    print(f"\nAll 5 requests completed in {total:.2f}s total")
    approved = sum(1 for r in results if r["decision"] == "allow")
    print(f"Governed: {approved}/{len(results)} allowed by policy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(cfg: AiGatewayConfig) -> None:
    await demo_langchain_astream(cfg)
    await demo_concurrent_requests(cfg)


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    try:
        demo_langchain_stream(cfg)
    except PolicyDeniedError as exc:
        print(f"\nPolicy denied: {exc}")
        sys.exit(1)
    except GatewayError as exc:
        print(f"\nGateway error: {exc}")
        print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        sys.exit(1)

    try:
        asyncio.run(async_main(cfg))
    except PolicyDeniedError as exc:
        print(f"Policy denied: {exc}")
        sys.exit(1)
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

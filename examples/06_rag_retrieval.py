"""Example 06: RAG Pipeline

Demonstrates:
- Loading the project README as a local knowledge base
- Splitting documents with RecursiveCharacterTextSplitter
- Building a FAISS vector store with FakeEmbeddings (no external embedding API needed)
- Retrieval-Augmented Generation chain: retrieve -> format -> governed generation
- Only the LLM generation step passes through the gateway; retrieval is fully local

Run:
    python 06_rag_retrieval.py
"""

import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from axemere.gateway.langchain import ChatAiGateway
from axemere.gateway import AiGatewayConfig, PolicyDeniedError, GatewayError


def load_readme() -> str:
    """Load the project README as the knowledge base."""
    candidates = [
        Path(__file__).parent.parent / "README.md",
        Path(__file__).parent / "README.md",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    # Fallback: minimal inline text so the example still runs without a README
    return """
    Axemere AI Gateway is an AI governance gateway written in Go.
    It supports explicit action requests via POST /v1/actions:execute.
    The gateway enforces policies, manages credentials, and records every AI API call.
    It supports OpenAI and Anthropic providers.
    Attribution fields (project_id, customer_id, account_id) enable cost chargeback.
    The policy DSL supports operators: equals, in, not_in, exists, regex, prefix, suffix,
    lt, lte, gt, gte, all, any.
    """


def build_retriever(text: str):
    try:
        from langchain_community.embeddings.fake import FakeEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Run: pip install langchain-community langchain-text-splitters faiss-cpu")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    print(f"Split into {len(docs)} chunks")

    # FakeEmbeddings generates random vectors — no external API call needed.
    embeddings = FakeEmbeddings(size=256)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, cfg: AiGatewayConfig):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer based only on the provided context. "
                "If the context does not contain the answer, say so.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=300,
        config=cfg,
    )

    # Retrieval is local; only the LLM call goes through Axemere AI Gateway governance.
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main() -> None:
    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")

    print("\nLoading knowledge base...")
    text = load_readme()
    print(f"Loaded {len(text)} characters")

    print("Building vector store (local, no API call)...")
    retriever = build_retriever(text)

    chain = build_rag_chain(retriever, cfg)

    questions = [
        "What is Axemere AI Gateway and what does it do?",
        "What AI providers does Axemere AI Gateway support?",
        "What attribution fields does Axemere AI Gateway use for cost tracking?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        try:
            answer = chain.invoke(question)
            print(f"A: {answer}")
        except PolicyDeniedError as exc:
            print(f"Policy denied: {exc}")
            sys.exit(1)
        except GatewayError as exc:
            print(f"Gateway error: {exc}")
            print("Is the Axemere AI Gateway running? Try: docker compose up -d")
            sys.exit(1)

    print("\nRAG complete. Only the generation steps appeared in the gateway execution log.")
    print('Check: docker compose exec gateway tail -n 3 /data/records.jsonl | python -c "import sys,json; [print(json.dumps(json.loads(l),indent=2)) for l in sys.stdin if l.strip()]"')


if __name__ == "__main__":
    main()

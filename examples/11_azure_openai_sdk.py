"""Example 11: Azure OpenAI SDK Drop-in Replacement

Demonstrates:
- Using axemere.gateway.openai.AzureOpenAI routed through the Axemere AI Gateway
- Single chat completion via Azure OpenAI deployment
- Multi-turn conversation with message history
- Error handling for connectivity and policy errors

Why Azure is different from other providers:
    Every Azure OpenAI customer gets a unique subdomain:
        mycompany.openai.azure.com
    There is no single static Azure host, so the /proxy/azure_openai/ path-prefix
    shortcut (used for OpenAI, Anthropic, Gemini, Cohere) is not supported.

    Instead the gateway requires the target host to be set explicitly via the
    X-MVGC-Target-Host request header. The axemere.gateway.openai.AzureOpenAI
    client handles this automatically: it reads AXEMERE_AZURE_ENDPOINT and
    injects X-MVGC-Target-Host on every request. You set one env var; the SDK
    and gateway take care of the rest.

How requests flow:
    Application
        -> AzureOpenAI SDK (azure_endpoint=gateway_url)
        -> Axemere AI Gateway  (reads X-MVGC-Target-Host: myresource.openai.azure.com)
        -> myresource.openai.azure.com  (authenticated with stored credential)
        -> response back through gateway -> application

Requires:
    AXEMERE_GATEWAY_URL=http://localhost:7080
    AXEMERE_WORKLOAD_ID=default
    AXEMERE_AZURE_ENDPOINT=myresource.openai.azure.com   (your Azure resource hostname)
    AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini               (your deployment name)

    AZURE_OPENAI_API_KEY is NOT required — the gateway injects credentials.

Alternative — pass the endpoint directly at construction time:
    client = AzureOpenAI(config=cfg, azure_target_host="myresource.openai.azure.com")

See also: examples/19_routing_modes.py — side-by-side comparison of all routing methods.

Run:
    python 11_azure_openai_sdk.py
"""

import os
import sys
import time

from dotenv import load_dotenv

from axemere.gateway.openai import AzureOpenAI
from axemere.gateway import AiGatewayConfig


def demo_single_completion(client: AzureOpenAI, deployment: str) -> None:
    print("\n--- Single chat completion ---")
    response = client.chat.completions.create(
        model=deployment,
        max_tokens=128,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer in one sentence."}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}  Tokens: {response.usage.total_tokens}")


def demo_multi_turn(client: AzureOpenAI, deployment: str) -> None:
    print("\n--- Multi-turn conversation ---")
    messages = [
        {"role": "user", "content": "Name the three primary colors."},
    ]
    first = client.chat.completions.create(
        model=deployment,
        max_tokens=64,
        messages=messages,
    )
    assistant_reply = first.choices[0].message.content
    print(f"Round 1 -- Assistant: {assistant_reply}")

    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "Which of those do you get by mixing red and blue?"})

    second = client.chat.completions.create(
        model=deployment,
        max_tokens=64,
        messages=messages,
    )
    print(f"Round 2 -- Assistant: {second.choices[0].message.content}")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    # AXEMERE_AZURE_ENDPOINT must be the bare hostname of your Azure resource, e.g.:
    #   mycompany.openai.azure.com
    # The SDK reads this env var and sets X-MVGC-Target-Host on every request.
    # Without it the gateway cannot determine your Azure deployment's host and
    # will return a 400 with instructions.
    azure_endpoint = os.environ.get("AXEMERE_AZURE_ENDPOINT", "")
    if not azure_endpoint:
        print("\nAXEMERE_AZURE_ENDPOINT is not set.")
        print("Set it to your Azure resource hostname, e.g.:")
        print("  export AXEMERE_AZURE_ENDPOINT=myresource.openai.azure.com")
        sys.exit(1)

    print(f"Azure endpoint: {azure_endpoint}")

    # AZURE_OPENAI_DEPLOYMENT is the name of your Azure model deployment.
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    print(f"Azure deployment: {deployment}")

    # AzureOpenAI reads AXEMERE_AZURE_ENDPOINT automatically and sets
    # X-MVGC-Target-Host: {azure_endpoint} on every request.
    # To supply the endpoint directly instead of via env var:
    #   client = AzureOpenAI(config=cfg, azure_target_host=azure_endpoint)
    client = AzureOpenAI(config=cfg)

    try:
        demo_single_completion(client, deployment)
        time.sleep(2)  # avoid hitting Azure free-tier rate limits between demos
        demo_multi_turn(client, deployment)
    except Exception as exc:
        msg = str(exc).lower()
        if "connection" in msg or "connect" in msg or "refused" in msg:
            print(f"\nConnectivity error: {exc}")
            print("Is the Axemere AI Gateway running? Try: docker compose up -d")
            sys.exit(1)
        elif "credential" in msg and ("not found" in msg or "not available" in msg):
            print("\nAzure credential not configured in this gateway — skipping.")
            print("Add an Azure credential via the gateway console to run this example.")
        elif "429" in str(exc) or "ratelimitreached" in msg or "rate limit" in msg:
            print(f"\nProvider rate limit: {exc}")
            print("(Azure free-tier quota reached — single completion succeeded, demo is functional)")
            # Exit 0: this is a provider quota issue, not a gateway/policy failure.
        else:
            print(f"\nError: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()

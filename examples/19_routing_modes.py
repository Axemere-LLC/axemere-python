"""Example 19: Gateway Target Routing Modes

Explains and demonstrates the three ways the Axemere AI Gateway determines where
to forward a proxied request, in priority order:

  1. X-MVGC-Target-Host header  — explicit override, works for any host
  2. /proxy/{provider}/ path prefix  — zero-config for fixed-host providers
  3. 400 error with instructions  — gateway refuses to guess

Provider path-prefix support matrix:

  Provider        Path prefix                    Upstream host
  --------------- ------------------------------ ----------------------------
  OpenAI          /proxy/openai/                 api.openai.com
  Anthropic       /proxy/anthropic/              api.anthropic.com
  Gemini          /proxy/gemini/                 generativelanguage.googleapis.com
  Cohere          /proxy/cohere/                 api.cohere.com
  Azure OpenAI    NOT SUPPORTED via path-prefix  requires X-MVGC-Target-Host
                  (customer-specific hostname)

Why Azure is different: every Azure customer gets a unique subdomain such as
  mycompany.openai.azure.com
There is no single static Azure host to embed in the provider registry. The
axemere.gateway.openai.AzureOpenAI SDK client handles this automatically by reading
AXEMERE_AZURE_ENDPOINT and setting X-MVGC-Target-Host on every request.

Required environment variables (all modes):
    AXEMERE_GATEWAY_URL=http://localhost:7080
    
    AXEMERE_WORKLOAD_ID=default

Additional for Mode 1 (path-prefix, OpenAI shown):
    OPENAI_API_KEY is NOT required — gateway manages credentials

Additional for Mode 2 (explicit target host):
    No extra vars — host is passed directly in the request

Additional for Mode 3 (Azure):
    AXEMERE_AZURE_ENDPOINT=myresource.openai.azure.com
    AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

Run:
    python 19_routing_modes.py
"""

import os
import sys

import httpx
import openai as raw_openai
from dotenv import load_dotenv

from axemere.gateway import AiGatewayConfig, ai_gateway_headers
from axemere.gateway.openai import AzureOpenAI, OpenAI


# ---------------------------------------------------------------------------
# Mode 1: Path-prefix routing  (/proxy/{provider}/)
# ---------------------------------------------------------------------------

def demo_path_prefix(cfg: AiGatewayConfig) -> None:
    """Path-prefix routing: set the SDK base URL to /proxy/{provider}/.

    The gateway reads the provider ID from the path, looks it up in the
    provider registry, strips the prefix, and forwards to the canonical host.
    No X-MVGC-Target-Host header is needed — the path alone resolves the target.

    IMPORTANT: Use the raw openai.OpenAI (not axemere.gateway.openai.OpenAI) for path-prefix
    routing. The axemere.gateway.openai wrapper always injects X-MVGC-Target-Host, which takes
    priority over path-based resolution and causes the full /proxy/openai/... path
    to be forwarded verbatim to api.openai.com (resulting in 404). With the raw SDK,
    only org_id and workload_id headers are needed — the path does the rest.
    """
    print("\n=== Mode 1: Path-prefix routing ===")
    print("SDK base_url -> http://gateway/proxy/openai/  =>  api.openai.com")

    # The OpenAI SDK appends /chat/completions (no /v1 prefix) to base_url,
    # so /v1 must be included in the base_url itself. The gateway parses labels
    # from the path and stops at the first non-label segment (v1), leaving
    # /v1/chat/completions as the upstream path forwarded to api.openai.com.
    # The Anthropic SDK includes /v1 in its endpoint paths so no /v1 is needed there.
    client = raw_openai.OpenAI(
        base_url=f"{cfg.gateway_url}/proxy/openai/v1",
        api_key="any-value",  # real key is held by the gateway
        default_headers={
            "X-MVGC-Workload-ID": cfg.workload_id,
        },
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'path-prefix routing works' in one sentence."}],
    )
    print(f"Response: {response.choices[0].message.content}")


# ---------------------------------------------------------------------------
# Mode 2: Explicit X-MVGC-Target-Host header
# ---------------------------------------------------------------------------

def demo_explicit_target_host(cfg: AiGatewayConfig) -> None:
    """Explicit header routing: set X-MVGC-Target-Host on the request.

    This is the escape hatch for any host not in the provider registry:
    custom OpenAI-compatible endpoints, self-hosted models, Azure OpenAI,
    or any scenario where you want to override the path-prefix resolution.

    The gateway uses this header as the highest-priority routing signal.
    """
    print("\n=== Mode 2: Explicit X-MVGC-Target-Host header ===")
    print("Header: X-MVGC-Target-Host: api.openai.com  (explicit override)")

    headers = {
        **ai_gateway_headers(cfg, target_host="api.openai.com"),
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say 'explicit routing works' in one sentence."}],
        "stream": False,
    }

    response = httpx.post(
        f"{cfg.gateway_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['choices'][0]['message']['content']}")
    else:
        print(f"Status {response.status_code}: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Mode 3: Azure OpenAI — SDK sets X-MVGC-Target-Host from AXEMERE_AZURE_ENDPOINT
# ---------------------------------------------------------------------------

def demo_azure(cfg: AiGatewayConfig) -> None:
    """Azure routing: AXEMERE_AZURE_ENDPOINT env var drives X-MVGC-Target-Host.

    Azure OpenAI uses customer-specific hostnames (e.g., myresource.openai.azure.com).
    There is no single static Azure host so /proxy/azure_openai/ is NOT supported —
    the gateway returns a 400 directing you here instead.

    The axemere.gateway.openai.AzureOpenAI SDK client reads AXEMERE_AZURE_ENDPOINT (or the
    azure_target_host constructor argument) and automatically sets
    X-MVGC-Target-Host on every request. From the application's perspective,
    this is just as zero-config as path-prefix: set one env var and it works.

    What the SDK does internally:
        target_host = AXEMERE_AZURE_ENDPOINT  (e.g. "myresource.openai.azure.com")
        X-MVGC-Target-Host: myresource.openai.azure.com
        azure_endpoint: http://gateway_url  (SDK sends to gateway, not Azure directly)
    """
    print("\n=== Mode 3: Azure OpenAI — X-MVGC-Target-Host set by SDK ===")

    azure_endpoint = os.environ.get("AXEMERE_AZURE_ENDPOINT", "")
    if not azure_endpoint:
        print("AXEMERE_AZURE_ENDPOINT not set — skipping Azure demo.")
        print("Set AXEMERE_AZURE_ENDPOINT=myresource.openai.azure.com to run this section.")
        return

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    print(f"Azure endpoint: {azure_endpoint}  Deployment: {deployment}")
    print(f"SDK will set X-MVGC-Target-Host: {azure_endpoint}")

    client = AzureOpenAI(config=cfg)
    # azure_endpoint is read from AXEMERE_AZURE_ENDPOINT automatically.
    # To override explicitly: AzureOpenAI(config=cfg, azure_target_host="myresource.openai.azure.com")

    try:
        response = client.chat.completions.create(
            model=deployment,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say 'Azure routing works' in one sentence."}],
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as exc:
        msg = str(exc)
        if "credential" in msg.lower() or "credential not found" in msg.lower():
            print(f"Azure credential not configured in this gateway — skipping.")
            print("Add an Azure credential to the gateway to enable this section.")
            return
        raise


# ---------------------------------------------------------------------------
# Show what happens when path-prefix is attempted for Azure
# ---------------------------------------------------------------------------

def demo_azure_path_prefix_error(cfg: AiGatewayConfig) -> None:
    """Show the actionable 400 returned when /proxy/azure_openai/ is used.

    The gateway detects azure_openai in the path registry, knows it has no
    static upstream host, and returns a 400 with instructions rather than
    silently failing or guessing a host.
    """
    print("\n=== What happens when you try /proxy/azure_openai/ ===")

    headers = {
        **ai_gateway_headers(cfg),
        "Content-Type": "application/json",
    }
    response = httpx.post(
        f"{cfg.gateway_url}/proxy/azure_openai/v1/chat/completions",
        headers=headers,
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]},
        timeout=10,
    )
    print(f"Status: {response.status_code}  (expected 400)")
    print(f"Body: {response.text[:300]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    try:
        demo_path_prefix(cfg)
        demo_explicit_target_host(cfg)
        demo_azure(cfg)
        demo_azure_path_prefix_error(cfg)
    except httpx.ConnectError as exc:
        print(f"\nConnectivity error: {exc}")
        print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        sys.exit(1)
    except Exception as exc:
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg:
            print(f"\nConnectivity error: {exc}")
            print("Is the Axemere AI Gateway running? Try: docker compose up -d")
            sys.exit(1)
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

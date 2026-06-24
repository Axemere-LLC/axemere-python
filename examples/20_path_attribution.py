"""Example 20: Transparent Proxy Path Attribution

Demonstrates encoding workload_id, project_id, account_id, and customer_id
directly in the SDK base URL path — no per-request headers, no middleware,
no Axemere AI Gateway-specific imports.

URL format:
    /proxy/{provider}[/w/{workload_id}][/p/{project_id}][/a/{account_id}][/c/{customer_id}]/

The gateway strips all labeled segments before forwarding to the upstream
provider. Attribution flows automatically on every request from that client.

Patterns covered:
  1. Anthropic SDK — workload + project encoded in base URL
  2. OpenAI SDK   — workload + project encoded in base URL
  3. Per-customer SaaS — one client instance per tenant, customer_id in path
  4. All four fields  — order-independent, same result
  5. Per-request override — X-MVGC-* header beats path value for one request

Priority waterfall (highest → lowest):
    X-MVGC-* header  >  path segment  >  workload default_attribution  >  gateway default

Required environment variables:
    AXEMERE_GATEWAY_URL=http://localhost:7080   (default shown)
    AXEMERE_WORKLOAD_ID=wl-prod-app-1          (must be registered)

Run:
    python 20_path_attribution.py
"""

import os
import sys

import anthropic
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load .env before reading any env vars so values set in .env are available.
load_dotenv()

GATEWAY_URL = os.environ.get("AXEMERE_GATEWAY_URL", "http://localhost:7080")
WORKLOAD_ID = os.environ.get("AXEMERE_WORKLOAD_ID", "wl-prod-app-1")


def _http_client(**extra_headers: str) -> httpx.Client:
    """Build a shared httpx.Client with any per-call extra headers."""
    headers = {k: v for k, v in extra_headers.items() if v}
    return httpx.Client(headers=headers)


# ---------------------------------------------------------------------------
# Pattern 1: Anthropic SDK — workload + project in base URL
# ---------------------------------------------------------------------------

def demo_anthropic_path_attribution() -> None:
    """Workload and project encoded once in the client; every request is attributed automatically."""
    print("\n--- Pattern 1: Anthropic SDK with path attribution ---")

    client = anthropic.Anthropic(
        base_url=f"{GATEWAY_URL}/proxy/anthropic/w/{WORKLOAD_ID}/p/proj-q3",
        api_key="any-value",  # real key is held by the gateway
        http_client=_http_client(),
    )

    print(f"Base URL: {GATEWAY_URL}/proxy/anthropic/w/{WORKLOAD_ID}/p/proj-q3")
    print("Attribution: workload_id + project_id encoded in path — no headers needed")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'path attribution works' in one sentence."}],
    )
    print(f"Response: {response.content[0].text}")


# ---------------------------------------------------------------------------
# Pattern 2: OpenAI SDK — workload + project in base URL
# ---------------------------------------------------------------------------

def demo_openai_path_attribution() -> None:
    """Same pattern with the OpenAI SDK."""
    print("\n--- Pattern 2: OpenAI SDK with path attribution ---")

    # OpenAI SDK appends /chat/completions (no /v1) to base_url, so /v1 must be
    # in the base_url after attribution labels. Anthropic SDK includes /v1 in its
    # endpoint paths so no /v1 is needed there.
    client = OpenAI(
        base_url=f"{GATEWAY_URL}/proxy/openai/w/{WORKLOAD_ID}/p/proj-analytics/v1",
        api_key="any-value",
        http_client=_http_client(),
    )

    print(f"Base URL: {GATEWAY_URL}/proxy/openai/w/{WORKLOAD_ID}/p/proj-analytics/v1")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'path attribution works' in one sentence."}],
    )
    print(f"Response: {response.choices[0].message.content}")


# ---------------------------------------------------------------------------
# Pattern 3: Per-customer SaaS — one client per tenant
# ---------------------------------------------------------------------------

def make_client(customer_id: str) -> anthropic.Anthropic:
    """Construct a gateway-backed Anthropic client scoped to one customer.

    customer_id is encoded in the base URL at construction time.
    No per-request header injection is needed — attribution flows automatically.
    Use opaque IDs (UUIDs, hashed IDs) rather than business names: customer_id
    appears in HTTP access logs and distributed traces.
    """
    return anthropic.Anthropic(
        base_url=f"{GATEWAY_URL}/proxy/anthropic/w/{WORKLOAD_ID}/c/{customer_id}",
        api_key="any-value",
        http_client=_http_client(),
    )


def demo_per_customer_saas() -> None:
    """Each tenant gets its own client; spend reports group by customer_id automatically."""
    print("\n--- Pattern 3: Per-customer SaaS ---")

    acme_client   = make_client("cust-acme-corp")
    globex_client = make_client("cust-globex-ind")

    print(f"ACME base URL:   {GATEWAY_URL}/proxy/anthropic/w/{WORKLOAD_ID}/c/cust-acme-corp")
    print(f"Globex base URL: {GATEWAY_URL}/proxy/anthropic/w/{WORKLOAD_ID}/c/cust-globex-ind")
    print("Both calls below are attributed to different customer_ids automatically")

    acme_resp = acme_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi"}],
    )
    globex_resp = globex_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(f"ACME response:   {acme_resp.content[0].text}")
    print(f"Globex response: {globex_resp.content[0].text}")


# ---------------------------------------------------------------------------
# Pattern 4: All four fields — order-independent
# ---------------------------------------------------------------------------

def demo_all_fields() -> None:
    """All four attribution fields encoded; segment order does not matter."""
    print("\n--- Pattern 4: All four fields (order-independent) ---")

    # These two base URLs produce identical attribution results:
    url_a = f"{GATEWAY_URL}/proxy/anthropic/w/{WORKLOAD_ID}/p/proj-abc/a/acct-eng/c/cust-xyz"
    url_b = f"{GATEWAY_URL}/proxy/anthropic/c/cust-xyz/w/{WORKLOAD_ID}/a/acct-eng/p/proj-abc"

    client = anthropic.Anthropic(base_url=url_a, api_key="any-value", http_client=_http_client())
    print(f"Using: {url_a}")
    print(f"Same as: {url_b}")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(f"Response: {response.content[0].text}")


# ---------------------------------------------------------------------------
# Pattern 5: Per-request override via X-MVGC-* header
# ---------------------------------------------------------------------------

def demo_header_override() -> None:
    """Header beats path for a single request — useful for debugging or testing.

    The base URL sets project_id=proj-default for all requests.
    One request overrides project_id via X-MVGC-Project-ID header without
    changing the client or the base URL.
    """
    print("\n--- Pattern 5: Per-request header override ---")

    client = anthropic.Anthropic(
        base_url=f"{GATEWAY_URL}/proxy/anthropic/w/{WORKLOAD_ID}/p/proj-default",
        api_key="any-value",
        http_client=_http_client(**{"X-MVGC-Project-ID": "proj-special-debug"}),
    )

    print(f"Path encodes project_id=proj-default")
    print(f"X-MVGC-Project-ID: proj-special-debug  (header wins for this request)")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(f"Response: {response.content[0].text}")
    print("This request was attributed to proj-special-debug, not proj-default")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Gateway: {GATEWAY_URL}")
    print(f"Workload: {WORKLOAD_ID}")

    try:
        demo_anthropic_path_attribution()
        demo_openai_path_attribution()
        demo_per_customer_saas()
        demo_all_fields()
        demo_header_override()
    except anthropic.APIConnectionError as exc:
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

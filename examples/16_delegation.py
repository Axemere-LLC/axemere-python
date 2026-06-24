"""Example 16: Delegation Tokens

Demonstrates:
- The structure of an Axemere AI Gateway delegation token (wire spec §4)
- Creating a signed delegation token using an Ed25519 key pair
- Including a delegation token in an explicit action request
- How the gateway enforces token scope (action type restriction)
- How the gateway enforces token budget constraints

Delegation tokens allow an issuing service to grant a narrowed set of
permissions to a sub-workload or principal without handing over the full
workload credentials.  In production, tokens are issued by a governance
service that holds the private signing key configured in the gateway via
MVGC_DELEGATION_VERIFY_KEY.  This example generates an ephemeral key pair
for illustration; requests will be denied by a real gateway unless you
configure matching keys.

Run:
    python 16_delegation.py

Requirements:
    pip install cryptography
"""

import base64
import datetime
import json
import sys
import uuid

import httpx
from dotenv import load_dotenv

from axemere.gateway import AiGatewayConfig

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )
except ImportError:
    print("cryptography package is required: pip install cryptography", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Token creation helpers
# ---------------------------------------------------------------------------

def _jcs_encode(obj: dict) -> bytes:
    """Minimal JCS (RFC 8785) encoder: sort keys, no spaces, Unicode escaping."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def create_delegation_token(
    private_key: Ed25519PrivateKey,
    org_id: str,
    workload_id: str,
    issued_by: str,
    audience: str,
    ttl_seconds: int = 3600,
    actions_allow: list[str] | None = None,
    targets_allow: list[str] | None = None,
    usd_max: str = "",
    max_requests: int = 0,
) -> str:
    """Build a signed Axemere AI Gateway delegation token and return it as a JSON string.

    The token schema follows wire spec mvgc.delegation.v2.
    The Sig field is appended after JCS-canonicalizing the unsigned body.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    expires = now + datetime.timedelta(seconds=ttl_seconds)

    unsigned: dict = {
        "schema": "mvgc.delegation.v2",
        "org_id": org_id,
        "workload_id": workload_id,
        "delegation_id": str(uuid.uuid4()),
        "jti": str(uuid.uuid4()),
        "issued_by": issued_by,
        "issued_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audience": audience,
        "scope": {
            "actions_allow": actions_allow or [],
            "targets_allow": targets_allow or [],
        },
        "budget": {
            **({"usd_max": usd_max} if usd_max else {}),
            **({"max_requests": max_requests} if max_requests > 0 else {}),
        },
        "depth": 0,
    }

    canonical = _jcs_encode(unsigned)
    raw_sig = private_key.sign(canonical)

    pub_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    key_id = base64.urlsafe_b64encode(pub_bytes[:8]).rstrip(b"=").decode()

    signed = {
        **unsigned,
        "sig": {
            "key_id": key_id,
            "algorithm": "ed25519",
            "sig": base64.b64encode(raw_sig).decode(),
        },
    }
    return json.dumps(signed, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _action_request(cfg: AiGatewayConfig, delegation_token: str, target_host: str, params: dict) -> dict:
    return {
        "schema": "mvgc.action_request.v2",
        "org_id": "",
        "workload_id": cfg.workload_id,
        "ingress_mode": "explicit_action_request",
        "delegation_token": delegation_token,
        "action": {
            "type": "ai.infer",
            "method": "POST",
            "target_host": target_host,
            "target_path": "/v1/chat/completions",
            "params": params,
        },
        "attribution": {
            "project_id": cfg.project_id,
            "customer_id": cfg.customer_id,
            "account_id": cfg.account_id,
            "labels": {"source": "delegation-example"},
        },
    }


def post_action(cfg: AiGatewayConfig, payload: dict) -> tuple[int, dict]:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{cfg.gateway_url}/v1/actions:execute", json=payload)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return resp.status_code, body


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_show_token_structure(token_json: str) -> None:
    print("\n--- Delegation token structure ---")
    tok = json.loads(token_json)
    # Omit sig for readability.
    display = {k: v for k, v in tok.items() if k != "sig"}
    print(json.dumps(display, indent=2))
    print(f"  [sig.key_id={tok['sig']['key_id']}  algorithm={tok['sig']['algorithm']}]")


def demo_delegated_request(cfg: AiGatewayConfig, token_json: str) -> None:
    print("\n--- Delegated request (ai.infer allowed) ---")
    payload = _action_request(
        cfg,
        token_json,
        "api.openai.com",
        {
            "model": "gpt-4o-mini",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say 'hello from delegation' in one sentence."}],
        },
    )
    status, body = post_action(cfg, payload)
    print(f"HTTP {status}  decision={body.get('decision', '?')}")
    if body.get("result"):
        result_body = body["result"].get("body", {})
        choices = result_body.get("choices", []) if isinstance(result_body, dict) else []
        if choices:
            print(f"Response: {choices[0]['message']['content']}")
    elif body.get("reason"):
        print(f"Reason: {body['reason']}")


def demo_scope_enforcement(cfg: AiGatewayConfig, private_key: Ed25519PrivateKey) -> None:
    """Create a token that only allows ai.embed — then try ai.infer (should be denied)."""
    print("\n--- Scope enforcement: ai.infer blocked by token (only ai.embed allowed) ---")
    embed_only_token = create_delegation_token(
        private_key,
        org_id="",
        workload_id=cfg.workload_id,
        issued_by="governance-service",
        audience=cfg.workload_id,
        ttl_seconds=300,
        actions_allow=["ai.embed"],  # does NOT include ai.infer
        targets_allow=["api.openai.com"],
    )
    payload = _action_request(
        cfg,
        embed_only_token,
        "api.openai.com",
        {
            "model": "gpt-4o-mini",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    status, body = post_action(cfg, payload)
    print(f"HTTP {status}  decision={body.get('decision', '?')}")
    print(f"Reason: {body.get('reason', '—')}")


def demo_budget_constrained_token(cfg: AiGatewayConfig, private_key: Ed25519PrivateKey) -> None:
    """Create a token with a $0.001 budget cap (tiny, to demonstrate budget enforcement)."""
    print("\n--- Budget-constrained delegation token ($0.001 cap) ---")
    micro_budget_token = create_delegation_token(
        private_key,
        org_id="",
        workload_id=cfg.workload_id,
        issued_by="governance-service",
        audience=cfg.workload_id,
        ttl_seconds=300,
        actions_allow=["ai.infer"],
        usd_max="0.001",
        max_requests=1,
    )
    tok = json.loads(micro_budget_token)
    print(f"Token budget: usd_max={tok['budget'].get('usd_max')}  max_requests={tok['budget'].get('max_requests')}")

    payload = _action_request(
        cfg,
        micro_budget_token,
        "api.openai.com",
        {
            "model": "gpt-4o-mini",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Write a 400-word essay about governance."}],
        },
    )
    status, body = post_action(cfg, payload)
    print(f"HTTP {status}  decision={body.get('decision', '?')}")
    if body.get("reason"):
        print(f"Reason: {body['reason']}")
    elif body.get("result"):
        print("Request succeeded (budget not yet exceeded at token level)")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")
    print()
    print("NOTE: This example uses an ephemeral Ed25519 key pair.")
    print("      A real gateway will reject tokens unless its MVGC_DELEGATION_VERIFY_KEY")
    print("      matches the public key used here.  The demos below show the wire format")
    print("      and scope enforcement logic independent of gateway key config.")

    # Generate an ephemeral signing key.  In production, the private key lives
    # in the issuing governance service; the public key is loaded by the gateway.
    private_key = Ed25519PrivateKey.generate()

    token_json = create_delegation_token(
        private_key,
        org_id="",
        workload_id=cfg.workload_id,
        issued_by="governance-service",
        audience=cfg.workload_id,
        ttl_seconds=3600,
        actions_allow=["ai.infer", "ai.embed"],
        targets_allow=["api.openai.com", "api.anthropic.com"],
        usd_max="5.00",
    )

    demo_show_token_structure(token_json)
    demo_delegated_request(cfg, token_json)
    demo_scope_enforcement(cfg, private_key)
    demo_budget_constrained_token(cfg, private_key)


if __name__ == "__main__":
    main()

"""Example 17: Governance Outcomes Beyond Allow/Deny

Demonstrates:
- HTTP 202 require_approval: request held for human review; poll for decision
- HTTP 429 rate_limit: budget or rate cap exceeded; Retry-After header
- HTTP 403 quarantine: workload flagged; request blocked with quarantine_id
- HTTP 403 deny: standard policy denial
- Polling the approval endpoint until a decision is made
- Reading decision_trace from every gateway response

These outcomes are triggered by policy rules in the active bundle.  The demo
sends requests whose attributes are crafted to match specific rule conditions
as set up in the mock gateway used for CI.  Against a real gateway you would
configure the policy bundle to produce each outcome.

Run:
    python 17_governance_outcomes.py
"""

import sys
import time

import httpx
from dotenv import load_dotenv

from axemere.gateway import AiGatewayConfig

ADMIN_TOKEN_ENV = "MVGC_ADMIN_TOKEN"


# ---------------------------------------------------------------------------
# Low-level request helper
# ---------------------------------------------------------------------------

def post_action(cfg: AiGatewayConfig, extra_fields: dict | None = None) -> tuple[int, dict, dict]:
    """POST an explicit action request.  Returns (status, body, headers)."""
    payload: dict = {
        "schema": "mvgc.action_request.v2",
        "org_id": "",
        "workload_id": cfg.workload_id,
        "ingress_mode": "explicit_action_request",
        "action": {
            "type": "ai.infer",
            "method": "POST",
            "target_host": "api.openai.com",
            "target_path": "/v1/chat/completions",
            "params": {
                "model": "gpt-4o-mini",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Hello."}],
            },
        },
        "attribution": {
            "project_id": cfg.project_id,
            "customer_id": cfg.customer_id,
            "account_id": cfg.account_id,
            "labels": {"source": "governance-outcomes-example"},
        },
    }
    if extra_fields:
        payload.update(extra_fields)

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{cfg.gateway_url}/v1/actions:execute", json=payload)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return resp.status_code, body, dict(resp.headers)


# ---------------------------------------------------------------------------
# Approval polling
# ---------------------------------------------------------------------------

def poll_approval(cfg: AiGatewayConfig, approval_id: str, admin_token: str, max_polls: int = 6) -> dict:
    """Poll GET /v1/admin/approvals/{id} until a terminal decision is reached."""
    with httpx.Client(timeout=10.0) as client:
        for attempt in range(1, max_polls + 1):
            resp = client.get(
                f"{cfg.gateway_url}/v1/admin/approvals/{approval_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            if resp.status_code != 200:
                print(f"  Poll {attempt}: HTTP {resp.status_code} — {resp.text[:100]}")
                break
            data = resp.json()
            status = data.get("status", "unknown")
            print(f"  Poll {attempt}: status={status}")
            if status in ("approved", "denied"):
                return data
            time.sleep(2)
    return {}


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_policy_deny(cfg: AiGatewayConfig) -> None:
    """Standard deny: omit project_id to trigger a missing-attribution deny."""
    print("\n--- HTTP 403 deny: missing project_id ---")
    no_proj_cfg = AiGatewayConfig(
        gateway_url=cfg.gateway_url,
        workload_id=cfg.workload_id,
        project_id="",
        customer_id=cfg.customer_id,
        account_id=cfg.account_id,
    )
    status, body, _ = post_action(no_proj_cfg)
    print(f"HTTP {status}  decision={body.get('decision', '?')}")
    print(f"Reason: {body.get('reason', body.get('error', '—'))}")
    trace = body.get("decision_trace", {})
    if trace.get("reason_codes"):
        print(f"Reason codes: {trace['reason_codes']}")


def demo_require_approval(cfg: AiGatewayConfig, admin_token: str) -> None:
    """HTTP 202: policy rule returns require_approval.

    The gateway returns 202 Accepted with an approval_id.  A human reviewer
    calls POST /v1/admin/approvals/{id}/approve or /deny.  Here we poll for
    the decision after auto-approving via the admin API.
    """
    print("\n--- HTTP 202 require_approval ---")
    # Use a label that triggers a require_approval rule in the mock gateway.
    status, body, _ = post_action(
        cfg,
        extra_fields={
            "attribution": {
                "project_id": cfg.project_id,
                "customer_id": cfg.customer_id,
                "account_id": cfg.account_id,
                "labels": {"risk_level": "high"},  # triggers require_approval in mock policy
            }
        },
    )
    print(f"HTTP {status}  decision={body.get('decision', '?')}")

    if status == 202:
        approval_id = body.get("approval_id", "")
        record_id = body.get("record_id", "")
        print(f"approval_id: {approval_id}")
        print(f"record_id:   {record_id}")

        if admin_token and approval_id:
            print("\nAuto-approving via admin API (simulating human review)…")
            with httpx.Client(timeout=10.0) as client:
                approve_resp = client.post(
                    f"{cfg.gateway_url}/v1/admin/approvals/{approval_id}/approve",
                    json={"decided_by": "example-script"},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                print(f"Approve response: HTTP {approve_resp.status_code}")

            print("Polling for final status…")
            result = poll_approval(cfg, approval_id, admin_token)
            if result:
                print(f"Final status: {result.get('status')}  decided_by={result.get('decided_by')}")
        else:
            print("(set MVGC_ADMIN_TOKEN to auto-approve and poll the decision)")
    elif status == 403:
        print("Gateway returned 403 — MVGC_APPROVAL_ENABLED may be false (legacy mode)")
        print(f"Reason: {body.get('reason', '—')}")
    else:
        print(f"Unexpected status; body: {body}")


def demo_rate_limit(cfg: AiGatewayConfig) -> None:
    """HTTP 429: send enough requests quickly to trigger the rate limiter.

    In most deployments the per-workload rate limit is set to handle bursts;
    this demo fires 50 rapid requests and reports the first 429 it receives.
    """
    print("\n--- HTTP 429 rate_limit: rapid-fire requests ---")
    for i in range(1, 51):
        status, body, headers = post_action(cfg)
        if status == 429:
            retry_after = headers.get("retry-after", "?")
            print(f"  Request {i}: HTTP 429  Retry-After: {retry_after}s")
            print(f"  decision={body.get('decision', '?')}  reason={body.get('reason', '—')}")
            return
        if i % 10 == 0:
            print(f"  Sent {i} requests — no 429 yet (decision={body.get('decision', '?')})")
    print("  Sent 50 requests without receiving a 429.")
    print("  Check that your gateway policy has a rate_limit rule for this workload.")


def demo_quarantine(cfg: AiGatewayConfig) -> None:
    """HTTP 403 quarantine: a workload flagged by the risk scorer.

    Quarantine returns the same HTTP 403 as a standard deny, but the
    decision field in the response body is 'quarantine' and the record
    is persisted to the quarantine store for admin review.
    """
    print("\n--- HTTP 403 quarantine ---")
    # Use a label combination that triggers the quarantine rule in mock policy.
    status, body, _ = post_action(
        cfg,
        extra_fields={
            "attribution": {
                "project_id": cfg.project_id,
                "customer_id": cfg.customer_id,
                "account_id": cfg.account_id,
                "labels": {"source": "governance-outcomes-example", "quarantine_trigger": "true"},
            }
        },
    )
    print(f"HTTP {status}  decision={body.get('decision', '?')}")
    if body.get("decision") == "quarantine":
        print(f"record_id: {body.get('record_id', '—')}")
        print("The record is stored in the quarantine table.")
        print("Admin can review via: GET /v1/admin/quarantine")
        print("Admin can release via: POST /v1/admin/quarantine/{id}/release")
    elif status == 403:
        print(f"Reason: {body.get('reason', body.get('error', '—'))}")
        print("(configure a quarantine rule in your policy bundle to see 'quarantine' decision)")
    else:
        print(f"Unexpected status; body: {body}")


def demo_decision_trace(cfg: AiGatewayConfig) -> None:
    """Every gateway response includes a decision_trace with matched rules."""
    print("\n--- decision_trace on a successful allow ---")
    status, body, _ = post_action(cfg)
    print(f"HTTP {status}  decision={body.get('decision', '?')}")
    trace = body.get("decision_trace", {})
    if trace:
        print(f"Schema:       {trace.get('schema', '—')}")
        print(f"Decision:     {trace.get('decision', '—')}")
        print(f"Reason codes: {trace.get('reason_codes', [])}")
        print(f"Evaluated at: {trace.get('evaluated_at', '—')}")
        attrs = trace.get("attributes", {})
        if attrs:
            for k, v in list(attrs.items())[:4]:
                print(f"  {k}: {v}")
    else:
        print("No decision_trace in response.")


def main() -> None:
    load_dotenv()
    import os

    cfg = AiGatewayConfig.from_env()
    admin_token = os.environ.get(ADMIN_TOKEN_ENV, "")
    print(f"Gateway:      {cfg.gateway_url}")
    print(f"Workload:     {cfg.workload_id}  Project: {cfg.project_id}")
    print(f"Admin token:  {'set' if admin_token else 'not set (set MVGC_ADMIN_TOKEN to enable approval polling)'}")

    try:
        demo_policy_deny(cfg)
        demo_decision_trace(cfg)
        demo_require_approval(cfg, admin_token)
        demo_rate_limit(cfg)
        demo_quarantine(cfg)
    except httpx.ConnectError as exc:
        print(f"\nConnectivity error: {exc}")
        print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

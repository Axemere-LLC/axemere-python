"""Example 14: Streaming request via Axemere AI Gateway

Demonstrates:
- Sending a streaming (SSE) request through the Axemere AI Gateway
- Parsing the raw SSE stream and printing content deltas as they arrive
- Using explicit action request mode with stream=True in action.params

Run:
    python 14_streaming.py
"""

import json
import sys

import httpx

from axemere.gateway import AiGatewayConfig


def demo_streaming(cfg: AiGatewayConfig) -> None:
    print("\n--- OpenAI streaming via Axemere AI Gateway (explicit mode) ---")

    action_request = {
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
                "messages": [{"role": "user", "content": "Tell me a short story."}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        },
        "attribution": {
            "project_id": cfg.project_id,
            "customer_id": cfg.customer_id,
            "account_id": cfg.account_id,
            "labels": {"source": "streaming-example"},
        },
    }

    # When stream=True, the gateway returns a raw SSE stream (text/event-stream)
    # instead of a JSON ConnectorResultDTO envelope.
    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{cfg.gateway_url}/v1/actions:execute",
            json=action_request,
        ) as response:
            if response.status_code != 200:
                response.read()
                print(f"HTTP {response.status_code}: {response.text}", file=sys.stderr)
                sys.exit(1)

            print(f"Content-Type: {response.headers.get('content-type')}\n")

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)

    print()  # newline after stream


def main() -> None:
    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    demo_streaming(cfg)


if __name__ == "__main__":
    main()

# Axemere AI Gateway — Python SDK

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Add governance, cost controls, and audit trails to every AI API call — without changing your application code.

**[Get started at axemere.ai →](https://axemere.ai)**

---

## What is Axemere?

The [Axemere AI Gateway](https://axemere.ai) sits between your application and AI providers (OpenAI, Anthropic, Gemini, and more). It enforces spend budgets, policy rules, and delegation controls, and records every request to an append-only audit ledger — all transparent to your existing SDK calls.

- **Drop-in replacement** — swap one import and your existing OpenAI, Anthropic, or Gemini code works unchanged
- **Cost controls** — per-workload and per-project spend limits enforced at the gateway
- **Policy engine** — allow/deny rules based on model, provider, context, and labels
- **Audit ledger** — tamper-evident record of every request, token count, and cost
- **Delegation** — issue scoped tokens to users or agents with budget and model limits

## Packages

| Package | Install | Use when |
|---------|---------|----------|
| `axemere-gateway` | `pip install axemere-gateway` | Framework-independent; explicit API |
| `axemere-gateway-openai` | `pip install axemere-gateway-openai` | You use `openai` SDK today |
| `axemere-gateway-anthropic` | `pip install axemere-gateway-anthropic` | You use `anthropic` SDK today |
| `axemere-gateway-google` | `pip install axemere-gateway-google` | You use `google-genai` SDK today |
| `axemere-gateway-langchain` | `pip install axemere-gateway-langchain` | You use LangChain |

## Quick start

```bash
pip install axemere-gateway-openai
```

```python
from axemere.gateway.openai import OpenAI

client = OpenAI()  # reads AXEMERE_GATEWAY_URL + AXEMERE_WORKLOAD_TOKEN

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

No other code changes needed. Every request is now governed, costed, and recorded.

## Get a gateway

- **Free tier** — self-hosted, single user, no account required. [Install →](https://axemere.ai/docs/free-gateway)
- **Self-Hosted** — team features, policy engine, full audit ledger. [Get started →](https://axemere.ai/docs/get-started/cp-connected)
- **Managed** — fully hosted, multi-tenant, SOC 2 ready. [Get started →](https://axemere.ai/docs/get-started/managed-gateway)

## Examples

See [`examples/`](examples/) for 20 runnable scripts covering all providers, streaming, delegation, governance outcomes, and proxy mode.

```bash
cd examples
pip install -r requirements.txt
python run_all.py
```

## Requirements

- Python 3.10+
- A running Axemere AI Gateway (`AXEMERE_GATEWAY_URL`)
- A workload token (`AXEMERE_WORKLOAD_TOKEN`)

## Links

- Website: [axemere.ai](https://axemere.ai)
- Docs: [axemere.ai/docs](https://axemere.ai/docs)
- Issues: [github.com/Axemere-LLC/axemere-python/issues](https://github.com/Axemere-LLC/axemere-python/issues)

## License

MIT — see [LICENSE](LICENSE).

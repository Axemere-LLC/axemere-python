# Axemere AI Gateway — Python Examples

Twenty examples covering the gateway's core capabilities plus Python-ecosystem integrations. Each runs standalone against a local or managed gateway.

## Setup

```bash
cp .env.example .env
# Edit .env to match your gateway

# Install all SDK packages (editable, from the repo)
pip install -e ../sdk/python/packages/axemere-gateway
pip install -e ../sdk/python/packages/axemere-gateway-openai
pip install -e ../sdk/python/packages/axemere-gateway-anthropic
pip install -e ../sdk/python/packages/axemere-gateway-langchain
pip install -e ../sdk/python/packages/axemere-gateway-google
pip install -e ../sdk/python/packages/axemere-gateway-llamaindex

python 01_basic_chat.py
```

Or run all examples and see a pass/fail summary:

```bash
python run_all.py
# Skip specific examples:  --skip 10,11
# Run only specific ones:  --only 1,2,3
```

## Examples

| # | File | What it demonstrates | Providers |
|---|------|----------------------|-----------|
| 01 | `01_basic_chat.py` | `AiGatewayClient.execute()` against OpenAI and Anthropic; `PolicyDeniedError` when `project_id` is omitted | OpenAI, Anthropic |
| 02 | `02_proxy_chat.py` | `ai_gateway_openai_client()` and `ai_gateway_anthropic_client()` proxy wrappers; attribution headers auto-injected | OpenAI, Anthropic |
| 03 | `03_chain_pipeline.py` | LangChain `ChatAiGateway` LLM in an LCEL chain; prompt template → LLM → output parser | OpenAI |
| 04 | `04_tool_agent.py` | LangChain `initialize_agent` with tool calling; `ChatAiGateway` as the reasoning LLM | OpenAI |
| 05 | `05_multi_provider.py` | Same question to OpenAI and Anthropic; `cfg.set_defaults()` to switch provider; spend comparison | OpenAI, Anthropic |
| 06 | `06_rag_retrieval.py` | LlamaIndex `AiGatewayLLM` in a RAG pipeline; embed a document, then query it | OpenAI |
| 07 | `07_openai_sdk.py` | `ai_gateway_openai_client()` returns a drop-in `openai.OpenAI`; single completion; multi-turn history | OpenAI |
| 08 | `08_anthropic_sdk.py` | `ai_gateway_anthropic_client()` returns a drop-in `anthropic.Anthropic`; single message; multi-turn history | Anthropic |
| 09 | `09_http_client.py` | Raw `httpx` calls to the proxy path with explicit `X-MVGC-*` headers; no SDK wrapper | OpenAI |
| 10 | `10_gemini_sdk.py` | `google-generativeai` SDK pointed at the gateway's Gemini proxy path | Gemini |
| 11 | `11_azure_openai_sdk.py` | `AzureOpenAI` from `axemere-gateway-openai`; sets `X-MVGC-Target-Host` from `AXEMERE_AZURE_ENDPOINT` | Azure OpenAI |
| 12 | `12_llamaindex_sdk.py` | `AiGatewayLLM` from `axemere-gateway-llamaindex`; `complete()` and `chat()` | OpenAI |
| 13 | `13_instructor.py` | Instructor structured outputs using the gateway proxy as the OpenAI base URL | OpenAI, Anthropic |
| 14 | `14_streaming.py` | `AiGatewayClient.execute(stream=True)` async generator; token-by-token output; metering event | OpenAI |
| 15 | `15_cohere_sdk.py` | `cohere.Client` pointed at the gateway's Cohere proxy path; single chat and streaming | Cohere |
| 16 | `16_delegation.py` | Ed25519 key generation; `mvgc.delegation.v2` wire token; scope enforcement; budget cap | OpenAI |
| 17 | `17_governance_outcomes.py` | HTTP 202 approval-required (+ polling); 403 policy deny; 429 rate-limit with Retry-After | OpenAI |
| 18 | `18_async.py` | `asyncio.gather()` fan-out across concurrent `AiGatewayClient` calls; LangChain `.astream()` | OpenAI |
| 19 | `19_routing_modes.py` | Path-prefix routing; explicit `X-MVGC-Target-Host`; Azure SDK; `/proxy/azure_openai/` 400 demo | OpenAI, Azure |
| 20 | `20_path_attribution.py` | Workload, project, account, customer, and label attribution; per-call override; proxy-URL path encoding | OpenAI, Anthropic |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AXEMERE_GATEWAY_URL` | Yes | Gateway base URL (e.g. `http://localhost:7080`) |
| `AXEMERE_WORKLOAD_ID` | Yes | Workload identifier for attribution |
| `AXEMERE_PROJECT_ID` | No | Project identifier for attribution |
| `AXEMERE_ACCOUNT_ID` | No | Account identifier for attribution |
| `AXEMERE_CUSTOMER_ID` | No | Customer identifier for attribution |
| `AXEMERE_GATEWAY_TOKEN` | Managed only | Bearer token for managed gateway auth |
| `AXEMERE_AZURE_ENDPOINT` | Example 11, 19 | Azure OpenAI hostname (e.g. `myresource.openai.azure.com`) |
| `AZURE_OPENAI_DEPLOYMENT` | Example 11, 19 | Azure deployment name (e.g. `gpt-4o-mini`) |
| `MVGC_ADMIN_TOKEN` | Example 17 | Gateway admin token for approval polling |

## Gateway credentials needed

Provider credentials live in the gateway, not in the Python process. You do not need `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` set locally.

| Examples | Provider | What to configure in the gateway |
|----------|----------|-----------------------------------|
| 01–09, 14, 17–20 | OpenAI | A credential with `provider: openai`, set as default |
| 01, 08, 14, 20 | Anthropic | A credential with `provider: anthropic`, set as default |
| 10 | Gemini | A credential with `provider: gemini`, set as default |
| 11, 19 | Azure OpenAI | A credential with `provider: azure_openai` + `AXEMERE_AZURE_ENDPOINT` |
| 15 | Cohere | A credential with `provider: cohere`, set as default |

## Troubleshooting

**Gateway not running:**
```
GatewayError: Network error: Connection refused
```
Start the gateway: `docker compose up -d` from the project root.

**Policy denied — missing project_id:**
```
PolicyDeniedError: Request denied by policy
```
Set `AXEMERE_PROJECT_ID` in your `.env`. The default policy requires a non-empty `project_id`.

**Model not found / 404 from provider:** Verify the model name matches what your credentials have access to. Examples use `gpt-4o-mini` and `claude-haiku-4-5-20251001`.

**`faiss-cpu` install fails on Apple Silicon:** `pip install faiss-cpu --no-binary faiss-cpu` or use conda.

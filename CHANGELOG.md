# Changelog

All notable changes to the Axemere Python SDK are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

---

## [0.1.6] - 2026-06-25

### Fixed
- `axemere-gateway`: `stream()` and `stream_sync()` are now properly public methods. Previously the streaming implementation was private (`_stream()`). The implementation now also correctly parses gateway metering events (`mvgc_metering`) and handles provider stream terminators (`message_stop` for Anthropic), so the final chunk carries token usage and cost data.
- `axemere-gateway-google`: added unit tests for `genai_client()` covering gateway URL, auth header, target-host header, placeholder API key, `x-goog-api-key` stripping, and private-API failure warning.

---

## [0.1.5] - 2026-06-25

### Added
- `axemere-gateway`: framework-independent async client (`AiGatewayClient`) for the Axemere AI Gateway Action API (`/v1/actions:execute`) supporting OpenAI-compatible, Anthropic, and Gemini providers.
- `axemere-gateway`: proxy mode URL builder (`AiGatewayConfig.proxy_url()`) that produces a drop-in base URL for existing provider SDKs without changing any other client code.
- `axemere-gateway`: unified `ExecuteResponse` with `content`, `metering` (cost, token counts), and `record_id` regardless of which provider was used.
- `axemere-gateway`: `stream()` / `stream_sync()` methods that return an async/sync iterable of text chunks with a final chunk carrying record ID and metering data.
- `axemere-gateway`: per-request and config-level attribution fields (`workload_id`, `project_id`, `account_id`, `customer_id`, `labels`) forwarded to the gateway for cost allocation and reporting.
- `axemere-gateway`: structured errors (`PolicyDeniedError`, `QuotaExceededError`, `TimeoutError`) with typed fields so callers can handle each case explicitly.
- `axemere-gateway-openai`: drop-in `openai.OpenAI` subclass that routes all requests through the gateway using proxy mode; swap the import and no other code changes are needed.
- `axemere-gateway-anthropic`: drop-in `anthropic.Anthropic` subclass that routes all requests through the gateway using proxy mode.
- `axemere-gateway-google`: `genai_client()` factory that returns a `google.genai.Client` configured to route through the gateway.
- `axemere-gateway-langchain`: `chat_openai()` and `chat_anthropic()` factories returning LangChain `ChatOpenAI`/`ChatAnthropic` subclasses pre-wired to the gateway, compatible with LangChain chains and agents.
- Support for 20+ providers via proxy mode: openai, anthropic, gemini, cohere, mistral, groq, deepseek, together, fireworks, perplexity, openrouter, xai, nvidia-nim, upstage, moonshot, minimax, zhipu, and more.
- All five packages carry `py.typed` markers for full type-checker visibility.

[Unreleased]: https://github.com/Axemere-LLC/axemere-python/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/Axemere-LLC/axemere-python/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Axemere-LLC/axemere-python/releases/tag/v0.1.5

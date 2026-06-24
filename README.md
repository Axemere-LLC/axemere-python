# Axemere AI Gateway — Python SDK

Python client packages for the [Axemere AI Gateway](https://axemere.ai).

## Packages

| Package | Install | Description |
|---------|---------|-------------|
| `axemere-gateway` | `pip install axemere-gateway` | Core client — explicit API, config, errors |
| `axemere-gateway-openai` | `pip install axemere-gateway-openai` | OpenAI + Azure OpenAI drop-in |
| `axemere-gateway-anthropic` | `pip install axemere-gateway-anthropic` | Anthropic drop-in |
| `axemere-gateway-google` | `pip install axemere-gateway-google` | Google Gemini drop-in |
| `axemere-gateway-langchain` | `pip install axemere-gateway-langchain` | LangChain `BaseChatModel` |

## Quick start

```bash
pip install axemere-gateway
```

```python
from axemere.gateway import AiGatewayClient, AiGatewayConfig

config = AiGatewayConfig()  # reads AXEMERE_GATEWAY_URL, AXEMERE_WORKLOAD_TOKEN
client = AiGatewayClient(config)

result = client.execute(
    provider="openai",
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(result.content)
print(result.metering.cost_usd)
```

## Examples

See [`examples/`](examples/) for 20 runnable examples covering all providers,
streaming, delegation, governance, and proxy mode.

```bash
cd examples
pip install -r requirements.txt
python run_all.py
```

## Requirements

- Python 3.10+
- A running Axemere AI Gateway (`AXEMERE_GATEWAY_URL`)
- A workload token (`AXEMERE_WORKLOAD_TOKEN`)

## License

MIT — see [LICENSE](LICENSE).

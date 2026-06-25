# axemere-gateway

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

Framework-independent Python client for the [Axemere AI Gateway](https://axemere.ai).

Use this package when you want explicit control over every request, or when you are not using OpenAI, Anthropic, or another supported SDK. If you are already using one of those SDKs, install the matching drop-in wrapper instead (`axemere-gateway-openai`, `axemere-gateway-anthropic`, etc.) — it requires no code changes beyond the import.

## Install

```bash
pip install axemere-gateway
```

## Usage

```python
from axemere.gateway import AiGatewayClient, AiGatewayConfig

config = AiGatewayConfig()  # reads AXEMERE_GATEWAY_URL + AXEMERE_GATEWAY_TOKEN
client = AiGatewayClient(config)

# execute() is async; execute_sync() is the synchronous entry point.
result = client.execute_sync(
    provider="openai",
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(result.content)
if result.metering:
    print(result.metering.cost_usd)  # float USD, e.g. 0.000042
```

## Configuration

| Env var | Description |
|---------|-------------|
| `AXEMERE_GATEWAY_URL` | Gateway base URL, e.g. `http://localhost:7080` |
| `AXEMERE_GATEWAY_TOKEN` | Bearer token issued by the gateway (legacy `AXEMERE_WORKLOAD_TOKEN` still accepted) |
| `AXEMERE_WORKLOAD_ID` | Workload identifier for attribution |
| `AXEMERE_PROJECT_ID` | Project identifier for spend grouping |

## Links

- [Axemere AI Gateway](https://axemere.ai)
- [Documentation](https://axemere.ai/docs)
- [GitHub](https://github.com/Axemere-LLC/axemere-python)

## License

MIT

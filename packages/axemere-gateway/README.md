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

config = AiGatewayConfig()  # reads AXEMERE_GATEWAY_URL + AXEMERE_WORKLOAD_TOKEN
client = AiGatewayClient(config)

result = client.execute(
    provider="openai",
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(result.content)
print(result.metering.cost_usd)  # "$0.000042"
```

## Configuration

| Env var | Description |
|---------|-------------|
| `AXEMERE_GATEWAY_URL` | Gateway base URL, e.g. `http://localhost:7080` |
| `AXEMERE_WORKLOAD_TOKEN` | Workload token issued by the gateway |
| `AXEMERE_WORKLOAD_ID` | Workload identifier for attribution |
| `AXEMERE_PROJECT_ID` | Project identifier for spend grouping |

## Links

- [Axemere AI Gateway](https://axemere.ai)
- [Documentation](https://axemere.ai/docs)
- [GitHub](https://github.com/Axemere-LLC/axemere-python)

## License

MIT

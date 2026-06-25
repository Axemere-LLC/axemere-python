# axemere-gateway-anthropic

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

Drop-in replacement for the Anthropic Python SDK that routes every request through the [Axemere AI Gateway](https://axemere.ai).

Change one import — all existing Anthropic code works unchanged. The gateway adds cost controls, policy enforcement, and an append-only audit ledger to every call.

## Install

```bash
pip install axemere-gateway-anthropic
```

## Usage

```python
# Before
from anthropic import Anthropic

# After — one line change
from axemere.gateway.anthropic import Anthropic

client = Anthropic()  # reads AXEMERE_GATEWAY_URL + AXEMERE_GATEWAY_TOKEN
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
)
print(message.content[0].text)
```

Streaming and async (`AsyncAnthropic`) are both supported.

## Configuration

| Env var | Description |
|---------|-------------|
| `AXEMERE_GATEWAY_URL` | Gateway base URL, e.g. `http://localhost:7080` |
| `AXEMERE_GATEWAY_TOKEN` | Bearer token issued by the gateway |

## Links

- [Axemere AI Gateway](https://axemere.ai)
- [Documentation](https://axemere.ai/docs)
- [GitHub](https://github.com/Axemere-LLC/axemere-python)

## License

MIT

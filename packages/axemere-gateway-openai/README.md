# axemere-gateway-openai

Drop-in replacement for the OpenAI Python SDK that routes every request through the [Axemere AI Gateway](https://axemere.ai).

Change one import — all existing OpenAI code works unchanged. The gateway adds cost controls, policy enforcement, and an append-only audit ledger to every call.

## Install

```bash
pip install axemere-gateway-openai
```

## Usage

```python
# Before
from openai import OpenAI

# After — one line change
from axemere.gateway.openai import OpenAI

client = OpenAI()  # reads AXEMERE_GATEWAY_URL + AXEMERE_WORKLOAD_TOKEN
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

Streaming, async (`AsyncOpenAI`), and Azure OpenAI (`AzureOpenAI`, `AsyncAzureOpenAI`) are all supported.

## Configuration

| Env var | Description |
|---------|-------------|
| `AXEMERE_GATEWAY_URL` | Gateway base URL, e.g. `http://localhost:7080` |
| `AXEMERE_WORKLOAD_TOKEN` | Workload token issued by the gateway |

## Links

- [Axemere AI Gateway](https://axemere.ai)
- [Documentation](https://axemere.ai/docs)
- [GitHub](https://github.com/Axemere-LLC/axemere-python)

## License

MIT

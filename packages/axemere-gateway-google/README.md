# axemere-gateway-google

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

Google Gemini client factory for the [Axemere AI Gateway](https://axemere.ai). Routes `google-genai` requests through the gateway with cost controls, policy enforcement, and an append-only audit ledger.

## Install

```bash
pip install axemere-gateway-google
```

## Usage

```python
from axemere.gateway.google import genai_client

client = genai_client()  # reads AXEMERE_GATEWAY_URL + AXEMERE_GATEWAY_TOKEN
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello",
)
print(response.text)
```

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

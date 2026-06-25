# axemere-gateway-langchain

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

LangChain `BaseChatModel` integration for the [Axemere AI Gateway](https://axemere.ai). Drop `ChatAiGateway` into any LangChain chain or agent — the gateway adds cost controls, policy enforcement, and an append-only audit ledger to every call.

## Install

```bash
pip install axemere-gateway-langchain
```

## Usage

```python
from axemere.gateway.langchain import ChatAiGateway

llm = ChatAiGateway(provider="openai", model="gpt-4o-mini")
# reads AXEMERE_GATEWAY_URL + AXEMERE_GATEWAY_TOKEN from env

response = llm.invoke("Hello")
print(response.content)
```

Works anywhere a LangChain `BaseChatModel` is accepted — chains, agents, LCEL pipelines.

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

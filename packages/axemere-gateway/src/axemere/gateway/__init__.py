from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from axemere.gateway._config import AiGatewayConfig
from axemere.gateway._headers import ai_gateway_headers, PLACEHOLDER_API_KEY
from axemere.gateway._http import AiGatewayHttpClient, AsyncAiGatewayHttpClient
from axemere.gateway._errors import (
    GatewayError,
    PolicyDeniedError,
    QuotaExceededError,
    GatewayTimeoutError,
)
from axemere.gateway._client import AiGatewayClient
from axemere.gateway._types import (
    ExecuteResponse,
    Metering,
    StreamChunk,
    CostBreakdownItem,
)

__version__ = "0.1.0"
__all__ = [
    "AiGatewayConfig",
    "AiGatewayClient",
    "AiGatewayHttpClient",
    "AsyncAiGatewayHttpClient",
    "ai_gateway_headers",
    "PLACEHOLDER_API_KEY",
    "GatewayError",
    "PolicyDeniedError",
    "QuotaExceededError",
    "GatewayTimeoutError",
    "ExecuteResponse",
    "Metering",
    "StreamChunk",
    "CostBreakdownItem",
]

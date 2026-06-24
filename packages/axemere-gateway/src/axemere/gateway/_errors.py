"""Axemere AI Gateway exception hierarchy."""

from __future__ import annotations

from typing import Any, Dict, Optional


class GatewayError(Exception):
    """Raised for unexpected gateway errors (network, non-2xx HTTP, malformed response, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class PolicyDeniedError(GatewayError):
    """Raised when the gateway denies a request via policy."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "",
        trace: Optional[Dict[str, Any]] = None,
        record_id: str = "",
    ) -> None:
        super().__init__(message, status_code=403)
        self.reason = reason
        self.trace = trace or {}
        self.record_id = record_id


class QuotaExceededError(GatewayError):
    """Raised when a budget or quota limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        upgrade_url: str = "",
        retry_after: int = 0,
    ) -> None:
        super().__init__(message, status_code=429)
        self.upgrade_url = upgrade_url
        self.retry_after = retry_after


class GatewayTimeoutError(GatewayError):
    """Raised when the connector execution times out."""

    def __init__(self) -> None:
        super().__init__("connector execution timed out", status_code=504)

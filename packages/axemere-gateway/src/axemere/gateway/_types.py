"""Response data types for the Axemere AI Gateway client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostBreakdownItem:
    """Per-tier cost breakdown entry."""

    label: str = ""
    tokens: int = 0
    rate_per_million: float = 0.0
    subtotal_usd: float = 0.0


@dataclass
class Metering:
    """Token usage and cost metering from a gateway response."""

    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    pricing_config_version: str = ""
    org_pricing_config_version: str = ""
    markup_multiplier_applied: float = 1.0
    cost_breakdown: List[CostBreakdownItem] = field(default_factory=list)


@dataclass
class ExecuteResponse:
    """Response from a successful gateway execute call."""

    content: str
    record_id: str = ""
    metering: Optional[Metering] = None
    provider: str = ""
    model: str = ""
    record_hash: str = ""
    provider_response: Optional[Dict[str, Any]] = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming gateway response."""

    content: str
    is_final: bool = False
    record_id: str = ""
    metering: Optional[Metering] = None

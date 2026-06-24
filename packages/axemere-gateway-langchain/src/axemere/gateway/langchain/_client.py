"""ChatAiGateway: LangChain BaseChatModel wrapping the Axemere AI Gateway explicit action API."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

import httpx
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from axemere.gateway import AiGatewayConfig
from axemere.gateway._errors import GatewayError, PolicyDeniedError

# ------------------------------------------------------------------
# Provider routing table
#
# Maps provider name → (target_host, target_path, message_format).
# message_format controls how LangChain messages are serialised and
# how the provider response is parsed.
#
# "openai"    — OpenAI Chat Completions format. Also used by Mistral
#               and any other openai_compat provider since the gateway
#               routes them all identically.
# "anthropic" — Anthropic Messages format (system prompt split out,
#               tool_use content blocks, etc.).
# "gemini"    — Google Gemini generateContent format (contents list
#               with parts, generationConfig, systemInstruction).
#
# target_path may contain {model} as a placeholder. Currently only
# Gemini uses a model-in-path pattern; it is substituted at call time.
# ------------------------------------------------------------------
_PROVIDER_ROUTES: Dict[str, Dict[str, str]] = {
    "openai": {
        "target_host": "api.openai.com",
        "target_path": "/v1/chat/completions",
        "format": "openai",
    },
    "anthropic": {
        "target_host": "api.anthropic.com",
        "target_path": "/v1/messages",
        "format": "anthropic",
    },
    "mistral": {
        "target_host": "api.mistral.ai",
        "target_path": "/v1/chat/completions",
        "format": "openai",  # Mistral is OpenAI-compatible
    },
    "google": {
        "target_host": "generativelanguage.googleapis.com",
        "target_path": "/v1beta/models/{model}:generateContent",
        "format": "gemini",
    },
    # OpenAI-compatible providers — share the openai format and connector.
    "xai": {
        "target_host": "api.x.ai",
        "target_path": "/v1/chat/completions",
        "format": "openai",
    },
    "deepseek": {
        "target_host": "api.deepseek.com",
        "target_path": "/v1/chat/completions",
        "format": "openai",
    },
    "groq": {
        "target_host": "api.groq.com",
        "target_path": "/openai/v1/chat/completions",
        "format": "openai",
    },
    "together": {
        "target_host": "api.together.ai",
        "target_path": "/v1/chat/completions",
        "format": "openai",
    },
    "fireworks": {
        "target_host": "api.fireworks.ai",
        "target_path": "/inference/v1/chat/completions",
        "format": "openai",
    },
    "perplexity": {
        "target_host": "api.perplexity.ai",
        "target_path": "/chat/completions",
        "format": "openai",
    },
    "openrouter": {
        "target_host": "openrouter.ai",
        "target_path": "/api/v1/chat/completions",
        "format": "openai",
    },
    "nvidia-nim": {
        "target_host": "integrate.api.nvidia.com",
        "target_path": "/v1/chat/completions",
        "format": "openai",
    },
    "upstage": {
        "target_host": "api.upstage.ai",
        "target_path": "/v1/chat/completions",
        "format": "openai",
    },
}

SupportedProvider = Literal[
    "openai", "anthropic", "mistral", "google",
    "xai", "deepseek", "groq", "together", "fireworks",
    "perplexity", "openrouter", "nvidia-nim", "upstage",
]


class ChatAiGateway(BaseChatModel):
    """LangChain chat model that routes through the Axemere AI Gateway.

    Explicit mode: every call is a POST /v1/actions:execute request.
    The gateway applies policy, enforces budgets, and selects credentials
    before forwarding to the upstream AI provider.

    Both buffered (_generate) and streaming (_stream) are supported.
    Use proxy mode (see proxy.py) as an alternative if you need streaming via
    LangChain's standard provider classes (ChatOpenAI, ChatAnthropic, etc.).

    Supported providers: openai, anthropic, mistral, google, xai, deepseek,
    groq, together, fireworks, perplexity, openrouter, nvidia-nim, upstage.

    Example::

        from axemere.gateway.langchain import ChatAiGateway
        from axemere.gateway import AiGatewayConfig

        llm = ChatAiGateway(provider="mistral", model="mistral-large-latest")
        response = llm.invoke("Summarise the key themes in these findings...")
        print(response.content)
    """

    provider: str
    model: str
    config: Optional[Any] = Field(default=None, exclude=True)
    max_tokens: int = 256
    temperature: Optional[float] = None
    # Per-call workload override — set this to attribute a single agent's calls
    # to its own workload_id without creating a new AiGatewayConfig object.
    workload_id: Optional[str] = None
    # Per-call attribution labels — merged with any labels in AiGatewayConfig.
    labels: Optional[Dict[str, str]] = None
    # Populated by bind_tools(); not part of the constructor signature.
    _bound_tools: Optional[List[Dict[str, Any]]] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.provider not in _PROVIDER_ROUTES:
            raise ValueError(
                f"Unsupported provider {self.provider!r}. "
                f"Supported: {sorted(_PROVIDER_ROUTES)}"
            )

    @property
    def _llm_type(self) -> str:
        return f"axemere-{self.provider}"

    def _get_config(self) -> AiGatewayConfig:
        if self.config is not None:
            return self.config
        return AiGatewayConfig.from_env()

    def _effective_workload_id(self, cfg: AiGatewayConfig) -> Optional[str]:
        return self.workload_id or cfg.workload_id or None

    def _effective_labels(self, cfg: AiGatewayConfig) -> Dict[str, str]:
        base: Dict[str, str] = {"source": "langchain"}
        if cfg.labels:
            base.update(cfg.labels)
        if self.labels:
            base.update(self.labels)
        return base

    # ------------------------------------------------------------------
    # Message conversion helpers
    # ------------------------------------------------------------------

    def _messages_to_openai(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry: Dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.additional_kwargs.get("tool_calls"):
                    entry["tool_calls"] = msg.additional_kwargs["tool_calls"]
                    entry["content"] = None
                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    entry["content"] = None
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _messages_to_anthropic(
        self, messages: List[BaseMessage]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Split messages into (system_prompt, messages_list) for Anthropic."""
        system_prompt: Optional[str] = None
        result: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    content_blocks = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("name", ""),
                                "input": tc.get("args", {}),
                            }
                        )
                    result.append({"role": "assistant", "content": content_blocks})
                else:
                    result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": str(msg.content),
                            }
                        ],
                    }
                )
            else:
                result.append({"role": "user", "content": str(msg.content)})

        return system_prompt, result

    def _messages_to_gemini(
        self, messages: List[BaseMessage]
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Convert messages to Gemini generateContent format.

        Returns (system_instruction, contents, generation_config).
        system_instruction is None if no SystemMessage was found.
        """
        system_instruction: Optional[Dict[str, Any]] = None
        contents: List[Dict[str, Any]] = []
        generation_config: Dict[str, Any] = {"maxOutputTokens": self.max_tokens}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Gemini uses a top-level systemInstruction field rather than
                # a message with role "system".
                system_instruction = {"parts": [{"text": str(msg.content)}]}
            elif isinstance(msg, HumanMessage):
                contents.append({"role": "user", "parts": [{"text": str(msg.content)}]})
            elif isinstance(msg, AIMessage):
                contents.append({"role": "model", "parts": [{"text": str(msg.content)}]})
            else:
                contents.append({"role": "user", "parts": [{"text": str(msg.content)}]})

        return system_instruction, contents, generation_config

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_action_request(
        self,
        messages: List[BaseMessage],
        cfg: AiGatewayConfig,
        bound_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        route = _PROVIDER_ROUTES[self.provider]
        fmt = route["format"]
        target_host = route["target_host"]
        target_path = route["target_path"].replace("{model}", self.model)

        if fmt == "openai":
            params: Dict[str, Any] = {
                "model": self.model,
                "messages": self._messages_to_openai(messages),
                "max_tokens": self.max_tokens,
            }
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if bound_tools:
                params["tools"] = bound_tools

        elif fmt == "anthropic":
            system_prompt, anthropic_messages = self._messages_to_anthropic(messages)
            params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
            }
            if system_prompt:
                params["system"] = system_prompt
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if bound_tools:
                params["tools"] = bound_tools

        elif fmt == "gemini":
            system_instruction, contents, generation_config = self._messages_to_gemini(messages)
            params = {
                "contents": contents,
                "generationConfig": generation_config,
            }
            if system_instruction:
                params["systemInstruction"] = system_instruction

        else:
            raise GatewayError(f"Unknown message format {fmt!r} for provider {self.provider!r}")

        return {
            "schema": "mvgc.action_request.v2",
            "request_id": str(uuid.uuid4()),
            "org_id": "",
            "caller_id": "langchain",
            "workload_id": self._effective_workload_id(cfg),
            "ingress_mode": "explicit_action_request",
            "action": {
                "type": "llm_chat",
                "method": "POST",
                "target_host": target_host,
                "target_path": target_path,
                "params": params,
            },
            "attribution": {
                "project_id": cfg.project_id,
                "customer_id": cfg.customer_id,
                "account_id": cfg.account_id,
                "labels": self._effective_labels(cfg),
            },
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_openai_response(self, body: Dict[str, Any]) -> AIMessage:
        choices = body.get("choices", [])
        if not choices:
            return AIMessage(content="")

        message = choices[0].get("message", {})
        content = message.get("content") or ""
        raw_tool_calls = message.get("tool_calls")

        if not raw_tool_calls:
            return AIMessage(content=content)

        tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"_raw": args_str}
            tool_calls.append(
                {
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "args": args,
                    "type": "tool_call",
                }
            )
        return AIMessage(content=content, tool_calls=tool_calls)

    def _parse_anthropic_response(self, body: Dict[str, Any]) -> AIMessage:
        content_blocks = body.get("content", [])
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                        "type": "tool_call",
                    }
                )

        text = "".join(text_parts)
        if tool_calls:
            return AIMessage(content=text, tool_calls=tool_calls)
        return AIMessage(content=text)

    def _parse_gemini_response(self, body: Dict[str, Any]) -> AIMessage:
        candidates = body.get("candidates", [])
        if not candidates:
            return AIMessage(content="")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        return AIMessage(content=text)

    def _parse_response(self, provider_format: str, body: Dict[str, Any]) -> AIMessage:
        if provider_format == "openai":
            return self._parse_openai_response(body)
        if provider_format == "anthropic":
            return self._parse_anthropic_response(body)
        if provider_format == "gemini":
            return self._parse_gemini_response(body)
        raise GatewayError(f"Unknown response format: {provider_format!r}")

    # ------------------------------------------------------------------
    # Core LangChain interface
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        cfg = self._get_config()

        bound_tools: Optional[List[Dict[str, Any]]] = kwargs.get("tools") or self._bound_tools
        action_request = self._build_action_request(messages, cfg, bound_tools)

        request_headers: Dict[str, str] = {}
        if cfg.gateway_token:
            request_headers["Authorization"] = f"Bearer {cfg.gateway_token}"

        try:
            with httpx.Client(timeout=float(cfg.timeout)) as client:
                response = client.post(
                    f"{cfg.gateway_url}/v1/actions:execute",
                    json=action_request,
                    headers=request_headers,
                )
        except httpx.RequestError as exc:
            raise GatewayError(f"Gateway connection failed: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:
            raise GatewayError(
                f"Invalid JSON from gateway (status={response.status_code}): {exc}"
            ) from exc

        decision = data.get("decision", "deny")

        if decision == "deny" or response.status_code == 403:
            trace = data.get("decision_trace", {})
            reason = (
                trace.get("reason")
                or data.get("reason")
                or data.get("message")   # require_attribution path returns "message"
                or data.get("error")
                or f"Policy denied the request (HTTP {response.status_code}): {data}"
            )
            raise PolicyDeniedError(reason, reason=reason, trace=trace)

        if response.status_code == 202:
            approval_id = data.get("approval_id", "")
            raise GatewayError(
                f"Request requires manual approval (approval_id={approval_id!r})"
            )

        if response.status_code == 429:
            raise GatewayError("Request rate limited by gateway")

        if response.status_code >= 400:
            raise GatewayError(
                f"Gateway returned HTTP {response.status_code}: {data}"
            )

        result_dto = data.get("result")
        if result_dto is None:
            raise GatewayError("No 'result' field in gateway response")

        body = result_dto.get("body")
        if body is None:
            raise GatewayError("No 'body' field in connector result")

        upstream_status = result_dto.get("status_code", 200)
        if upstream_status >= 400:
            # Extract the human-readable message from the provider's error shape.
            # Gemini:    {"error": {"code": 503, "message": "...", "status": "..."}}
            # OpenAI:    {"error": {"message": "...", "type": "...", "code": "..."}}
            # Anthropic: {"type": "error", "error": {"type": "...", "message": "..."}}
            error_msg = ""
            if isinstance(body, dict):
                err = body.get("error") or {}
                error_msg = (
                    err.get("message")
                    or body.get("message")
                    or str(body)
                )
            raise GatewayError(
                f"Upstream {self.provider} returned HTTP {upstream_status}: {error_msg}"
            )

        fmt = _PROVIDER_ROUTES[self.provider]["format"]
        ai_message = self._parse_response(fmt, body)

        generation_info = {
            "record_id": data.get("record_id"),
            "metering": data.get("metering", {}),
            "provider": self.provider,
            "model": self.model,
        }

        return ChatResult(
            generations=[ChatGeneration(message=ai_message, generation_info=generation_info)]
        )

    def _extract_stream_text(self, fmt: str, chunk: Dict[str, Any]) -> str:
        """Extract the incremental text from one SSE data chunk."""
        if fmt == "openai":
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("delta", {}).get("content") or ""
        if fmt == "anthropic":
            # Only content_block_delta with type text_delta carries text.
            if chunk.get("type") == "content_block_delta":
                return chunk.get("delta", {}).get("text") or ""
            return ""
        if fmt == "gemini":
            candidates = chunk.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if "text" in p)
        return ""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream tokens from the gateway.

        The gateway forwards the raw SSE stream from the upstream provider, so
        this method parses the provider-specific SSE format (OpenAI, Anthropic,
        or Gemini) and yields LangChain ChatGenerationChunk objects.

        Unlike buffered _generate(), streaming bypasses the gateway's
        ConnectorTimeout, making it the right choice for large prompts or
        long-running completions.
        """
        cfg = self._get_config()
        fmt = _PROVIDER_ROUTES[self.provider]["format"]

        bound_tools: Optional[List[Dict[str, Any]]] = kwargs.get("tools") or self._bound_tools
        action_request = self._build_action_request(messages, cfg, bound_tools)

        # Setting stream=true in action params tells the gateway to forward the
        # upstream SSE response directly instead of buffering it.
        action_request["action"]["params"]["stream"] = True

        request_headers: Dict[str, str] = {}
        if cfg.gateway_token:
            request_headers["Authorization"] = f"Bearer {cfg.gateway_token}"

        record_id: Optional[str] = None
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    f"{cfg.gateway_url}/v1/actions:execute",
                    json=action_request,
                    headers=request_headers,
                ) as response:
                    if response.status_code != 200:
                        body = response.read()
                        try:
                            data = json.loads(body)
                        except Exception:
                            raise GatewayError(
                                f"Gateway returned HTTP {response.status_code}"
                            )
                        if response.status_code == 403:
                            trace = data.get("decision_trace", {})
                            reason = (
                                trace.get("reason")
                                or data.get("reason")
                                or data.get("message")
                                or data.get("error")
                                or f"Policy denied the request (HTTP {response.status_code}): {data}"
                            )
                            raise PolicyDeniedError(reason, reason=reason, trace=trace)
                        raise GatewayError(
                            f"Gateway returned HTTP {response.status_code}: {data}"
                        )

                    record_id = response.headers.get("X-MVGC-Record-ID")
                    metering: dict = {}

                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        # Gateway injects an mvgc_metering event before the
                        # terminal signal — capture it but don't yield to caller.
                        if chunk_data.get("type") == "mvgc_metering":
                            metering = chunk_data.get("metering", {})
                            if not record_id:
                                record_id = chunk_data.get("record_id")
                            continue

                        # Anthropic emits message_stop instead of [DONE].
                        if chunk_data.get("type") == "message_stop":
                            break

                        text = self._extract_stream_text(fmt, chunk_data)
                        if text:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=text)
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(text, chunk=chunk)
                            yield chunk

        except httpx.RequestError as exc:
            raise GatewayError(f"Gateway connection failed: {exc}") from exc

        yield ChatGenerationChunk(
            message=AIMessageChunk(content=""),
            generation_info={
                "record_id": record_id,
                "metering": metering,
                "provider": self.provider,
                "model": self.model,
            },
        )

    # ------------------------------------------------------------------
    # Tool binding
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> "ChatAiGateway":
        """Return a copy of this model with the given tools bound.

        Tools are converted to the provider-specific wire format and included
        in every subsequent _generate() call.
        """
        fmt = _PROVIDER_ROUTES[self.provider]["format"]
        formatted: List[Dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted.append(tool)
            elif hasattr(tool, "name") and hasattr(tool, "description"):
                if fmt == "anthropic":
                    formatted.append(self._tool_to_anthropic_format(tool))
                else:
                    formatted.append(self._tool_to_openai_format(tool))
            else:
                formatted.append(tool)

        return super().bind(tools=formatted, **kwargs)  # type: ignore[return-value]

    def _tool_to_openai_format(self, tool: Any) -> Dict[str, Any]:
        schema = _get_tool_schema(tool)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "parameters": schema,
            },
        }

    def _tool_to_anthropic_format(self, tool: Any) -> Dict[str, Any]:
        schema = _get_tool_schema(tool)
        return {
            "name": tool.name,
            "description": getattr(tool, "description", ""),
            "input_schema": schema,
        }


def _get_tool_schema(tool: Any) -> Dict[str, Any]:
    """Extract JSON schema from a LangChain tool or Pydantic model."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return {"type": "object", "properties": {}}
    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()
    if hasattr(args_schema, "schema"):
        return args_schema.schema()
    return {"type": "object", "properties": {}}

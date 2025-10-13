from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import sys
import json
import hashlib
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    Iterator,
    AsyncIterator,
    Tuple,
    TypedDict,
)
from itertools import count

from litellm import completion, acompletion, embedding
import litellm

from python.helpers import dotenv
from python.helpers.dotenv import load_dotenv
from python.helpers.providers import get_provider_config
from python.helpers.rate_limiter import RateLimiter
from python.helpers.tokens import approximate_tokens
from python.helpers.defer import EventLoopThread
from python.helpers.mlx_server import MLXServerManager
from python.models.apple_mlx_provider import MLXServerClient
from pydantic import ConfigDict

logger = logging.getLogger("a0.mlx_cache")
reasoning_logger = logging.getLogger("a0.reasoning")

_REASONING_FIELD_NAMES = (
    "reasoning_content",
    "reasoning",
    "x_gpt_thinking",
    "thinking",
    "thoughts",
    "internal_thoughts",
)

_DEBUG_REASONING = os.getenv("A0_DEBUG_REASONING", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_REASONING_CHUNK_COUNTER = count(1)

if _DEBUG_REASONING and reasoning_logger.level == logging.NOTSET:
    reasoning_logger.setLevel(logging.INFO)


def _normalize_obj(value: Any) -> Any:
    """Best-effort convert provider specific objects into plain Python types."""

    if isinstance(value, dict):
        return {k: _normalize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_obj(v) for v in value]

    # LiteLLM / OpenAI objects often provide to_dict/model_dump helpers.
    for attr in ("to_dict", "model_dump", "dict"):
        helper = getattr(value, attr, None)
        if callable(helper):
            try:
                return _normalize_obj(helper())
            except Exception:
                continue
    return value


def _stringify_content(value: Any) -> str:
    """Convert mixed content structures (lists, dicts) into plain text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "".join(_stringify_content(v) for v in value)
    if isinstance(value, dict):
        # Prefer explicit text/content style keys before falling back to values
        for key in ("text", "content", "value", "message"):
            if key in value:
                text = _stringify_content(value[key])
                if text:
                    return text
        if "parts" in value:
            return _stringify_content(value["parts"])
        if "data" in value:
            return _stringify_content(value["data"])
        # Fallback: concatenate any scalar string-like values
        return "".join(
            _stringify_content(v)
            for v in value.values()
        )
    return str(value)


def _collect_reasoning_fields(
    container: Any,
    detected_keys: set[str],
    parts: list[str],
) -> None:
    if container is None:
        return
    if isinstance(container, dict):
        for key, value in container.items():
            if key in _REASONING_FIELD_NAMES:
                text = _stringify_content(value)
                if text:
                    parts.append(text)
                    detected_keys.add(key)
            elif isinstance(value, (dict, list, tuple)):
                _collect_reasoning_fields(value, detected_keys, parts)
    elif isinstance(container, (list, tuple)):
        for item in container:
            _collect_reasoning_fields(item, detected_keys, parts)


def _extract_reasoning_text(*containers: Any) -> tuple[str, set[str]]:
    detected: set[str] = set()
    fragments: list[str] = []
    for container in containers:
        _collect_reasoning_fields(container, detected, fragments)
    return "".join(fragments), detected


def _debug_reasoning_keys(chunk_index: int, keys: set[str]) -> None:
    if not _DEBUG_REASONING:
        return

    if keys:
        reasoning_logger.info(
            "[reasoning] chunk %s keys=%s",
            chunk_index,
            sorted(keys),
        )
    else:
        reasoning_logger.info("[reasoning] chunk %s keys=(none)", chunk_index)


def _extract_response_text(*sources: Any) -> str:
    for source in sources:
        if source is None:
            continue

        candidate = None
        if isinstance(source, dict):
            candidate = source.get("content")
            if not candidate:
                for alt in ("text", "message"):
                    if alt in source:
                        candidate = source[alt]
                        break
        else:
            candidate = getattr(source, "content", None)
            if not candidate:
                candidate = getattr(source, "text", None)

        if candidate:
            text = _stringify_content(_normalize_obj(candidate))
            if text:
                return text

    return ""

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.messages import (
    BaseMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class MLXCacheManager:
    """
    Persistent cache manager for MLX provider that survives module reloads.
    Uses file-based storage to persist provider state across Flask/Werkzeug reloads.
    """

    def __init__(self):
        # Create cache directory in tmp folder
        self.cache_dir = Path(__file__).parent / "tmp" / "mlx_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "mlx_provider_cache.json"

    def _get_cache_key(self, model_path: str) -> str:
        """Generate a cache key based on model path"""
        return hashlib.md5(model_path.encode()).hexdigest()

    def save_provider_state(self, model_path: str, provider_state: dict):
        """Save provider state to persistent cache"""
        try:
            cache_key = self._get_cache_key(model_path)
            cache_data = {
                "model_path": model_path,
                "cache_key": cache_key,
                "provider_state": provider_state,
                "timestamp": os.path.getmtime(__file__) if os.path.exists(__file__) else 0,
            }

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Saved MLX provider state for model: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save MLX cache: {e}")

    def load_provider_state(self, model_path: str) -> Optional[dict]:
        """Load provider state from persistent cache"""
        try:
            if not self.cache_file.exists():
                return None

            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            # Validate cache key matches current model path
            expected_key = self._get_cache_key(model_path)
            if cache_data.get("cache_key") != expected_key:
                logger.debug(f"MLX cache key mismatch for {model_path}; ignoring cache")
                return None

            # Check if models.py has been modified since cache was created
            current_mtime = os.path.getmtime(__file__) if os.path.exists(__file__) else 0
            cache_mtime = cache_data.get("timestamp", 0)
            if current_mtime > cache_mtime:
                logger.debug("models.py modified since cache creation; ignoring MLX cache")
                return None

            logger.debug(f"Loaded MLX provider state for model: {model_path}")
            return cache_data.get("provider_state")

        except Exception as e:
            logger.warning(f"Failed to load MLX cache: {e}")
            return None

    def clear_cache(self):
        """Clear all cached provider state"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.debug("Cleared MLX cache")
        except Exception as e:
            logger.warning(f"Failed to clear MLX cache: {e}")

    def is_cache_valid(self, model_path: str) -> bool:
        """Check if cache exists and is valid for the given model path"""
        state = self.load_provider_state(model_path)
        return state is not None


# Global cache manager instance
_mlx_cache_manager = MLXCacheManager()


# disable extra logging, must be done repeatedly, otherwise browser-use will turn it back on for some reason
def turn_off_logging():
    os.environ["LITELLM_LOG"] = "ERROR"  # only errors
    litellm.suppress_debug_info = True
    # Silence **all** LiteLLM sub-loggers (utils, cost_calculator…)
    for name in logging.Logger.manager.loggerDict:
        if name.lower().startswith("litellm"):
            logging.getLogger(name).setLevel(logging.ERROR)


# init
load_dotenv()
turn_off_logging()


class ModelType(Enum):
    CHAT = "Chat"
    EMBEDDING = "Embedding"


@dataclass
class ModelConfig:
    type: ModelType
    provider: str
    name: str
    api_base: str = ""
    ctx_length: int = 0
    limit_requests: int = 0
    limit_input: int = 0
    limit_output: int = 0
    vision: bool = False
    kwargs: dict = field(default_factory=dict)

    def build_kwargs(self):
        kwargs = self.kwargs.copy() or {}
        if self.api_base and "api_base" not in kwargs:
            kwargs["api_base"] = self.api_base
        return kwargs


class ChatChunk(TypedDict):
    """Simplified response chunk for chat models."""

    response_delta: str
    reasoning_delta: str


class ChatGenerationResult:
    """Aggregates reasoning/response content from streamed chat chunks."""

    def __init__(self, chunk: ChatChunk | None = None):
        self.reasoning = ""
        self.response = ""
        self.thinking = False
        self.thinking_tag = ""
        self.unprocessed = ""
        self.native_reasoning = False
        self.thinking_pairs = [("<think>", "</think>"), ("<reasoning>", "</reasoning>")]
        if chunk:
            self.add_chunk(chunk)

    def add_chunk(self, chunk: ChatChunk) -> ChatChunk:
        incoming_reasoning = chunk.get("reasoning_delta", "")
        if incoming_reasoning:
            self.native_reasoning = True

        if self.native_reasoning:
            reasoning_delta = self._diff_native_reasoning(incoming_reasoning)
            processed = ChatChunk(
                response_delta=chunk["response_delta"],
                reasoning_delta=reasoning_delta,
            )
        else:
            processed = self._process_thinking_chunk(chunk)

        self.reasoning += processed["reasoning_delta"]
        self.response += processed["response_delta"]
        return processed

    def output(self) -> ChatChunk:
        response = self.response
        reasoning = self.reasoning
        if self.unprocessed:
            if reasoning and not response:
                reasoning += self.unprocessed
            else:
                response += self.unprocessed
        return ChatChunk(response_delta=response, reasoning_delta=reasoning)

    def _diff_native_reasoning(self, incoming: str) -> str:
        if not incoming:
            return ""

        if not self.reasoning:
            return incoming

        if incoming.startswith(self.reasoning):
            return incoming[len(self.reasoning) :]

        # Fall back to treating the payload as a delta
        return incoming

    def _process_thinking_chunk(self, chunk: ChatChunk) -> ChatChunk:
        response_delta = self.unprocessed + chunk["response_delta"]
        self.unprocessed = ""
        return self._process_thinking_tags(response_delta, chunk["reasoning_delta"])

    def _process_thinking_tags(self, response: str, reasoning: str) -> ChatChunk:
        if self.thinking:
            close_pos = response.find(self.thinking_tag)
            if close_pos != -1:
                reasoning += response[:close_pos]
                response = response[close_pos + len(self.thinking_tag) :]
                self.thinking = False
                self.thinking_tag = ""
            else:
                if self._is_partial_closing_tag(response):
                    self.unprocessed = response
                    response = ""
                else:
                    reasoning += response
                    response = ""
        else:
            for opening_tag, closing_tag in self.thinking_pairs:
                if response.startswith(opening_tag):
                    response = response[len(opening_tag) :]
                    self.thinking = True
                    self.thinking_tag = closing_tag

                    close_pos = response.find(closing_tag)
                    if close_pos != -1:
                        reasoning += response[:close_pos]
                        response = response[close_pos + len(closing_tag) :]
                        self.thinking = False
                        self.thinking_tag = ""
                    else:
                        if self._is_partial_closing_tag(response):
                            self.unprocessed = response
                            response = ""
                        else:
                            reasoning += response
                            response = ""
                    break
                elif len(response) < len(opening_tag) and self._is_partial_opening_tag(
                    response, opening_tag
                ):
                    self.unprocessed = response
                    response = ""
                    break

        return ChatChunk(response_delta=response, reasoning_delta=reasoning)

    def _is_partial_opening_tag(self, text: str, opening_tag: str) -> bool:
        for idx in range(1, len(opening_tag)):
            if text == opening_tag[:idx]:
                return True
        return False

    def _is_partial_closing_tag(self, text: str) -> bool:
        if not self.thinking_tag or not text:
            return False
        max_check = min(len(text), len(self.thinking_tag) - 1)
        for idx in range(1, max_check + 1):
            if text.endswith(self.thinking_tag[:idx]):
                return True
        return False


rate_limiters: dict[str, RateLimiter] = {}
api_keys_round_robin: dict[str, int] = {}


# Global state for MLX server switching
_last_provider = None

def get_api_key(service: str) -> str:
    # get api key for the service
    key = (
        dotenv.get_dotenv_value(f"API_KEY_{service.upper()}")
        or dotenv.get_dotenv_value(f"{service.upper()}_API_KEY")
        or dotenv.get_dotenv_value(f"{service.upper()}_API_TOKEN")
        or "None"
    )
    # if the key contains a comma, use round-robin
    if "," in key:
        api_keys = [k.strip() for k in key.split(",") if k.strip()]
        api_keys_round_robin[service] = api_keys_round_robin.get(service, -1) + 1
        key = api_keys[api_keys_round_robin[service] % len(api_keys)]
    return key


def get_rate_limiter(
    provider: str, name: str, requests: int, input: int, output: int
) -> RateLimiter:
    key = f"{provider}\\{name}"
    rate_limiters[key] = limiter = rate_limiters.get(key, RateLimiter(seconds=60))
    limiter.limits["requests"] = requests or 0
    limiter.limits["input"] = input or 0
    limiter.limits["output"] = output or 0
    return limiter

async def apply_rate_limiter(model_config: ModelConfig|None, input_text: str, rate_limiter_callback: Callable[[str, str, int, int], Awaitable[bool]] | None = None):
    if not model_config:
        return
    limiter = get_rate_limiter(
        model_config.provider,
        model_config.name,
        model_config.limit_requests,
        model_config.limit_input,
        model_config.limit_output,
    )
    limiter.add(input=approximate_tokens(input_text))
    limiter.add(requests=1)
    await limiter.wait(rate_limiter_callback)
    return limiter

def apply_rate_limiter_sync(model_config: ModelConfig|None, input_text: str, rate_limiter_callback: Callable[[str, str, int, int], Awaitable[bool]] | None = None):
    """Run the async rate limiter on a shared background loop without creating new loops.

    Avoids asyncio.run per call (which creates event loops and FDs) by using
    the process-wide EventLoopThread.
    """
    if not model_config:
        return
    elt = EventLoopThread("Background")
    future = elt.run_coroutine(apply_rate_limiter(model_config, input_text, rate_limiter_callback))
    return future.result()


class LiteLLMChatWrapper(SimpleChatModel):
    model_name: str
    provider: str
    kwargs: dict = {}

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        validate_assignment=False,
    )

    def __init__(self, model: str, provider: str, model_config: Optional[ModelConfig] = None, **kwargs: Any):
        model_value = f"{provider}/{model}"
        super().__init__(model_name=model_value, provider=provider, kwargs=kwargs)  # type: ignore
        # Set A0 model config as instance attribute after parent init
        self.a0_model_conf = model_config

    @property
    def _llm_type(self) -> str:
        return "litellm-chat"
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        result = []
        # Map LangChain message types to LiteLLM roles
        role_mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        for m in messages:
            role = role_mapping.get(m.type, m.type)
            message_dict = {"role": role, "content": m.content}

            # Handle tool calls for AI messages
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                # Convert LangChain tool calls to LiteLLM format
                new_tool_calls = []
                for tool_call in tool_calls:
                    # Ensure arguments is a JSON string
                    args = tool_call["args"]
                    if isinstance(args, dict):
                        import json

                        args_str = json.dumps(args)
                    else:
                        args_str = str(args)

                    new_tool_calls.append(
                        {
                            "id": tool_call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": args_str,
                            },
                        }
                    )
                message_dict["tool_calls"] = new_tool_calls

            # Handle tool call ID for ToolMessage
            tool_call_id = getattr(m, "tool_call_id", None)
            if tool_call_id:
                message_dict["tool_call_id"] = tool_call_id

            result.append(message_dict)
        return result

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        import asyncio
        
        msgs = self._convert_messages(messages)
        
        # Apply rate limiting if configured
        apply_rate_limiter_sync(self.a0_model_conf, str(msgs))
        
        # Call the model
        resp = completion(
            model=self.model_name, messages=msgs, stop=stop, **{**self.kwargs, **kwargs}
        )

        # Parse output
        parsed = _parse_chunk(resp)
        output = ChatGenerationResult(parsed).output()
        return output["response_delta"]

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import asyncio
        
        msgs = self._convert_messages(messages)
        
        # Apply rate limiting if configured
        apply_rate_limiter_sync(self.a0_model_conf, str(msgs))
        
        result = ChatGenerationResult()

        for chunk in completion(
            model=self.model_name,
            messages=msgs,
            stream=True,
            stop=stop,
            **{**self.kwargs, **kwargs},
        ):
            parsed = _parse_chunk(chunk)
            output = result.add_chunk(parsed)
            # Only yield chunks with non-None content
            if output["response_delta"]:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=output["response_delta"])
                )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        msgs = self._convert_messages(messages)
        
        # Apply rate limiting if configured
        await apply_rate_limiter(self.a0_model_conf, str(msgs))
        
        result = ChatGenerationResult()

        response = await acompletion(
            model=self.model_name,
            messages=msgs,
            stream=True,
            stop=stop,
            **{**self.kwargs, **kwargs},
        )
        async for chunk in response:  # type: ignore
            parsed = _parse_chunk(chunk)
            # Only yield chunks with non-None content
            output = result.add_chunk(parsed)
            if output["response_delta"]:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=output["response_delta"])
                )

    async def unified_call(
        self,
        system_message="",
        user_message="",
        messages: List[BaseMessage] | None = None,
        response_callback: Callable[[str, str], Awaitable[None]] | None = None,
        reasoning_callback: Callable[[str, str], Awaitable[None]] | None = None,
        tokens_callback: Callable[[str, int], Awaitable[None]] | None = None,
        rate_limiter_callback: Callable[[str, str, int, int], Awaitable[bool]] | None = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:

        turn_off_logging()

        if not messages:
            messages = []
        # construct messages
        if system_message:
            messages.insert(0, SystemMessage(content=system_message))
        if user_message:
            messages.append(HumanMessage(content=user_message))

        # convert to litellm format
        msgs_conv = self._convert_messages(messages)

        # Apply rate limiting if configured
        limiter = await apply_rate_limiter(self.a0_model_conf, str(msgs_conv), rate_limiter_callback)

        # call model
        _completion = await acompletion(
            model=self.model_name,
            messages=msgs_conv,
            stream=True,
            **{**self.kwargs, **kwargs},
        )

        result = ChatGenerationResult()

        # iterate over chunks
        async for chunk in _completion:  # type: ignore
            parsed = _parse_chunk(chunk)
            output = result.add_chunk(parsed)

            if output["reasoning_delta"]:
                if reasoning_callback:
                    await reasoning_callback(output["reasoning_delta"], result.reasoning)
                if tokens_callback:
                    await tokens_callback(
                        output["reasoning_delta"],
                        approximate_tokens(output["reasoning_delta"]),
                    )
                if limiter:
                    limiter.add(output=approximate_tokens(output["reasoning_delta"]))

            if output["response_delta"]:
                if response_callback:
                    await response_callback(output["response_delta"], result.response)
                if tokens_callback:
                    await tokens_callback(
                        output["response_delta"],
                        approximate_tokens(output["response_delta"]),
                    )
                if limiter:
                    limiter.add(output=approximate_tokens(output["response_delta"]))

        return result.response, result.reasoning


class BrowserCompatibleChatWrapper(LiteLLMChatWrapper):
    """
    A wrapper for browser agent that can filter/sanitize messages
    before sending them to the LLM.
    """

    def __init__(self, *args, **kwargs):
        turn_off_logging()
        super().__init__(*args, **kwargs)
        # Browser-use may expect a 'model' attribute
        self.model = self.model_name

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        turn_off_logging()
        result = super()._call(messages, stop, run_manager, **kwargs)
        return result

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        turn_off_logging()
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


class LiteLLMEmbeddingWrapper(Embeddings):
    model_name: str
    kwargs: dict = {}
    a0_model_conf: Optional[ModelConfig] = None

    def __init__(self, model: str, provider: str, model_config: Optional[ModelConfig] = None, **kwargs: Any):
        self.model_name = f"{provider}/{model}" if provider != "openai" else model
        self.kwargs = kwargs
        self.a0_model_conf = model_config
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Apply rate limiting if configured
        apply_rate_limiter_sync(self.a0_model_conf, " ".join(texts))
        
        resp = embedding(model=self.model_name, input=texts, **self.kwargs)
        return [
            item.get("embedding") if isinstance(item, dict) else item.embedding  # type: ignore
            for item in resp.data  # type: ignore
        ]

    def embed_query(self, text: str) -> List[float]:
        # Apply rate limiting if configured
        apply_rate_limiter_sync(self.a0_model_conf, text)
        
        resp = embedding(model=self.model_name, input=[text], **self.kwargs)
        item = resp.data[0]  # type: ignore
        return item.get("embedding") if isinstance(item, dict) else item.embedding  # type: ignore


class LocalSentenceTransformerWrapper(Embeddings):
    """Local wrapper for sentence-transformers models to avoid HuggingFace API calls"""

    def __init__(self, provider: str, model: str, model_config: Optional[ModelConfig] = None, **kwargs: Any):
        # Clean common user-input mistakes
        model = model.strip().strip('"').strip("'")

        # Remove the "sentence-transformers/" prefix if present
        if model.startswith("sentence-transformers/"):
            model = model[len("sentence-transformers/") :]

        self.model = SentenceTransformer(model, **kwargs)
        self.model_name = model
        self.a0_model_conf = model_config
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Apply rate limiting if configured
        apply_rate_limiter_sync(self.a0_model_conf, " ".join(texts))
        
        embeddings = self.model.encode(texts, convert_to_tensor=False)  # type: ignore
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings  # type: ignore

    def embed_query(self, text: str) -> List[float]:
        # Apply rate limiting if configured
        apply_rate_limiter_sync(self.a0_model_conf, text)
        
        embedding = self.model.encode([text], convert_to_tensor=False)  # type: ignore
        result = (
            embedding[0].tolist() if hasattr(embedding[0], "tolist") else embedding[0]
        )
        return result  # type: ignore


def _get_litellm_chat(
    cls: type = LiteLLMChatWrapper,
    model_name: str = "",
    provider_name: str = "",
    model_config: Optional[ModelConfig] = None,
    **kwargs: Any,
):
    # use api key from kwargs or env
    api_key = kwargs.pop("api_key", None) or get_api_key(provider_name)

    # Only pass API key if key is not a placeholder
    if api_key and api_key not in ("None", "NA"):
        kwargs["api_key"] = api_key

    provider_name, model_name, kwargs = _adjust_call_args(
        provider_name, model_name, kwargs
    )
    return cls(provider=provider_name, model=model_name, model_config=model_config, **kwargs)


def _get_litellm_embedding(model_name: str, provider_name: str, model_config: Optional[ModelConfig] = None, **kwargs: Any):
    # Check if this is a local sentence-transformers model
    if provider_name == "huggingface" and model_name.startswith(
        "sentence-transformers/"
    ):
        # Use local sentence-transformers instead of LiteLLM for local models
        provider_name, model_name, kwargs = _adjust_call_args(
            provider_name, model_name, kwargs
        )
        return LocalSentenceTransformerWrapper(
            provider=provider_name, model=model_name, model_config=model_config, **kwargs
        )

    # use api key from kwargs or env
    api_key = kwargs.pop("api_key", None) or get_api_key(provider_name)

    # Only pass API key if key is not a placeholder
    if api_key and api_key not in ("None", "NA"):
        kwargs["api_key"] = api_key

    provider_name, model_name, kwargs = _adjust_call_args(
        provider_name, model_name, kwargs
    )
    return LiteLLMEmbeddingWrapper(model=model_name, provider=provider_name, model_config=model_config, **kwargs)


def _parse_chunk(chunk: Any) -> ChatChunk:
    choices = chunk.get("choices") or [{}]
    choice = choices[0] if choices else {}

    raw_delta = choice.get("delta", {})
    raw_message = choice.get("message")
    if not raw_message:
        raw_message = (choice.get("model_extra", {}) or {}).get("message", {})

    delta = _normalize_obj(raw_delta)
    message = _normalize_obj(raw_message)
    model_extra = _normalize_obj(choice.get("model_extra", {}))

    response_delta = _extract_response_text(delta, message, choice)
    if not response_delta:
        # fallback to raw data if normalization removed structured content
        response_delta = _extract_response_text(raw_delta, raw_message, choice)

    reasoning_delta, detected_keys = _extract_reasoning_text(delta, message, model_extra)

    if _DEBUG_REASONING:
        chunk_index = next(_REASONING_CHUNK_COUNTER)
        _debug_reasoning_keys(chunk_index, detected_keys)

    return ChatChunk(
        reasoning_delta=reasoning_delta or "",
        response_delta=response_delta or "",
    )


def _adjust_call_args(provider_name: str, model_name: str, kwargs: dict):
    # for openrouter add app reference
    if provider_name == "openrouter":
        kwargs["extra_headers"] = {
            "HTTP-Referer": "https://agent-zero.ai",
            "X-Title": "Apple Zero",
        }

    # remap other to openai for litellm
    if provider_name == "other":
        provider_name = "openai"

    return provider_name, model_name, kwargs


def _merge_provider_defaults(
    provider_type: str, original_provider: str, kwargs: dict
) -> tuple[str, dict]:
    provider_name = original_provider  # default: unchanged
    cfg = get_provider_config(provider_type, original_provider)
    if cfg:
        provider_name = cfg.get("litellm_provider", original_provider).lower()

        # Extra arguments nested under `kwargs` for readability
        extra_kwargs = cfg.get("kwargs") if isinstance(cfg, dict) else None  # type: ignore[arg-type]
        if isinstance(extra_kwargs, dict):
            for k, v in extra_kwargs.items():
                kwargs.setdefault(k, v)

    # Inject API key based on the *original* provider id if still missing
    if "api_key" not in kwargs:
        key = get_api_key(original_provider)
        if key and key not in ("None", "NA"):
            kwargs["api_key"] = key

    return provider_name, kwargs


def get_chat_model(provider: str, name: str, model_config: Optional[ModelConfig] = None, **kwargs: Any) -> LiteLLMChatWrapper:
    global _last_provider
    orig = provider.lower()

    # Provider switching hook: stop MLX server when switching away from apple_mlx
    if _last_provider == "apple_mlx" and orig != "apple_mlx":
        try:
            manager = MLXServerManager.get_instance()
            status = manager.get_status()
            if status.get("managed"):
                print(f"[MLX Server] Stopping MLX server due to provider switch from {_last_provider} to {orig}")
                result = manager.stop_server()
                if result.get("success"):
                    print("[MLX Server] Server stopped successfully")
                else:
                    print(f"[MLX Server] Failed to stop server: {result.get('message')}")
            else:
                print(
                    "[MLX Server] Skipping stop after provider switch because current "
                    "server is unmanaged"
                )
        except Exception as e:
            print(f"[MLX Server] Error stopping server: {e}")

    _last_provider = orig

    if orig == "apple_mlx":
        # Handle Apple MLX provider via server client
        print("[MLX Server] Checking MLX server status...")

        manager = MLXServerManager.get_instance()
        status = manager.get_status()

        if status["status"] != "running":
            print("[MLX Server] Server not running, starting it...")
            result = manager.start_server()
            if not result.get("success"):
                raise Exception(f"Failed to start MLX server: {result.get('message')}")
            print("[MLX Server] Server started successfully")

        # Create MLXServerClient
        client = MLXServerClient(system_message=kwargs.get("system_message", ""))
        return client

    provider_name, kwargs = _merge_provider_defaults("chat", orig, kwargs)
    return _get_litellm_chat(LiteLLMChatWrapper, name, provider_name, model_config, **kwargs)


def get_browser_model(
    provider: str, name: str, model_config: Optional[ModelConfig] = None, **kwargs: Any
) -> BrowserCompatibleChatWrapper:
    orig = provider.lower()
    provider_name, kwargs = _merge_provider_defaults("chat", orig, kwargs)
    return _get_litellm_chat(
        BrowserCompatibleChatWrapper, name, provider_name, model_config, **kwargs
    )


def get_embedding_model(
    provider: str, name: str, model_config: Optional[ModelConfig] = None, **kwargs: Any
) -> LiteLLMEmbeddingWrapper | LocalSentenceTransformerWrapper:
    orig = provider.lower()
    provider_name, kwargs = _merge_provider_defaults("embedding", orig, kwargs)
    return _get_litellm_embedding(name, provider_name, model_config, **kwargs)

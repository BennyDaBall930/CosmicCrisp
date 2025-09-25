# python/models/apple_mlx_provider.py
"""
Proper MLX provider implementation based on LM Studio's mlx-engine.
Provides robust local MLX model inference with thinking model support.

This implementation integrates key components from LM Studio's mlx-engine
for proper MLX model handling, tokenization, and generation.
"""

from __future__ import annotations
from typing import AsyncIterator, Optional, List, Dict, Any, Iterator, Callable, NamedTuple, Sequence, Union, Tuple
import asyncio, gc, json, threading
from pathlib import Path as FilePath
from contextlib import asynccontextmanager
import uvicorn # Explicitly import uvicorn for type checking

# JSON schema validation
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# MLX and core dependencies
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import load as mlx_load
from mlx_lm.generate import stream_generate as mlx_stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from transformers import AutoTokenizer, AutoProcessor

# FastAPI components
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# HTTP client for MLX server
import httpx

try:
    from langchain_core.messages import BaseMessage
    HAS_LANGCHAIN_MESSAGES = True
except ImportError:  # pragma: no cover - optional dependency
    BaseMessage = object  # type: ignore
    HAS_LANGCHAIN_MESSAGES = False


# ============================================================================
# LM Studio MLX Engine Components (Integrated)
# ============================================================================

class UnsupportedConfigError(Exception):
    pass

def load_model(
    model_path: str | FilePath,
    *,
    vocab_only: bool = False,
    max_kv_size: Optional[int] = 4096,
    trust_remote_code: bool = False,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
) -> "ModelKit":
    model_path = FilePath(model_path)
    config_json = json.loads((model_path / "config.json").read_text())
    model_type = config_json.get("model_type", None)

    # Return ModelKit for standard text models
    return ModelKit(
        model_path,
        vocab_only,
        max_kv_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )

class ModelKit:
    def __init__(
        self,
        model_path: FilePath,
        vocab_only: bool = False,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        self.model_path = model_path
        self.vocab_only = vocab_only
        self.max_kv_size = max_kv_size or 4096
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

        # Load model and tokenizer like LM Studio does
        self.model, self._tokenizer = mlx_load(model_path)

    @property
    def tokenizer(self):
        return self._tokenizer

    def tokenize(self, prompt: str) -> List[int]:
        return self._tokenizer.encode(prompt)


# Generation components from LM Studio
class GenerationStopCondition(NamedTuple):
    stop_reason: str
    stop_string: str
    stop_tokens: List[int]

class GenerationResult(NamedTuple):
    text: str
    tokens: List[Any]  # Simplified for this integration
    top_logprobs: List[List[Any]]
    stop_condition: Optional[GenerationStopCondition]

def create_generator(
    model_kit: ModelKit,
    prompt_tokens: List[int],
    *,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_strings: Optional[List[str]] = None,
    max_tokens: Optional[int] = 100000,
    prompt_progress_callback: Optional[Callable[[float], bool]] = None,
    repetition_penalty: Optional[float] = None,
    max_kv_size: Optional[int] = None,
) -> Iterator[GenerationResult]:
    """Simplified create_generator implementation based on LM Studio's approach."""
    sampler = make_sampler(
        temp=temp or 0.7,
        top_p=top_p or 0.95,
        min_p=min_p or 0.0,
        top_k=top_k or 0,
    )

    # Simulate generation result structure
    text_accumulator = ""

    logits_processors = None
    if repetition_penalty and repetition_penalty != 1.0:
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty
        )

    for chunk in mlx_stream_generate(
        model=model_kit.model,
        tokenizer=model_kit.tokenizer,
        prompt=prompt_tokens,
        max_tokens=max_tokens or 512,
        sampler=sampler,
        max_kv_size=max_kv_size or model_kit.max_kv_size,
        logits_processors=logits_processors,
    ):
        text_accumulator += chunk.text

        # Check for stop strings
        stop_condition = None
        if stop_strings:
            for stop_string in stop_strings:
                if stop_string in text_accumulator:
                    stop_idx = text_accumulator.find(stop_string)
                    text_to_yield = text_accumulator[:stop_idx + len(stop_string)]
                    stop_condition = GenerationStopCondition(
                        stop_reason="stop_string",
                        stop_string=stop_string,
                        stop_tokens=[model_kit.tokenize(stop_string)[-1]]  # Simplified
                    )
                    yield GenerationResult(
                        text=text_to_yield,
                        tokens=[],  # Simplified
                        top_logprobs=[],
                        stop_condition=stop_condition
                    )
                    return

        # Yield current text
        yield GenerationResult(
            text=chunk.text,
            tokens=[],  # Simplified
            top_logprobs=[],
            stop_condition=stop_condition
        )


class AppleMLXProvider:
    """
    MLX provider based on LM Studio's mlx-engine.
    Properly handles Qwen3 thinking models and uses transformers AutoTokenizer.
    """
    def __init__(
        self,
        model_path: str,
        max_kv_size: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        quantization: Optional[str] = None,
        wired_limit_mb: Optional[int] = None,
        cache_limit_mb: Optional[int] = None,
        system_message: str = "",
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ):
        self._model_path = model_path
        # Convert string values to appropriate types to handle settings that may contain strings
        self._max_kv_size = int(max_kv_size) if max_kv_size is not None else 2048
        self._temperature = float(temperature) if temperature is not None else 0.7
        self._top_p = float(top_p) if top_p is not None else 0.95
        self._top_k = int(top_k) if top_k is not None else None
        self._min_p = float(min_p) if min_p is not None else None
        self._quantization = quantization
        self._wired_limit_mb = int(wired_limit_mb) if wired_limit_mb is not None else None
        self._cache_limit_mb = int(cache_limit_mb) if cache_limit_mb is not None else None
        self.system_message = system_message
        self._max_tokens_default = int(max_tokens) if max_tokens is not None else 512
        self._repetition_penalty = float(repetition_penalty) if repetition_penalty is not None else None
        print(f"AppleMLXProvider.__init__: model_path='{model_path}', max_kv_size={self._max_kv_size}, temperature={self._temperature}, top_p={self._top_p}")

        self._model_kit = None
        # Detect thinking models: thinking, yoyo, ThinkCode, etc. (but exclude coder models)
        self._is_thinking_model = ("thinking" in model_path.lower() or
                                   "yoyo" in model_path.lower() or
                                   "thinkcode" in model_path.lower()) and "coder" not in model_path.lower()
        self._lock = asyncio.Lock()

    async def aload(self) -> None:
        async with self._lock:
            if self._wired_limit_mb:
                mx.set_wired_limit(self._wired_limit_mb)
            if self._cache_limit_mb:
                mx.set_cache_limit(self._cache_limit_mb)

            try:
                print(f"Loading model from: {self._model_path}")

                from pathlib import Path
                model_path_obj = Path(self._model_path)

                # Load model using integrated LM Studio approach
                self._model_kit = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: load_model(model_path_obj, max_kv_size=self._max_kv_size)
                )

                # Load transformers tokenizer separately (like LM Studio does)
                self._hf_tokenizer = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: AutoTokenizer.from_pretrained(
                        str(model_path_obj), trust_remote_code=True
                    ),
                )

                # Load chat template if available
                chat_template_path = model_path_obj / "chat_template.jinja"
                if chat_template_path.exists():
                    with open(chat_template_path, "r") as f:
                        self._chat_template = f.read()
                    self._hf_tokenizer.chat_template = self._chat_template
                    print("Chat template loaded successfully")
                else:
                    self._chat_template = None
                    print("No chat template found, using fallback")

                print("Apple MLX model loaded successfully!")

            except Exception as e:
                print(f"Error loading Apple MLX model: {e}")
                import traceback
                traceback.print_exc()
                raise e

    async def aunload(self) -> None:
        async with self._lock:
            self._model_kit = None
            self._hf_tokenizer = None
            self._chat_template = None
            gc.collect()
            mx.clear_cache()  # free MLX buffer cache

    async def astream(
        self,
        messages: Sequence[Union[Dict[str, Any], BaseMessage]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        thinking: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[Dict[str, Any]]]]:
        # Ensure model is loaded
        if self._model_kit is None:
            await self.aload()

        processed_messages = self._normalize_input_messages(messages)

        max_kv_override = None
        if "max_kv_size" in kwargs:
            candidate = kwargs.pop("max_kv_size")
            try:
                max_kv_override = int(candidate) if candidate is not None else None
            except (TypeError, ValueError):
                max_kv_override = None

        # Add system message if not already present
        if self.system_message and not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": self.system_message})

        # Prepare messages for structured output if response_format is specified
        processed_messages = self._prepare_structured_output_prompt(processed_messages, response_format)

        # Apply chat template using transformers AutoTokenizer (like LM Studio does)
        if self._hf_tokenizer and hasattr(self._hf_tokenizer, 'apply_chat_template'):
            chat_template_kwargs = {
                "tokenize": False,  # Get text prompt, not tokens
                "add_generation_prompt": True
            }

            # Enable thinking mode if explicitly requested
            if thinking:
                chat_template_kwargs["thinking"] = True

            prompt = self._hf_tokenizer.apply_chat_template(processed_messages, **chat_template_kwargs)
            print(f"Applied chat template successfully, thinking={thinking}")
        else:
            # Fallback for models without chat template
            prompt = self._build_simple_chat_prompt(processed_messages, thinking)

        # Convert prompt to tokens using the model kit tokenizer
        prompt_tokens = self._model_kit.tokenize(prompt)
        prompt_token_count = len(prompt_tokens)

        # Use LM Studio's create_generator with thinking model support
        generator_kwargs = {
            "temp": self._temperature if temperature is None else temperature,
            "top_p": self._top_p if top_p is None else top_p,
            "min_p": self._min_p,
            "top_k": self._top_k if top_k is None else top_k,
            "stop_strings": stop,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens_default,
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else self._repetition_penalty,
            "max_kv_size": max_kv_override if max_kv_override is not None else self._max_kv_size,
        }

        if kwargs:
            generator_kwargs.update(kwargs)

        def run_generator():
            return create_generator(self._model_kit, prompt_tokens, **generator_kwargs)

        # Run the blocking generator in thread pool
        generation_iterator = await asyncio.get_event_loop().run_in_executor(None, run_generator)

        # Process generation results and handle thinking output
        full_response = ""
        in_think_block = False
        finish_reason = "stop"

        for result in generation_iterator:
            # Handle thinking tags for thinking models
            if self._is_thinking_model:
                processed_text = await self._process_thinking_output(result.text, full_response, in_think_block)
                full_response += result.text  # Always add to accumulator

                if processed_text:
                    yield processed_text, None

                if result.stop_condition:
                    finish_reason = result.stop_condition.stop_reason or "stop"
                    break
            else:
                full_response += result.text
                if result.text:
                    yield result.text, None

                if result.stop_condition:
                    finish_reason = result.stop_condition.stop_reason or "stop"
                    break

        if response_format:
            full_response = self._validate_structured_output(full_response, response_format)

        completion_tokens = len(self._model_kit.tokenize(full_response)) if full_response else 0
        usage = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_token_count + completion_tokens,
        }

        yield "", {"usage": usage, "finish_reason": finish_reason}

    async def _process_thinking_output(self, new_text: str, full_response: str, in_think_block: bool) -> str:
        """Process thinking model output with proper <think> tag handling."""
        accumulated = full_response + new_text

        # Handle thinking tags properly for streaming
        if "<think>" in accumulated and "</think>" not in accumulated:
            if not in_think_block:
                # Just started thinking block - find where <think> begins
                think_start = accumulated.find("<think>")
                if think_start >= 0:
                    # Return content up to and including <think>
                    prefix = accumulated[:think_start + 7]  # Include "<think>"
                    return prefix
            else:
                # Inside thinking block - return the new text as-is
                return new_text
        elif "</think>" in accumulated and in_think_block:
            # Found end of thinking block
            think_end = accumulated.find("</think>")
            if think_end >= 0:
                # Return everything up to and including </think>
                result = accumulated[:think_end + 8]  # Include "</think>"
                return result

        # No special thinking handling needed
        return new_text

    def _prepare_structured_output_prompt(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Prepare messages for structured output by adding system instructions."""
        if not response_format:
            return messages

        response_type = response_format.get("type")

        if response_type == "json_object":
            # Add JSON instruction to the last user message or as a system message
            json_instruction = "You must respond with valid JSON. Do not include any other text, explanations, or formatting outside of the JSON object."

            # If there's a schema, include it in the instruction
            if "json_schema" in response_format:
                schema = response_format["json_schema"]
                schema_str = json.dumps(schema, indent=2)
                json_instruction += f"\n\nFollow this JSON schema:\n{schema_str}"

            # Add as a system message if none exists, otherwise prepend to first user message
            modified_messages = messages.copy()
            has_system = any(msg.get("role") == "system" for msg in modified_messages)

            if has_system:
                # Prepend to the first user message
                for i, msg in enumerate(modified_messages):
                    if msg.get("role") == "user":
                        msg["content"] = f"{json_instruction}\n\n{msg['content']}"
                        break
            else:
                # Add as system message
                modified_messages.insert(0, {"role": "system", "content": json_instruction})

            return modified_messages

        return messages

    def _validate_structured_output(self, response: str, response_format: Optional[Dict[str, Any]]) -> str:
        """Validate and clean structured output response."""
        if not response_format:
            return response

        response_type = response_format.get("type")

        if response_type == "json_object":
            # Try to clean and validate JSON
            try:
                # Strip any leading/trailing whitespace and potential markdown formatting
                cleaned_response = response.strip()

                # Remove potential markdown code block formatting
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]

                cleaned_response = cleaned_response.strip()

                # Parse JSON to validate
                parsed_json = json.loads(cleaned_response)

                # If schema is provided, validate against it
                if "json_schema" in response_format and HAS_JSONSCHEMA:
                    schema = response_format["json_schema"]
                    jsonschema.validate(instance=parsed_json, schema=schema)
                elif "json_schema" in response_format and not HAS_JSONSCHEMA:
                    print("Warning: jsonschema library not available, skipping schema validation")

                # Return the cleaned JSON string
                return json.dumps(parsed_json)

            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                print(f"Structured output validation failed: {e}")
                # Return the original response if validation fails
                return response

        return response

    # Optional: single-shot convenience mirroring ChatWrapper
    async def acomplete(
        self,
        messages: Sequence[Union[Dict[str, Any], BaseMessage]],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        out: List[str] = []
        metadata: Dict[str, Any] = {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "stop",
        }

        async for chunk, info in self.astream(messages, response_format=response_format, **kwargs):
            if chunk:
                out.append(chunk)
            if info:
                metadata = info

        full_response = "".join(out)

        if response_format:
            full_response = self._validate_structured_output(full_response, response_format)

        return full_response, metadata

    async def unified_call(
        self,
        system_message: str = "",
        user_message: str = "",
        messages: Sequence[Union[Dict[str, Any], BaseMessage]] | None = None,
        response_callback=None,
        reasoning_callback=None,
        tokens_callback=None,
        rate_limiter_callback=None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        # Build messages list if not provided
        if messages is None:
            working_messages: List[Union[Dict[str, Any], BaseMessage]] = []
        else:
            working_messages = list(messages)

        if system_message:
            working_messages.insert(0, {"role": "system", "content": system_message})
        if user_message:
            working_messages.append({"role": "user", "content": user_message})

        # Not supporting reasoning for now - return empty string for reasoning
        reasoning = ""

        # Stream the response and collect it
        response = ""
        async for chunk, meta in self.astream(working_messages, response_format=response_format, **kwargs):
            if chunk:
                response += chunk
                if response_callback:
                    await response_callback(chunk, response)
                if tokens_callback:
                    from python.helpers.tokens import approximate_tokens
                    await tokens_callback(chunk, approximate_tokens(chunk))

        return response, reasoning

    def _normalize_input_messages(
        self, messages: Sequence[Union[Dict[str, Any], BaseMessage]]
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        role_mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }

        for message in messages:
            if HAS_LANGCHAIN_MESSAGES and isinstance(message, BaseMessage):
                role = role_mapping.get(getattr(message, "type", ""), getattr(message, "type", ""))
                message_dict: Dict[str, Any] = {
                    "role": role,
                    "content": getattr(message, "content", ""),
                }

                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    new_tool_calls = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            args = tool_call.get("args")
                            if isinstance(args, dict):
                                args_str = json.dumps(args)
                            elif args is not None:
                                args_str = json.dumps(args)
                            else:
                                args_str = "{}"
                            new_tool_calls.append(
                                {
                                    "id": tool_call.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.get("name", ""),
                                        "arguments": args_str,
                                    },
                                }
                            )
                    if new_tool_calls:
                        message_dict["tool_calls"] = new_tool_calls

                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id:
                    message_dict["tool_call_id"] = tool_call_id

                normalized.append(message_dict)
            elif isinstance(message, dict):
                normalized.append({
                    "role": message.get("role"),
                    "content": message.get("content", ""),
                    **{k: v for k, v in message.items() if k not in {"role", "content"}}
                })
            else:
                raise TypeError(
                    "Unsupported message type for AppleMLXProvider: "
                    f"{type(message).__name__}"
                )

        return normalized

    def _build_simple_chat_prompt(self, messages: List[Dict[str, str]], thinking: bool = False) -> str:
        """Fallback chat prompt builder for models without chat template."""
        prompt_parts = []

        if thinking:
            prompt_parts.append("<think>\n")

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                # For thinking models, we need to handle <think> tags in content
                if thinking and not content.startswith("<think>") and not "thinking" in content.lower():
                    prompt_parts.append(f"Assistant: <think>\n{content}")
                else:
                    prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)


class MLXServerClient:
    """
    HTTP client for connecting to the MLX FastAPI server.
    Provides the same interface as AppleMLXProvider but uses HTTP requests.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082",
        timeout: float = 300.0,
        system_message: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.system_message = system_message
        self._client: Optional[httpx.AsyncClient] = None
        try:
            from python.helpers import settings as settings_module

            current_settings = settings_module.get_settings()
        except Exception:
            current_settings = {}

        self._max_tokens_default = int(current_settings.get("mlx_server_max_tokens", 512))
        self._temperature_default = float(current_settings.get("mlx_server_temperature", 0.7))
        self._top_p_default = float(current_settings.get("mlx_server_top_p", 0.95))
        self._top_k_default = int(current_settings.get("mlx_server_top_k", 0))
        self._repetition_penalty_default = float(current_settings.get("mlx_server_repetition_penalty", 1.0))
        self._max_kv_size_default = int(current_settings.get("mlx_server_max_kv_size", 2048))

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_client()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )

    async def _close_client(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _normalize_messages(
        self, messages: Sequence[Union[Dict[str, Any], BaseMessage]]
    ) -> List[Dict[str, Any]]:
        """Convert incoming messages into OpenAI-compatible dicts."""
        normalized: List[Dict[str, Any]] = []
        role_mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }

        for message in messages:
            if HAS_LANGCHAIN_MESSAGES and isinstance(message, BaseMessage):
                role = role_mapping.get(getattr(message, "type", ""), getattr(message, "type", ""))
                message_dict: Dict[str, Any] = {
                    "role": role,
                    "content": getattr(message, "content", ""),
                }

                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    new_tool_calls = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            args = tool_call.get("args")
                            if isinstance(args, dict):
                                args_str = json.dumps(args)
                            elif args is not None:
                                args_str = json.dumps(args)
                            else:
                                args_str = "{}"
                            new_tool_calls.append(
                                {
                                    "id": tool_call.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.get("name", ""),
                                        "arguments": args_str,
                                    },
                                }
                            )
                    if new_tool_calls:
                        message_dict["tool_calls"] = new_tool_calls

                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id:
                    message_dict["tool_call_id"] = tool_call_id

                normalized.append(message_dict)
            elif isinstance(message, dict):
                normalized.append({
                    "role": message.get("role"),
                    "content": message.get("content", ""),
                    **{k: v for k, v in message.items() if k not in {"role", "content"}}
                })
            else:
                raise TypeError(
                    "Unsupported message type for MLXServerClient: "
                    f"{type(message).__name__}"
                )

        return normalized

    async def astream(
        self,
        messages: Sequence[Union[Dict[str, Any], BaseMessage]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_kv_size: Optional[int] = None,
        stop: Optional[List[str]] = None,
        thinking: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response from MLX server using SSE."""
        await self._ensure_client()

        serialized_messages = self._normalize_messages(messages)

        # Prepare request data
        final_top_k = top_k if top_k is not None else self._top_k_default
        if final_top_k == 0:
            final_top_k = None

        final_repetition = (
            repetition_penalty if repetition_penalty is not None else self._repetition_penalty_default
        )
        if final_repetition == 1.0:
            final_repetition = None

        request_data = {
            "messages": serialized_messages,
            "model": "apple-mlx",
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens_default,
            "temperature": temperature if temperature is not None else self._temperature_default,
            "top_p": top_p if top_p is not None else self._top_p_default,
            "top_k": final_top_k,
            "repetition_penalty": final_repetition,
            "max_kv_size": max_kv_size if max_kv_size is not None else self._max_kv_size_default,
            "stream": True,
            "stop": stop,
            "response_format": response_format
        }

        if request_data["top_k"] is None:
            request_data.pop("top_k")
        if request_data["repetition_penalty"] is None:
            request_data.pop("repetition_penalty")

        # Add system message if not already present
        if self.system_message and not any(msg.get("role") == "system" for msg in serialized_messages):
            request_data["messages"].insert(0, {"role": "system", "content": self.system_message})

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=request_data
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

        except httpx.HTTPStatusError as e:
            raise Exception(f"MLX server error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Failed to connect to MLX server: {str(e)}")

    async def acomplete(
        self,
        messages: Sequence[Union[Dict[str, Any], BaseMessage]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_kv_size: Optional[int] = None,
        stop: Optional[List[str]] = None,
        thinking: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Get complete response from MLX server."""
        await self._ensure_client()

        serialized_messages = self._normalize_messages(messages)

        final_top_k = top_k if top_k is not None else self._top_k_default
        if final_top_k == 0:
            final_top_k = None

        final_repetition = (
            repetition_penalty if repetition_penalty is not None else self._repetition_penalty_default
        )
        if final_repetition == 1.0:
            final_repetition = None

        # Prepare request data
        request_data = {
            "messages": serialized_messages,
            "model": "apple-mlx",
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens_default,
            "temperature": temperature if temperature is not None else self._temperature_default,
            "top_p": top_p if top_p is not None else self._top_p_default,
            "top_k": final_top_k,
            "repetition_penalty": final_repetition,
            "max_kv_size": max_kv_size if max_kv_size is not None else self._max_kv_size_default,
            "stream": False,
            "stop": stop,
            "response_format": response_format
        }

        if request_data["top_k"] is None:
            request_data.pop("top_k")
        if request_data["repetition_penalty"] is None:
            request_data.pop("repetition_penalty")

        # Add system message if not already present
        if self.system_message and not any(msg.get("role") == "system" for msg in serialized_messages):
            request_data["messages"].insert(0, {"role": "system", "content": self.system_message})

        try:
            response = await self._client.post("/v1/chat/completions", json=request_data)
            response.raise_for_status()

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("No response content received")

        except httpx.HTTPStatusError as e:
            raise Exception(f"MLX server error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Failed to connect to MLX server: {str(e)}")

    async def unified_call(
        self,
        system_message: str = "",
        user_message: str = "",
        messages: List[Dict[str, str]] | None = None,
        response_callback=None,
        reasoning_callback=None,
        tokens_callback=None,
        rate_limiter_callback=None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Unified call method with callbacks."""
        # Build messages list if not provided
        if messages is None:
            working_messages: List[Union[Dict[str, Any], BaseMessage]] = []
        else:
            working_messages = list(messages)

        if system_message or self.system_message:
            effective_system = system_message or self.system_message
            working_messages.insert(0, {"role": "system", "content": effective_system})
        if user_message:
            working_messages.append({"role": "user", "content": user_message})

        # Not supporting reasoning for now - return empty string for reasoning
        reasoning = ""

        # Stream the response and collect it
        response = ""
        async for chunk in self.astream(working_messages, response_format=response_format, **kwargs):
            response += chunk
            if response_callback:
                await response_callback(chunk, response)
            if tokens_callback:
                from python.helpers.tokens import approximate_tokens
                await tokens_callback(chunk, approximate_tokens(chunk))

        return response, reasoning

    async def health_check(self) -> bool:
        """Check if the MLX server is healthy."""
        await self._ensure_client()

        try:
            response = await self._client.get("/healthz")
            return response.status_code == 200
        except Exception:
            return False


# Global provider instance
_provider: Optional[AppleMLXProvider] = None

# FastAPI app
app = FastAPI(title="Apple MLX Server", version="1.0.0")

# Global settings for startup
_startup_settings: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    global _provider, _startup_settings
    if _startup_settings and _provider is None:
        await init_provider(_startup_settings)
    yield

# FastAPI app with modern lifespan
app = FastAPI(title="Apple MLX Server", version="1.0.0", lifespan=lifespan)

# Request/Response models
class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_kv_size: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[str, Any]] = None

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[ChatCompletionUsage] = None

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "apple-mlx",
                "object": "model",
                "created": 0,
                "owned_by": "apple_mlx"
            }
        ]
    }

async def init_provider(settings: Dict[str, Any]):
    global _provider
    if _provider is None:
        _provider = AppleMLXProvider(
            model_path=settings.get("model_path", ""),
            max_kv_size=settings.get("max_kv_size", 2048),
            temperature=settings.get("temperature", 0.7),
            top_p=settings.get("top_p", 0.95),
            top_k=settings.get("top_k"),
            min_p=settings.get("min_p"),
            quantization=settings.get("quantization"),
            wired_limit_mb=settings.get("wired_limit_mb"),
            cache_limit_mb=settings.get("cache_limit_mb"),
            max_tokens=settings.get("max_tokens"),
            repetition_penalty=settings.get("repetition_penalty"),
        )
        await _provider.aload()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global _provider
    if _provider is None:
        raise HTTPException(status_code=503, detail="Provider not initialized")

    if request.stream:
        async def generate():
            try:
                async for chunk, info in _provider.astream(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    max_kv_size=request.max_kv_size,
                    stop=request.stop,
                    response_format=request.response_format
                ):
                    if chunk:
                        payload = {
                            "id": "chatcmpl-magic",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

                    if info:
                        usage_dict = info.get("usage")
                        finish_reason = info.get("finish_reason", "stop")
                        payload = {
                            "id": "chatcmpl-magic",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                        if usage_dict:
                            payload["usage"] = usage_dict
                        yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Content-Type": "text/event-stream"}
        )
    else:
        try:
            content, metadata = await _provider.acomplete(
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                max_kv_size=request.max_kv_size,
                stop=request.stop,
                response_format=request.response_format
            )
            usage_dict = metadata.get("usage") if metadata else None
            finish_reason = metadata.get("finish_reason", "stop") if metadata else "stop"
            usage_model = ChatCompletionUsage(**usage_dict) if usage_dict else ChatCompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
            return ChatCompletionResponse(
                id="chatcmpl-magic",
                created=0,
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": content},
                    finish_reason=finish_reason
                )],
                usage=usage_model,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python apple_mlx_provider.py <settings_json>")
        sys.exit(1)

    settings_path = sys.argv[1]
    with open(settings_path) as f:
        settings = json.load(f)

    # Check if MLX server is enabled
    if not settings.get("mlx_server_enabled", False):
        print("Apple MLX not enabled")
        sys.exit(1)

    model_path = settings.get("apple_mlx_model_path", "")
    config_data: Dict[str, Any] = {}
    config_path = None
    if model_path:
        config_path = FilePath(model_path) / "generation_config.json"
        if config_path.exists():
            try:
                with open(config_path) as cf:
                    config_data = json.load(cf)
            except Exception as exc:
                print(f"Warning: Failed to read generation_config.json: {exc}")

    def _coerce_float(value, fallback):
        for candidate in (value, fallback):
            if candidate is None:
                continue
            if isinstance(candidate, str) and not candidate.strip():
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return float(fallback) if fallback is not None else 0.0

    def _coerce_int(value, fallback):
        for candidate in (value, fallback):
            if candidate is None:
                continue
            if isinstance(candidate, str) and not candidate.strip():
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        return int(fallback) if fallback is not None else 0

    final_temperature = _coerce_float(settings.get("mlx_server_temperature"), config_data.get("temperature", 0.7))
    final_top_p = _coerce_float(settings.get("mlx_server_top_p"), config_data.get("top_p", 0.95))
    final_top_k = _coerce_int(settings.get("mlx_server_top_k"), config_data.get("top_k", 0))
    effective_top_k = final_top_k if final_top_k > 0 else None
    final_repetition = _coerce_float(settings.get("mlx_server_repetition_penalty"), config_data.get("repetition_penalty", 1.0))
    final_max_tokens = _coerce_int(
        settings.get("mlx_server_max_tokens"),
        config_data.get("max_new_tokens") or config_data.get("max_output_tokens") or 512,
    )
    final_max_kv_size = _coerce_int(settings.get("mlx_server_max_kv_size"), config_data.get("max_kv_size", 2048))

    if config_path and config_path.exists():
        try:
            updated = False
            if config_data.get("temperature") != final_temperature:
                config_data["temperature"] = final_temperature
                updated = True
            if config_data.get("top_p") != final_top_p:
                config_data["top_p"] = final_top_p
                updated = True
            if effective_top_k is not None and config_data.get("top_k") != effective_top_k:
                config_data["top_k"] = effective_top_k
                updated = True
            if config_data.get("repetition_penalty") != final_repetition:
                config_data["repetition_penalty"] = final_repetition
                updated = True
            if config_data.get("max_new_tokens") != final_max_tokens:
                config_data["max_new_tokens"] = final_max_tokens
                updated = True
            if updated:
                with open(config_path, "w") as cf:
                    json.dump(config_data, cf, indent=2)
        except Exception as exc:
            print(f"Warning: Failed to update generation_config.json: {exc}")

    # Prepare settings for provider initialization
    global _startup_settings
    _startup_settings = {
        "model_path": model_path,
        "max_kv_size": final_max_kv_size,
        "temperature": final_temperature,
        "top_p": final_top_p,
        "top_k": effective_top_k,
        "max_tokens": final_max_tokens,
        "repetition_penalty": final_repetition,
        "min_p": settings.get("mlx_server_min_p"),
        "quantization": settings.get("mlx_server_quantization"),
        "wired_limit_mb": settings.get("mlx_server_wired_limit_mb"),
        "cache_limit_mb": settings.get("mlx_server_cache_limit_mb"),
    }

    # Get the correct port from settings
    port = settings.get("mlx_server_port", 8082)
    uvicorn.run(app, host="127.0.0.1", port=port)

if __name__ == "__main__":
    main()

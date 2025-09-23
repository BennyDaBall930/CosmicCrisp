# python/models/apple_mlx_provider.py
"""
Proper MLX provider implementation based on LM Studio's mlx-engine.
Provides robust local MLX model inference with thinking model support.

This implementation integrates key components from LM Studio's mlx-engine
for proper MLX model handling, tokenization, and generation.
"""

from __future__ import annotations
from typing import AsyncIterator, Optional, List, Dict, Any, Iterator, Callable, NamedTuple
import asyncio, gc, uvicorn, json, threading
from pathlib import Path as FilePath

# MLX and core dependencies
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import load as mlx_load
from mlx_lm.generate import stream_generate as mlx_stream_generate
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer, AutoProcessor

# FastAPI components
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


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

    for chunk in mlx_stream_generate(
        model=model_kit.model,
        tokenizer=model_kit.tokenizer,
        prompt=prompt_tokens,
        max_tokens=max_tokens or 512,
        sampler=sampler,
        max_kv_size=model_kit.max_kv_size,
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
    ):
        self._model_path = model_path
        print(f"AppleMLXProvider.__init__: model_path='{model_path}', max_kv_size={max_kv_size}, temperature={temperature}, top_p={top_p}")
        self._max_kv_size = max_kv_size
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._min_p = min_p
        self._quantization = quantization
        self._wired_limit_mb = wired_limit_mb
        self._cache_limit_mb = cache_limit_mb
        self.system_message = system_message

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
                    None, lambda: AutoTokenizer.from_pretrained(str(model_path_obj))
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
            self._model = None
            self._tokenizer = None
            gc.collect()
            mx.clear_cache()  # free MLX buffer cache

    async def astream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        # Ensure model is loaded
        if self._model_kit is None:
            await self.aload()

        # Add system message if not already present
        processed_messages = messages.copy()
        if self.system_message and not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": self.system_message})

        # Apply chat template using transformers AutoTokenizer (like LM Studio does)
        if self._hf_tokenizer and hasattr(self._hf_tokenizer, 'apply_chat_template'):
            chat_template_kwargs = {
                "tokenize": False,  # Get text prompt, not tokens
                "add_generation_prompt": True
            }

            # Enable thinking mode if this is a thinking model or thinking is explicitly requested
            if self._is_thinking_model or thinking:
                chat_template_kwargs["thinking"] = True

            prompt = self._hf_tokenizer.apply_chat_template(processed_messages, **chat_template_kwargs)
            print(f"Applied chat template successfully, thinking={self._is_thinking_model or thinking}")
        else:
            # Fallback for models without chat template
            prompt = self._build_simple_chat_prompt(processed_messages, self._is_thinking_model or thinking)

        # Convert prompt to tokens using the model kit tokenizer
        prompt_tokens = self._model_kit.tokenize(prompt)

        # Use LM Studio's create_generator with thinking model support
        generator_kwargs = {
            "temp": self._temperature if temperature is None else temperature,
            "top_p": self._top_p if top_p is None else top_p,
            "min_p": self._min_p,
            "top_k": self._top_k,
            "stop_strings": stop,
            "max_tokens": max_tokens,
        }

        def run_generator():
            return create_generator(self._model_kit, prompt_tokens, **generator_kwargs)

        # Run the blocking generator in thread pool
        generation_iterator = await asyncio.get_event_loop().run_in_executor(None, run_generator)

        # Process generation results and handle thinking output
        full_response = ""
        in_think_block = False

        for result in generation_iterator:
            # Handle thinking tags for thinking models
            if self._is_thinking_model:
                processed_text = await self._process_thinking_output(result.text, full_response, in_think_block)
                full_response += result.text  # Always add to accumulator

                # Yield the processed chunk (might be empty if in the middle of processing)
                if processed_text:
                    yield processed_text

                # Check for stop condition
                if result.stop_condition:
                    break
            else:
                # Non-thinking models: yield raw text
                full_response += result.text
                yield result.text

                # Check for stop condition
                if result.stop_condition:
                    break

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

    # Optional: single-shot convenience mirroring ChatWrapper
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        out = []
        async for t in self.astream(messages, **kwargs):
            out.append(t)
        return "".join(out)

    async def unified_call(
        self,
        system_message: str = "",
        user_message: str = "",
        messages: List[Dict[str, str]] | None = None,
        response_callback=None,
        reasoning_callback=None,
        tokens_callback=None,
        rate_limiter_callback=None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        # Build messages list if not provided
        if messages is None:
            messages = []
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        if user_message:
            messages.append({"role": "user", "content": user_message})

        # Not supporting reasoning for now - return empty string for reasoning
        reasoning = ""

        # Stream the response and collect it
        response = ""
        async for chunk in self.astream(messages, **kwargs):
            response += chunk
            if response_callback:
                await response_callback(chunk, response)
            if tokens_callback:
                from python.helpers.tokens import approximate_tokens
                await tokens_callback(chunk, approximate_tokens(chunk))

        return response, reasoning

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


# Global provider instance
_provider: Optional[AppleMLXProvider] = None

# FastAPI app
app = FastAPI(title="Apple MLX Server", version="1.0.0")

# Request/Response models
class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

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

def init_provider(settings: Dict[str, Any]):
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
            cache_limit_mb=settings.get("cache_limit_mb")
        )
        asyncio.create_task(_provider.aload())

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global _provider
    if _provider is None:
        raise HTTPException(status_code=503, detail="Provider not initialized")

    if request.stream:
        async def generate():
            try:
                async for chunk in _provider.astream(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop
                ):
                    response = ChatCompletionStreamResponse(
                        id="chatcmpl-magic",
                        created=0,
                        model=request.model,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": chunk}
                        )]
                    )
                    yield f"data: {json.dumps(response.model_dump())}\n\n"
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
            content = await _provider.acomplete(
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop
            )
            return ChatCompletionResponse(
                id="chatcmpl-magic",
                created=0,
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": content},
                    finish_reason="stop"
                )]
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

    mlx_settings = settings.get("apple_mlx", {})
    if not mlx_settings.get("enabled", False):
        print("Apple MLX not enabled")
        sys.exit(1)

    init_provider(mlx_settings)
    uvicorn.run(app, host="127.0.0.1", port=8001)

if __name__ == "__main__":
    main()

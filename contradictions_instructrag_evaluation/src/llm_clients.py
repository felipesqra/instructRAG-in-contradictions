from __future__ import annotations

import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from abc import ABC, abstractmethod
from typing import Any

from config import DeepSeekConfig, GroqConfig, HuggingFaceConfig, ModelArmConfig
from prompt_builder import PromptBundle


class LLMClientError(RuntimeError):
    pass


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: PromptBundle) -> str:
        raise NotImplementedError


class DeepSeekClient(BaseLLMClient):
    def __init__(self, model: ModelArmConfig, config: DeepSeekConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise LLMClientError(f"Missing DeepSeek API key. Set environment variable {config.api_key_env}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError("The openai package is required for provider=deepseek.") from exc
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds)

    def generate(self, prompt: PromptBundle) -> str:
        response = self.client.chat.completions.create(
            model=self.model.model_name,
            messages=prompt.as_chat_messages(),
            temperature=self.model.temperature,
            max_tokens=self.model.max_new_tokens,
            top_p=self.model.top_p,
        )
        return (response.choices[0].message.content or "").strip()


class GroqClient(BaseLLMClient):
    def __init__(self, model: ModelArmConfig, config: GroqConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise LLMClientError(f"Missing GroqCloud API key. Set environment variable {config.api_key_env}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError("The openai package is required for provider=groq.") from exc
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds)

    def generate(self, prompt: PromptBundle) -> str:
        response = self.client.chat.completions.create(
            model=self.model.model_name,
            messages=prompt.as_chat_messages(),
            temperature=self.model.temperature,
            max_tokens=self.model.max_new_tokens,
            top_p=self.model.top_p,
        )
        return (response.choices[0].message.content or "").strip()


class HuggingFaceInferenceProviderClient(BaseLLMClient):
    def __init__(self, model: ModelArmConfig, config: HuggingFaceConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise LLMClientError(
                f"Missing Hugging Face token. Set environment variable {config.api_key_env}."
            )
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = config.timeout_seconds
        self.chat_completions_url = _chat_completions_url(config.base_url)

    def generate(self, prompt: PromptBundle) -> str:
        payload = {
            "model": self.model.model_name,
            "messages": prompt.as_chat_messages(),
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_new_tokens,
            "top_p": self.model.top_p,
            "stream": False,
        }
        request = Request(
            self.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "contradictions-instructrag-evaluation/1.0",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(
                f"Hugging Face Inference Providers request failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except URLError as exc:
            raise LLMClientError(f"Hugging Face Inference Providers request failed: {exc}") from exc

        try:
            parsed_response = json.loads(response_body)
            choice = parsed_response["choices"][0]
            message = choice["message"]
            content = _message_content_to_text(message.get("content"))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMClientError(
                f"Unexpected Hugging Face Inference Providers response: {response_body[:1000]}"
            ) from exc
        if not content:
            raise LLMClientError(
                "Hugging Face Inference Providers returned an empty message.content "
                f"(finish_reason={choice.get('finish_reason')!r}): {response_body[:1000]}"
            )
        return content


class HuggingFaceLocalClient(BaseLLMClient):
    def __init__(self, model: ModelArmConfig, config: HuggingFaceConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise LLMClientError(
                "transformers, torch, and accelerate are required for provider=huggingface_local."
            ) from exc
        token = os.getenv(config.use_auth_token_env) or None
        model_kwargs = {"trust_remote_code": config.trust_remote_code}
        tokenizer_kwargs = {"trust_remote_code": config.trust_remote_code}
        if token:
            model_kwargs["token"] = token
            tokenizer_kwargs["token"] = token
        if config.device == "auto":
            model_kwargs["device_map"] = "auto"
        if config.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        elif config.torch_dtype:
            model_kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.model_name, **tokenizer_kwargs)
        self.generator = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(model.model_name, **model_kwargs),
            tokenizer=self.tokenizer,
        )

    def generate(self, prompt: PromptBundle) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                prompt.as_chat_messages(),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt.prompt_text
        kwargs = {
            "max_new_tokens": self.model.max_new_tokens,
            "return_full_text": False,
            "do_sample": self.model.temperature > 0,
        }
        if self.model.temperature > 0:
            kwargs["temperature"] = self.model.temperature
            kwargs["top_p"] = self.model.top_p
        output = self.generator(text, **kwargs)
        return (output[0].get("generated_text") or "").strip()


class OriginalRagPipelineClient(BaseLLMClient):
    def generate(self, prompt: PromptBundle) -> str:
        raise LLMClientError(
            "provider=original_rag_pipeline was requested, but the cloned rag_pipeline.py uses notebook/global "
            "placeholders for its model client and does not expose a reusable callable generation client. "
            "Use provider=deepseek, provider=groq, or provider=huggingface, or adapt the original pipeline manually."
        )


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


def create_llm_client(
    model: ModelArmConfig,
    deepseek: DeepSeekConfig,
    huggingface: HuggingFaceConfig,
    groq: GroqConfig,
) -> BaseLLMClient:
    if model.provider == "deepseek":
        return DeepSeekClient(model, deepseek)
    if model.provider == "groq":
        return GroqClient(model, groq)
    if model.provider == "huggingface":
        return HuggingFaceInferenceProviderClient(model, huggingface)
    if model.provider == "huggingface_local":
        return HuggingFaceLocalClient(model, huggingface)
    if model.provider == "original_rag_pipeline":
        return OriginalRagPipelineClient()
    raise LLMClientError(f"Unsupported provider: {model.provider}")

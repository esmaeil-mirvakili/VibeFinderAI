"""Provider-agnostic LLM connector layer."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib import error, request

from loguru import logger
from pydantic import BaseModel

from vibefinder.agents import AGENT_OUTPUT_SCHEMAS, agent_output_schema_prompt_specs
from vibefinder.prompts import build_schema_prompt
from vibefinder.tracing import trace_llm_json_call


LLMProvider = Literal["gemini", "openai", "ollama", "local", "self_hosted"]
DEFAULT_LLM_PROVIDER: LLMProvider = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class LLMClient(Protocol):
    """Provider-agnostic structured JSON generation contract."""

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        """Return Pydantic-validated JSON-like output for an agent prompt."""
        ...

    def complete_json_model(
        self,
        prompt: str,
        schema_name: str,
        schema_model: type[BaseModel],
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return Pydantic-validated JSON output for a caller-supplied schema."""
        ...


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for one LLM connector."""

    provider: LLMProvider = DEFAULT_LLM_PROVIDER
    model: str = DEFAULT_GEMINI_MODEL
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Create settings from documented environment variables."""

        provider = _provider_from_env(os.getenv("VIBEFINDER_LLM_PROVIDER", DEFAULT_LLM_PROVIDER))
        model = os.getenv("VIBEFINDER_LLM_MODEL") or _default_model(provider)
        base_url = _base_url_from_env(provider)
        api_key = _api_key_from_env(provider)
        timeout_seconds = _float_env("VIBEFINDER_LLM_TIMEOUT_SECONDS", 60.0)
        temperature = _float_env("VIBEFINDER_LLM_TEMPERATURE", 0.0)
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )


class LLMConnectorError(RuntimeError):
    """Raised when a provider call or JSON validation fails."""


class BaseJSONLLMClient:
    """Shared JSON parsing and schema validation behavior."""

    provider: str

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        """Call the provider, parse JSON, and validate against the named agent schema."""

        if schema_name not in AGENT_OUTPUT_SCHEMAS:
            raise ValueError(f"Unknown agent output schema: {schema_name}")

        return self._complete_json_payload(prompt=prompt, schema_name=schema_name)["validated_output"]

    def complete_json_model(
        self,
        prompt: str,
        schema_name: str,
        schema_model: type[BaseModel],
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the provider and validate JSON against an arbitrary Pydantic model."""

        schema_spec = {
            "schema_name": schema_name,
            "json_schema": schema_model.model_json_schema(),
            "constraints": constraints or {},
        }
        provider_prompt = build_schema_prompt(prompt, schema_spec)
        raw_payload = self._traced_text_completion(
            prompt=provider_prompt,
            schema_name=schema_name,
            schema_spec=schema_spec,
            attempt="initial",
        )
        raw_text = str(raw_payload["raw_response"])
        logger.info(
            "llm_raw_response",
            provider=self.provider,
            schema_name=schema_name,
            raw_response=raw_text,
        )
        parsed, raw_text = self._parse_validate_model_with_retry(
            raw_text=raw_text,
            original_prompt=provider_prompt,
            schema_name=schema_name,
            schema_spec=schema_spec,
            schema_model=schema_model,
        )
        validated = schema_model.model_validate(parsed)
        validated_output = validated.model_dump(mode="json")
        logger.info(
            "llm_complete_json_finished",
            provider=self.provider,
            schema_name=schema_name,
            response=validated_output,
        )
        return validated_output

    def _complete_json_payload(self, *, prompt: str, schema_name: str) -> dict[str, Any]:
        """Return raw and validated structured LLM output for logging/tracing."""

        schema_spec = agent_output_schema_prompt_specs()[schema_name]
        provider_prompt = build_schema_prompt(prompt, schema_spec)
        provider = getattr(self, "provider", self.__class__.__name__)
        model = str(getattr(self, "model", "unknown"))
        raw_payload = self._traced_text_completion(
            prompt=provider_prompt,
            schema_name=schema_name,
            schema_spec=schema_spec,
            attempt="initial",
        )
        raw_text = str(raw_payload["raw_response"])
        logger.info(
            "llm_raw_response",
            provider=self.provider,
            schema_name=schema_name,
            raw_response=raw_text,
        )
        parsed, raw_text = self._parse_and_validate_with_retry(
            raw_text=raw_text,
            original_prompt=provider_prompt,
            schema_name=schema_name,
            schema_spec=schema_spec,
        )
        validated = AGENT_OUTPUT_SCHEMAS[schema_name].model_validate(parsed)
        validated_output = validated.model_dump(mode="json")
        logger.info(
            "llm_complete_json_finished",
            provider=self.provider,
            schema_name=schema_name,
            response=validated_output,
        )
        return {
            "raw_response": raw_text,
            "parsed_response": parsed,
            "validated_output": validated_output,
        }

    def _traced_text_completion(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema_spec: dict[str, Any],
        attempt: str,
    ) -> dict[str, Any]:
        provider = getattr(self, "provider", self.__class__.__name__)
        model = str(getattr(self, "model", "unknown"))
        return trace_llm_json_call(
            provider=provider,
            model=model,
            schema_name=schema_name,
            prompt=prompt,
            execute=lambda: {
                "raw_response": self._complete_text(prompt, schema_name, schema_spec),
            },
            langsmith_extra={
                "metadata": {
                    "application": "vibefinder-ai",
                    "llm_provider": provider,
                    "llm_model": model,
                    "schema_name": schema_name,
                    "llm_attempt": attempt,
                    "ls_provider": provider,
                    "ls_model_name": model,
                    "ls_model_type": "chat",
                },
                "tags": [
                    "vibefinder",
                    f"backend:{provider}",
                    f"model:{model}",
                    f"schema:{schema_name}",
                    f"attempt:{attempt}",
                ],
            },
        )

    def _parse_and_validate_with_retry(
        self,
        *,
        raw_text: str,
        original_prompt: str,
        schema_name: str,
        schema_spec: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        try:
            parsed = _parse_json_object(raw_text)
            AGENT_OUTPUT_SCHEMAS[schema_name].model_validate(parsed)
            return parsed, raw_text
        except Exception as first_exc:
            logger.warning(
                "llm_structured_output_parse_failed",
                provider=self.provider,
                schema_name=schema_name,
                error_type=first_exc.__class__.__name__,
                message=str(first_exc),
                raw_response_preview=raw_text[:1000],
            )

        repair_prompt = _repair_prompt(
            original_prompt=original_prompt,
            raw_text=raw_text,
            schema_name=schema_name,
            schema_spec=schema_spec,
        )
        retry_payload = self._traced_text_completion(
            prompt=repair_prompt,
            schema_name=schema_name,
            schema_spec=schema_spec,
            attempt="structured_output_repair",
        )
        retry_text = str(retry_payload["raw_response"])
        logger.info(
            "llm_raw_response",
            provider=self.provider,
            schema_name=schema_name,
            raw_response=retry_text,
            attempt="structured_output_repair",
        )
        try:
            parsed = _parse_json_object(retry_text)
            AGENT_OUTPUT_SCHEMAS[schema_name].model_validate(parsed)
            return parsed, retry_text
        except Exception as retry_exc:
            logger.warning(
                "llm_structured_output_repair_failed",
                provider=self.provider,
                schema_name=schema_name,
                error_type=retry_exc.__class__.__name__,
                message=str(retry_exc),
                raw_response_preview=retry_text[:1000],
            )
            raise retry_exc

    def _parse_validate_model_with_retry(
        self,
        *,
        raw_text: str,
        original_prompt: str,
        schema_name: str,
        schema_spec: dict[str, Any],
        schema_model: type[BaseModel],
    ) -> tuple[dict[str, Any], str]:
        try:
            parsed = _parse_json_object(raw_text)
            schema_model.model_validate(parsed)
            return parsed, raw_text
        except Exception as first_exc:
            logger.warning(
                "llm_structured_output_parse_failed",
                provider=self.provider,
                schema_name=schema_name,
                error_type=first_exc.__class__.__name__,
                message=str(first_exc),
                raw_response_preview=raw_text[:1000],
            )

        repair_prompt = _repair_prompt(
            original_prompt=original_prompt,
            raw_text=raw_text,
            schema_name=schema_name,
            schema_spec=schema_spec,
        )
        retry_payload = self._traced_text_completion(
            prompt=repair_prompt,
            schema_name=schema_name,
            schema_spec=schema_spec,
            attempt="structured_output_repair",
        )
        retry_text = str(retry_payload["raw_response"])
        logger.info(
            "llm_raw_response",
            provider=self.provider,
            schema_name=schema_name,
            raw_response=retry_text,
            attempt="structured_output_repair",
        )
        try:
            parsed = _parse_json_object(retry_text)
            schema_model.model_validate(parsed)
            return parsed, retry_text
        except Exception as retry_exc:
            logger.warning(
                "llm_structured_output_repair_failed",
                provider=self.provider,
                schema_name=schema_name,
                error_type=retry_exc.__class__.__name__,
                message=str(retry_exc),
                raw_response_preview=retry_text[:1000],
            )
            raise retry_exc

    def _complete_text(
        self,
        prompt: str,
        schema_name: str,
        schema_spec: dict[str, Any],
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class GeminiClient(BaseJSONLLMClient):
    """Google Gemini REST connector."""

    model: str = DEFAULT_GEMINI_MODEL
    api_key: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0

    provider: str = "gemini"

    def _complete_text(
        self,
        prompt: str,
        schema_name: str,
        schema_spec: dict[str, Any],
    ) -> str:
        if not self.api_key:
            raise LLMConnectorError("Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema_spec["json_schema"],
            },
        }
        response = _post_json(url=url, payload=payload, timeout_seconds=self.timeout_seconds)
        try:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMConnectorError("Gemini response did not contain generated text.") from exc


@dataclass(frozen=True)
class OpenAICompatibleClient(BaseJSONLLMClient):
    """OpenAI or OpenAI-compatible chat completions connector."""

    model: str
    api_key: str | None = None
    base_url: str = DEFAULT_OPENAI_BASE_URL
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    provider: str = "openai"

    def _complete_text(
        self,
        prompt: str,
        schema_name: str,
        schema_spec: dict[str, Any],
    ) -> str:
        if self.provider == "openai" and not self.api_key:
            raise LLMConnectorError("OpenAI API key is required. Set OPENAI_API_KEY.")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        response = _post_json(
            url=url,
            payload=payload,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMConnectorError("OpenAI-compatible response did not contain message content.") from exc


@dataclass(frozen=True)
class OllamaClient(BaseJSONLLMClient):
    """Ollama local chat connector."""

    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_seconds: float = 60.0
    temperature: float = 0.0

    provider: str = "ollama"

    def _complete_text(
        self,
        prompt: str,
        schema_name: str,
        schema_spec: dict[str, Any],
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = _post_json(url=url, payload=payload, timeout_seconds=self.timeout_seconds)
        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMConnectorError("Ollama response did not contain message content.") from exc


def create_llm_client(settings: LLMSettings | None = None) -> LLMClient:
    """Create a provider-specific client behind the shared LLM interface."""

    resolved = settings or LLMSettings.from_env()
    provider = _provider_from_env(resolved.provider)
    if provider == "gemini":
        return GeminiClient(
            model=resolved.model or DEFAULT_GEMINI_MODEL,
            api_key=resolved.api_key,
            timeout_seconds=resolved.timeout_seconds,
            temperature=resolved.temperature,
        )
    if provider == "openai":
        return OpenAICompatibleClient(
            model=resolved.model or DEFAULT_OPENAI_MODEL,
            api_key=resolved.api_key,
            base_url=resolved.base_url or DEFAULT_OPENAI_BASE_URL,
            timeout_seconds=resolved.timeout_seconds,
            temperature=resolved.temperature,
            provider="openai",
        )
    if provider in {"local", "self_hosted"}:
        return OpenAICompatibleClient(
            model=resolved.model or DEFAULT_OPENAI_MODEL,
            api_key=resolved.api_key,
            base_url=resolved.base_url or DEFAULT_OPENAI_BASE_URL,
            timeout_seconds=resolved.timeout_seconds,
            temperature=resolved.temperature,
            provider=provider,
        )
    if provider == "ollama":
        return OllamaClient(
            model=resolved.model or DEFAULT_OLLAMA_MODEL,
            base_url=resolved.base_url or DEFAULT_OLLAMA_BASE_URL,
            timeout_seconds=resolved.timeout_seconds,
            temperature=resolved.temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = _strip_markdown_json_fence(text)
    text = _extract_json_object_text(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMConnectorError("LLM response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMConnectorError("LLM response JSON must be an object.")
    return parsed


def _extract_json_object_text(text: str) -> str:
    """Extract a single JSON object from common LLM markdown/prose wrappers."""

    if not text:
        return text
    json_match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


def _strip_markdown_json_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _repair_prompt(
    *,
    original_prompt: str,
    raw_text: str,
    schema_name: str,
    schema_spec: dict[str, Any],
) -> str:
    return (
        "Repair the previous model output into one valid JSON object only.\n"
        f"Schema name: {schema_name}\n"
        "Do not include markdown, explanations, or fields outside the schema.\n"
        "The JSON object must validate against this schema:\n"
        f"{json.dumps(schema_spec['json_schema'], ensure_ascii=True)}\n\n"
        "Original task prompt:\n"
        f"{original_prompt}\n\n"
        "Previous invalid output:\n"
        f"{raw_text[:8000]}"
    )


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_data = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMConnectorError(f"LLM provider HTTP error {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise LLMConnectorError(f"LLM provider connection failed: {exc.reason}") from exc

    try:
        parsed = json.loads(response_data)
    except json.JSONDecodeError as exc:
        raise LLMConnectorError("LLM provider response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMConnectorError("LLM provider response JSON must be an object.")
    return parsed


def _provider_from_env(value: str) -> LLMProvider:
    normalized = value.strip().casefold()
    if normalized in {"gemini", "openai", "ollama", "local", "self_hosted"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unsupported LLM provider: {value}")


def _default_model(provider: LLMProvider) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL
    if provider == "ollama":
        return DEFAULT_OLLAMA_MODEL
    return DEFAULT_OPENAI_MODEL


def _api_key_from_env(provider: LLMProvider) -> str | None:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider in {"openai", "local", "self_hosted"}:
        return os.getenv("OPENAI_API_KEY")
    return None


def _base_url_from_env(provider: LLMProvider) -> str | None:
    if provider == "ollama":
        return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    if provider in {"openai", "local", "self_hosted"}:
        return os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    return None


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)

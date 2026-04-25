from __future__ import annotations

import json
from urllib import error

import pytest

from vibefinder.llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    GeminiClient,
    LLMConnectorError,
    LLMSettings,
    OllamaClient,
    OpenAICompatibleClient,
    create_llm_client,
)


def test_llm_settings_default_to_gemini(monkeypatch):
    _clear_llm_env(monkeypatch)

    settings = LLMSettings.from_env()

    assert settings.provider == "gemini"
    assert settings.model == DEFAULT_GEMINI_MODEL
    assert settings.api_key is None
    assert settings.base_url is None


def test_llm_settings_read_provider_specific_env(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("VIBEFINDER_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VIBEFINDER_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VIBEFINDER_LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("VIBEFINDER_LLM_TEMPERATURE", "0.2")

    settings = LLMSettings.from_env()

    assert settings.provider == "openai"
    assert settings.model == "gpt-test"
    assert settings.api_key == "openai-key"
    assert settings.base_url == "http://localhost:8000/v1"
    assert settings.timeout_seconds == 12.5
    assert settings.temperature == 0.2


def test_create_llm_client_selects_supported_providers():
    assert isinstance(create_llm_client(LLMSettings(provider="gemini", api_key="key")), GeminiClient)
    assert isinstance(
        create_llm_client(LLMSettings(provider="openai", model="gpt-test", api_key="key")),
        OpenAICompatibleClient,
    )
    assert isinstance(
        create_llm_client(LLMSettings(provider="local", model="local-test")),
        OpenAICompatibleClient,
    )
    assert isinstance(
        create_llm_client(LLMSettings(provider="self_hosted", model="hosted-test")),
        OpenAICompatibleClient,
    )
    assert isinstance(create_llm_client(LLMSettings(provider="ollama", model="llama-test")), OllamaClient)


def test_gemini_client_posts_schema_prompt_and_validates_json(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "vibefinder.llm.request.urlopen",
        _fake_urlopen(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "raw_query": "English high energy songs",
                                            "rationale": "The query asks for energetic English songs.",
                                            "feature_targets": [
                                                {"feature": "energy", "direction": "high"}
                                            ],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
            captured,
        ),
    )
    client = GeminiClient(model="gemini-test", api_key="gemini-key", temperature=0.1)

    output = client.complete_json("Extract preferences.", "PreferenceExtractionOutput")

    assert output["raw_query"] == "English high energy songs"
    assert output["feature_targets"][0]["feature"] == "energy"
    assert captured["url"].endswith("/models/gemini-test:generateContent?key=gemini-key")
    assert captured["payload"]["generationConfig"]["responseMimeType"] == "application/json"
    assert (
        captured["payload"]["generationConfig"]["responseJsonSchema"]["title"]
        == "PreferenceExtractionOutput"
    )
    assert captured["payload"]["generationConfig"]["temperature"] == 0.1
    prompt = captured["payload"]["contents"][0]["parts"][0]["text"]
    assert "PreferenceExtractionOutput" in prompt
    assert "Return only a valid JSON object" in prompt


def test_openai_compatible_client_posts_json_request(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "vibefinder.llm.request.urlopen",
        _fake_urlopen(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "issues": ["candidate pool too narrow"],
                                    "should_revise": True,
                                    "summary": "The result set is too small.",
                                    "rationale": "Too few candidates passed verification.",
                                }
                            )
                        }
                    }
                ]
            },
            captured,
        ),
    )
    client = OpenAICompatibleClient(
        model="gpt-test",
        api_key="openai-key",
        base_url=DEFAULT_OPENAI_BASE_URL,
        provider="openai",
    )

    output = client.complete_json("Critique results.", "CritiqueOutput")

    assert output["should_revise"] is True
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer openai-key"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["messages"][0]["role"] == "user"


def test_openai_compatible_local_provider_does_not_require_api_key(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "vibefinder.llm.request.urlopen",
        _fake_urlopen(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "recommendations": [],
                                    "overall_summary": "No final recommendations.",
                                    "rationale": "No scored candidates were available to explain.",
                                }
                            )
                        }
                    }
                ]
            },
            captured,
        ),
    )
    client = OpenAICompatibleClient(
        model="local-test",
        api_key=None,
        base_url="http://localhost:8000/v1",
        provider="local",
    )

    output = client.complete_json("Explain recommendations.", "ExplanationOutput")

    assert output["recommendations"] == []
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in captured["headers"]


def test_ollama_client_posts_local_json_request(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "vibefinder.llm.request.urlopen",
        _fake_urlopen(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "candidates": [],
                            "summary": "No candidates to verify.",
                            "rationale": "The candidate list was empty.",
                        }
                    )
                }
            },
            captured,
        ),
    )
    client = OllamaClient(model="llama-test", base_url=DEFAULT_OLLAMA_BASE_URL)

    output = client.complete_json("Verify candidates.", "VerificationOutput")

    assert output["summary"] == "No candidates to verify."
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["stream"] is False


def test_connector_rejects_unknown_schema_and_invalid_model_output(monkeypatch):
    client = GeminiClient(model="gemini-test", api_key="key")

    with pytest.raises(ValueError, match="Unknown agent output schema"):
        client.complete_json("Prompt.", "MissingSchema")

    monkeypatch.setattr(
        "vibefinder.llm.request.urlopen",
        _fake_urlopen({"candidates": [{"content": {"parts": [{"text": "{bad json"}]}}]}),
    )
    with pytest.raises(LLMConnectorError, match="not valid JSON"):
        client.complete_json("Prompt.", "PreferenceExtractionOutput")

    monkeypatch.setattr(
        "vibefinder.llm.request.urlopen",
        _fake_urlopen(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "raw_query": "x",
                                            "rationale": "Reject extra fields.",
                                            "bad": 1,
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    )
    with pytest.raises(ValueError):
        client.complete_json("Prompt.", "PreferenceExtractionOutput")


def test_llm_payload_keeps_raw_response_before_json_validation(monkeypatch):
    captured_raw: dict = {}

    def capture_trace_call(**kwargs):
        payload = kwargs["execute"]()
        captured_raw.update(payload)
        return payload

    monkeypatch.setattr(
        "vibefinder.llm.request.urlopen",
        _fake_urlopen({"candidates": [{"content": {"parts": [{"text": "{bad json"}]}}]}),
    )
    monkeypatch.setattr("vibefinder.llm.trace_llm_json_call", capture_trace_call)
    client = GeminiClient(model="gemini-test", api_key="key")

    with pytest.raises(LLMConnectorError, match="not valid JSON"):
        client._complete_json_payload(prompt="Prompt.", schema_name="PreferenceExtractionOutput")

    assert captured_raw == {"raw_response": "{bad json"}


def test_connector_surfaces_provider_http_errors(monkeypatch):
    def failing_urlopen(*args, **kwargs):
        raise error.URLError("connection refused")

    monkeypatch.setattr("vibefinder.llm.request.urlopen", failing_urlopen)
    client = OllamaClient(model="llama-test")

    with pytest.raises(LLMConnectorError, match="connection failed"):
        client.complete_json("Prompt.", "VerificationOutput")


def test_openai_requires_api_key_for_hosted_provider():
    client = OpenAICompatibleClient(model="gpt-test", api_key=None, provider="openai")

    with pytest.raises(LLMConnectorError, match="OpenAI API key is required"):
        client.complete_json("Prompt.", "CritiqueOutput")


def test_agents_package_does_not_import_provider_sdks():
    import sys
    import vibefinder.agents  # noqa: F401

    assert "google.generativeai" not in sys.modules
    assert "openai" not in sys.modules
    assert "ollama" not in sys.modules


def _fake_urlopen(response_payload: dict, captured: dict | None = None):
    def fake_urlopen(http_request, timeout):
        if captured is not None:
            captured["url"] = http_request.full_url
            captured["headers"] = dict(http_request.header_items())
            captured["payload"] = json.loads(http_request.data.decode("utf-8"))
            captured["timeout"] = timeout
        return _FakeHTTPResponse(response_payload)

    return fake_urlopen


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _clear_llm_env(monkeypatch):
    for key in (
        "VIBEFINDER_LLM_PROVIDER",
        "VIBEFINDER_LLM_MODEL",
        "VIBEFINDER_LLM_TIMEOUT_SECONDS",
        "VIBEFINDER_LLM_TEMPERATURE",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

from configuration.config import (
    get_chat_model_settings,
    get_context_compression_settings,
    get_vector_service_settings,
)


def test_chat_model_settings_use_omlx_api_key(monkeypatch) -> None:
    get_chat_model_settings.cache_clear()
    monkeypatch.setenv("OMLX_API_KEY", "e2e-test-key")
    try:
        assert get_chat_model_settings().api_key == "e2e-test-key"
    finally:
        get_chat_model_settings.cache_clear()


def test_chat_model_settings_keep_local_fallback(monkeypatch) -> None:
    get_chat_model_settings.cache_clear()
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    try:
        assert get_chat_model_settings().api_key == "local"
    finally:
        get_chat_model_settings.cache_clear()


def test_system_prompt_requires_filename_lookup_before_capability_denial() -> None:
    get_chat_model_settings.cache_clear()
    try:
        prompt = get_chat_model_settings().system_prompt or ""
        assert "必须先调用 find_ingested_files" in prompt
        assert "read_knowledge_file" in prompt
        assert "read_credential_file" in prompt
        assert "不能在查询前声称无法访问文件" in prompt
    finally:
        get_chat_model_settings.cache_clear()


def test_context_compression_retains_headroom_after_summary() -> None:
    settings = get_context_compression_settings()

    assert settings.trigger_tokens == 4_000
    assert settings.keep_messages == 8


def test_vector_service_settings_use_independent_environment(monkeypatch) -> None:
    get_vector_service_settings.cache_clear()
    monkeypatch.setenv("VECTOR_SERVICE_BASE_URL", "http://127.0.0.1:18083/")
    monkeypatch.setenv("VECTOR_SERVICE_TIMEOUT", "12.5")
    try:
        settings = get_vector_service_settings()
        assert settings.base_url == "http://127.0.0.1:18083"
        assert settings.timeout == 12.5
    finally:
        get_vector_service_settings.cache_clear()

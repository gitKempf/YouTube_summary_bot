import pytest
from src.config import get_config, Config


def test_config_loads_telegram_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key")
    config = get_config()
    assert config.telegram_bot_token == "test-token"


def test_config_loads_elevenlabs_key(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key")
    config = get_config()
    assert config.elevenlabs_api_key == "test-el-key"


def test_config_loads_anthropic_key(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key")
    config = get_config()
    assert config.anthropic_api_key == "test-ant-key"


def test_config_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing required environment variables"):
        get_config()


def test_config_anthropic_key_optional(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "e")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = get_config()
    assert config.anthropic_api_key == ""


def test_config_has_defaults(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key")
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("REQUIRED_CHANNEL", raising=False)
    config = get_config()
    assert config.tts_voice == "en-US-RogerNeural"
    assert config.claude_model == "claude-sonnet-4-6"
    assert config.max_tokens == 4096
    assert config.allowed_user_ids == frozenset()
    assert config.required_channel == ""


def test_config_loads_required_channel(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "e")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("REQUIRED_CHANNEL", "@alexkampf")
    config = get_config()
    assert config.required_channel == "@alexkampf"


def test_config_loads_allowed_user_ids(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key")
    monkeypatch.setenv("ALLOWED_USER_IDS", "123,456,789")
    config = get_config()
    assert config.allowed_user_ids == frozenset({123, 456, 789})


def test_is_user_allowed_empty_whitelist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "e")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    config = get_config()
    assert config.is_user_allowed(999) is True


def test_is_user_allowed_in_whitelist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "e")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("ALLOWED_USER_IDS", "123,456")
    config = get_config()
    assert config.is_user_allowed(123) is True
    assert config.is_user_allowed(999) is False


def test_config_memory_disabled_by_default(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "e")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.delenv("MEM0_ENABLED", raising=False)
    config = get_config()
    assert config.memory_enabled is False


def test_config_memory_enabled_when_configured(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "e")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("MEM0_ENABLED", "true")
    monkeypatch.setenv("MEM0_QDRANT_HOST", "qdrant.example.com")
    monkeypatch.setenv("MEM0_NEO4J_URL", "bolt://graph:7687")
    config = get_config()
    assert config.memory_enabled is True
    assert config.qdrant_host == "qdrant.example.com"
    assert config.neo4j_url == "bolt://graph:7687"

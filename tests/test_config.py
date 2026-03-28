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


def test_config_has_defaults(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key")
    config = get_config()
    assert config.tts_voice == "en-US-RogerNeural"
    assert config.claude_model == "claude-sonnet-4-6"
    assert config.max_tokens == 4096

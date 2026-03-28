import os
from dataclasses import dataclass, field
from typing import Set
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    elevenlabs_api_key: str
    anthropic_api_key: str
    allowed_user_ids: frozenset = field(default_factory=frozenset)
    tts_voice: str = "en-US-RogerNeural"
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096

    def is_user_allowed(self, user_id: int) -> bool:
        if not self.allowed_user_ids:
            return True  # empty whitelist = allow all
        return user_id in self.allowed_user_ids


def get_config() -> Config:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    el_key = os.getenv("ELEVENLABS_API_KEY")
    ant_key = os.getenv("ANTHROPIC_API_KEY")

    missing = []
    if not token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not el_key:
        missing.append("ELEVENLABS_API_KEY")
    if not ant_key:
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    raw_ids = os.getenv("ALLOWED_USER_IDS", "")
    allowed = frozenset(
        int(uid.strip()) for uid in raw_ids.split(",") if uid.strip()
    )

    return Config(
        telegram_bot_token=token,
        elevenlabs_api_key=el_key,
        anthropic_api_key=ant_key,
        allowed_user_ids=allowed,
    )

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_update():
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    update.message.reply_text = AsyncMock()
    update.message.reply_voice = AsyncMock()
    update.message.chat_id = 12345
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    return update


@pytest.fixture
def mock_context():
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_voice = AsyncMock()
    context.bot.send_message = AsyncMock()
    context.bot.send_chat_action = AsyncMock()
    return context

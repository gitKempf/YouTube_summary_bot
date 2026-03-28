import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_status_msg():
    msg = MagicMock()
    msg.edit_text = AsyncMock()
    return msg


@pytest.fixture
def mock_update(mock_status_msg):
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
    update.message.reply_voice = AsyncMock()
    update.message.chat_id = 12345
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    update.effective_user = MagicMock()
    update.effective_user.id = 12345
    return update


@pytest.fixture
def mock_context():
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_voice = AsyncMock()
    context.bot.send_message = AsyncMock()
    context.bot.send_chat_action = AsyncMock()
    context.bot_data = {}
    return context

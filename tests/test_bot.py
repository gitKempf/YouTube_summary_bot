import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.bot import start_command, handle_message, _progress_bar, split_message, ProgressTracker
from src.downloader import TranscriptFetchResult
from src.transcriber import TranscriptionError, TranscriptionResult
from src.summarizer import SummarizationError
from src.tts import TTSError
from src.fact_checker import Claim, ClassifiedClaim, ClaimStatus, FactCheckResult


def _make_memory_mgr_mock(**overrides):
    """Create a minimal MemoryManager mock that returns per-user API keys."""
    mgr = MagicMock()
    mgr.get_user_settings = AsyncMock(
        return_value={"anthropic_api_key": "fake-user-key", "elevenlabs_api_key": "fake-el-key"}
    )
    mgr.save_user_settings = AsyncMock()
    mgr.search = AsyncMock(return_value=[])
    mgr.add = AsyncMock()
    mgr.add_if_new = AsyncMock(return_value=True)
    mgr.store_transcript = AsyncMock()
    for k, v in overrides.items():
        setattr(mgr, k, v)
    return mgr


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.telegram_bot_token = "fake-token"
    config.elevenlabs_api_key = "fake-el-key"
    config.anthropic_api_key = ""
    config.tts_voice = "en-US-RogerNeural"
    config.claude_model = "claude-sonnet-4-6"
    config.max_tokens = 4096
    config.memory_enabled = False
    config.is_user_allowed.return_value = True
    return config


@pytest.fixture
def mock_status_msg():
    msg = MagicMock()
    msg.edit_text = AsyncMock()
    return msg


def _happy_patches(mock_config, transcript=None, summary="Summary"):
    if transcript is None:
        transcript = TranscriptFetchResult(text="Transcript", language_code="en")

    return {
        "config": patch("src.bot.get_config", return_value=mock_config),
        "vid": patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"),
        "thread": patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                        return_value=transcript),
        "title": patch("src.bot.fetch_video_title", return_value="Test Video"),
        "sum": patch("src.bot.summarize_text", new_callable=AsyncMock,
                     return_value=summary),
        "voice": patch("src.bot.get_voice_for_language",
                       return_value="en-US-RogerNeural"),
        "tts": patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                     return_value=[Path("/tmp/v.ogg")]),
        "open": patch("builtins.open", MagicMock()),
        "exists": patch("src.bot.Path.exists", return_value=False),
        "unlink": patch("src.bot.Path.unlink"),
    }


class TestProgressBar:
    def test_zero_percent(self):
        result = _progress_bar(0, "Starting...")
        assert "0%" in result
        assert "Starting..." in result

    def test_hundred_percent(self):
        result = _progress_bar(100, "Done!")
        assert "100%" in result
        assert "Done!" in result

    def test_mid_percent(self):
        result = _progress_bar(50, "Summarizing...")
        assert "50%" in result

    def test_bar_width(self):
        result = _progress_bar(50, "Test")
        bar_line = result.split("\n")[0]
        assert len(bar_line.split(" ")[0]) == 15  # PROGRESS_BAR_WIDTH


class TestProgressTracker:
    @pytest.mark.asyncio
    async def test_updates_message(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        tracker = ProgressTracker(msg)
        await tracker.update(50, "Working...")
        msg.edit_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_duplicate_update(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        tracker = ProgressTracker(msg)
        await tracker.update(50, "Working...")
        await tracker.update(50, "Working...")
        assert msg.edit_text.await_count == 1

    @pytest.mark.asyncio
    async def test_handles_telegram_error(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=Exception("Rate limited"))
        tracker = ProgressTracker(msg)
        await tracker.update(50, "Working...")  # Should not raise


class TestSplitMessage:
    def test_short_message_unchanged(self):
        assert split_message("Hello") == ["Hello"]

    def test_long_message_splits(self):
        text = "\n\n".join([f"Paragraph {i}." for i in range(100)])
        parts = split_message(text, max_length=200)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 200

    def test_preserves_all_content(self):
        text = "\n\n".join([f"Paragraph {i}." for i in range(20)])
        parts = split_message(text, max_length=100)
        combined = "\n\n".join(parts)
        for i in range(20):
            assert f"Paragraph {i}." in combined


class TestAccessControl:
    @pytest.mark.asyncio
    async def test_whitelist_user_allowed(self, mock_update, mock_context):
        mock_config = MagicMock()
        mock_config.is_user_allowed.return_value = True
        with patch("src.bot.get_config", return_value=mock_config):
            await start_command(mock_update, mock_context)
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Welcome" in call_text

    @pytest.mark.asyncio
    async def test_channel_member_allowed(self, mock_update, mock_context):
        mock_config = MagicMock()
        mock_config.is_user_allowed.return_value = False
        mock_config.required_channel = "@testchannel"
        mock_member = MagicMock()
        mock_member.status = "member"
        mock_context.bot.get_chat_member = AsyncMock(return_value=mock_member)
        with patch("src.bot.get_config", return_value=mock_config):
            await start_command(mock_update, mock_context)
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Welcome" in call_text

    @pytest.mark.asyncio
    async def test_non_member_blocked(self, mock_update, mock_context):
        mock_update.effective_user.id = 999
        mock_config = MagicMock()
        mock_config.is_user_allowed.return_value = False
        mock_config.required_channel = "@testchannel"
        mock_member = MagicMock()
        mock_member.status = "left"
        mock_context.bot.get_chat_member = AsyncMock(return_value=mock_member)
        with patch("src.bot.get_config", return_value=mock_config):
            await start_command(mock_update, mock_context)
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "join" in call_text.lower()

    @pytest.mark.asyncio
    async def test_channel_check_error_blocks(self, mock_update, mock_context):
        mock_update.effective_user.id = 999
        mock_config = MagicMock()
        mock_config.is_user_allowed.return_value = False
        mock_config.required_channel = "@testchannel"
        mock_context.bot.get_chat_member = AsyncMock(side_effect=Exception("API error"))
        with patch("src.bot.get_config", return_value=mock_config):
            await start_command(mock_update, mock_context)
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "join" in call_text.lower()


class TestStartCommand:
    @pytest.mark.asyncio
    async def test_sends_welcome_message(self, mock_update, mock_context, mock_config):
        with patch("src.bot.get_config", return_value=mock_config):
            await start_command(mock_update, mock_context)
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "YouTube" in call_text or "youtube" in call_text.lower()


class TestHandleMessageWithCaptions:
    @pytest.mark.asyncio
    async def test_progress_bar_updates(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config)
        with p["config"], p["vid"], p["thread"], p["title"], p["sum"], p["voice"], \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        # Should have multiple progress edits (start, transcript, summarize, voice, done)
        assert mock_status_msg.edit_text.await_count >= 4

    @pytest.mark.asyncio
    async def test_uses_youtube_captions(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config,
                           TranscriptFetchResult(text="Caption text", language_code="en"))
        with p["config"], p["vid"], p["thread"] as mt, p["title"], p["sum"] as ms, \
             p["voice"], p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        assert mt.call_count == 1  # fetch_transcript only (title fetched directly)
        assert "Caption text" in str(ms.call_args)

    @pytest.mark.asyncio
    async def test_sends_summary_and_voice(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config, summary="My summary")
        with p["config"], p["vid"], p["thread"], p["title"], p["sum"], p["voice"], \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        calls = mock_update.message.reply_text.call_args_list
        assert any("My summary" in str(c) for c in calls)
        mock_update.message.reply_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tts_uses_caption_language_when_available(self, mock_update, mock_context, mock_config, mock_status_msg):
        """When captions exist, TTS uses the caption language code."""
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config,
                           TranscriptFetchResult(text="Bonjour le monde", language_code="fr"))
        with p["config"], p["vid"], p["thread"], p["title"], p["sum"], \
             patch("src.bot.get_voice_for_language", return_value="fr-FR-HenriNeural") as gv, \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        gv.assert_called_once_with("fr")


class TestHandleMessageFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_audio_download(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/a.mp4"),
                                TranscriptionResult(text="EL text", language_code="en"),
                                "Fallback Video"]) as mt, \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="S") as ms, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        assert mt.call_count == 3  # fetch_transcript + download + transcribe
        assert "EL text" in str(ms.call_args)


class TestHandleMessageErrors:
    @pytest.mark.asyncio
    async def test_download_error(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, RuntimeError("Fail")]), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        assert "Sorry" in mock_update.message.reply_text.call_args_list[-1][0][0]

    @pytest.mark.asyncio
    async def test_tts_failure_still_sends_text(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="T", language_code="en")), \
             patch("src.bot.fetch_video_title", return_value="T"), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Summary"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   side_effect=TTSError("Fail")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        calls = mock_update.message.reply_text.call_args_list
        assert any("Summary" in str(c) for c in calls)
        mock_update.message.reply_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/a.mp4"), TranscriptionError("Fail")]), \
             patch("src.bot.Path.exists", return_value=True), \
             patch("src.bot.Path.unlink") as mu:
            await handle_message(mock_update, mock_context)
        assert mu.called


class TestMemoryIntegration:
    def _mem_config(self):
        c = MagicMock()
        c.memory_enabled = True
        c.is_user_allowed.return_value = True
        c.anthropic_api_key = ""
        c.claude_model = "claude-sonnet-4-6"
        c.max_tokens = 4096
        return c

    def _mem_mgr(self):
        return _make_memory_mgr_mock()

    @pytest.mark.asyncio
    async def test_pipeline_uses_memory_when_enabled(self, mock_update, mock_context, mock_status_msg):
        cfg = self._mem_config()
        mgr = self._mem_mgr()
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        mock_update.effective_user.id = 12345
        mock_context.bot_data = {"memory_mgr": mgr}

        fact_result = FactCheckResult(
            new_claims=[ClassifiedClaim(Claim("New fact", "X", "is", "Y"), ClaimStatus.NEW)],
            supported_claims=[], contradicted_claims=[], context_summary="",
        )

        with patch("src.bot.get_config", return_value=cfg), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="T", language_code="en")), \
             patch("src.bot.extract_claims", new_callable=AsyncMock,
                   return_value=[Claim("New fact", "X", "is", "Y")]), \
             patch("src.bot.classify_claims", new_callable=AsyncMock,
                   return_value=fact_result), \
             patch("src.bot.build_context_prompt", return_value="Context here"), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Sum") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        assert mock_sum.call_args[1].get("past_context") == "Context here"
        mgr.add_if_new.assert_awaited()

    @pytest.mark.asyncio
    async def test_pipeline_skips_memory_when_disabled(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_config.memory_enabled = False
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config)
        with p["config"], p["vid"], p["thread"], p["title"], p["sum"] as mock_sum, p["voice"], \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        assert mock_sum.call_args[1].get("past_context") is None

    @pytest.mark.asyncio
    async def test_memory_failure_falls_back(self, mock_update, mock_context, mock_status_msg):
        cfg = self._mem_config()
        mgr = self._mem_mgr()
        mgr.search = AsyncMock(side_effect=Exception("DB error"))
        mock_context.bot_data = {"memory_mgr": mgr}
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])

        with patch("src.bot.get_config", return_value=cfg), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="T", language_code="en")), \
             patch("src.bot.extract_claims", new_callable=AsyncMock,
                   side_effect=Exception("LLM failed")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Sum") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        mock_sum.assert_awaited_once()
        assert mock_sum.call_args[1].get("past_context") is None

    @pytest.mark.asyncio
    async def test_memory_store_failure_no_break(self, mock_update, mock_context, mock_status_msg):
        cfg = self._mem_config()
        mgr = self._mem_mgr()
        mgr.add_if_new = AsyncMock(side_effect=Exception("Write failed"))
        mock_context.bot_data = {"memory_mgr": mgr}
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])

        fact_result = FactCheckResult(
            new_claims=[ClassifiedClaim(Claim("Fact", "X", "is", "Y"), ClaimStatus.NEW)],
            supported_claims=[], contradicted_claims=[], context_summary="",
        )

        with patch("src.bot.get_config", return_value=cfg), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="T", language_code="en")), \
             patch("src.bot.extract_claims", new_callable=AsyncMock,
                   return_value=[Claim("Fact", "X", "is", "Y")]), \
             patch("src.bot.classify_claims", new_callable=AsyncMock,
                   return_value=fact_result), \
             patch("src.bot.build_context_prompt", return_value="Ctx"), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Sum"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        calls = [str(c) for c in mock_update.message.reply_text.call_args_list]
        assert any("Sum" in c for c in calls)

    @pytest.mark.asyncio
    async def test_user_id_passed_correctly(self, mock_update, mock_context, mock_status_msg):
        cfg = self._mem_config()
        mgr = self._mem_mgr()
        mock_context.bot_data = {"memory_mgr": mgr}
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        mock_update.effective_user.id = 99999

        fact_result = FactCheckResult(
            new_claims=[ClassifiedClaim(Claim("F", "X", "is", "Y"), ClaimStatus.NEW)],
            supported_claims=[], contradicted_claims=[], context_summary="",
        )

        with patch("src.bot.get_config", return_value=cfg), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="T", language_code="en")), \
             patch("src.bot.extract_claims", new_callable=AsyncMock,
                   return_value=[Claim("F", "X", "is", "Y")]), \
             patch("src.bot.classify_claims", new_callable=AsyncMock, return_value=fact_result), \
             patch("src.bot.build_context_prompt", return_value=""), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="S"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        search_uid = mgr.search.call_args[1]["user_id"]
        assert search_uid == "tg_99999"


class TestApiKeyRequired:
    @pytest.mark.asyncio
    async def test_no_key_rejects_with_message(self, mock_update, mock_context, mock_config, mock_status_msg):
        """User without API key gets a rejection message, not a summary."""
        no_key_mgr = _make_memory_mgr_mock()
        no_key_mgr.get_user_settings = AsyncMock(return_value={})
        mock_context.bot_data = {"memory_mgr": no_key_mgr}
        mock_update.message.reply_text = AsyncMock()
        with patch("src.bot.get_config", return_value=mock_config):
            await handle_message(mock_update, mock_context)
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "API key" in call_text

    @pytest.mark.asyncio
    async def test_no_memory_mgr_rejects(self, mock_update, mock_context, mock_config, mock_status_msg):
        """No memory manager means no way to look up key — reject."""
        mock_context.bot_data = {}
        mock_update.message.reply_text = AsyncMock()
        with patch("src.bot.get_config", return_value=mock_config):
            await handle_message(mock_update, mock_context)
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "API key" in call_text

    @pytest.mark.asyncio
    async def test_with_key_proceeds(self, mock_update, mock_context, mock_config, mock_status_msg):
        """User with API key configured should proceed normally."""
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config)
        with p["config"], p["vid"], p["thread"], p["title"], p["sum"] as mock_sum, p["voice"], \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        mock_sum.assert_awaited_once()

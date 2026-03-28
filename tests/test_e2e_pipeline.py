"""
End-to-end pipeline test: verifies the full memory-aware summarization pipeline
using mocked external services but real data flow.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import json

from src.bot import handle_message
from src.downloader import TranscriptFetchResult
from src.fact_checker import Claim, ClassifiedClaim, ClaimStatus, FactCheckResult
from src.memory import MemoryEntry


@pytest.fixture
def mock_status_msg():
    msg = MagicMock()
    msg.edit_text = AsyncMock()
    return msg


@pytest.fixture
def pipeline_config():
    c = MagicMock()
    c.memory_enabled = True
    c.is_user_allowed.return_value = True
    c.anthropic_api_key = "fake"
    c.claude_model = "claude-sonnet-4-6"
    c.max_tokens = 4096
    return c


@pytest.fixture
def pipeline_mgr():
    m = MagicMock()
    m.search = AsyncMock(return_value=[])
    m.add = AsyncMock()
    return m


class TestE2EPipelineVideoOne:
    """Simulate processing the first video — no prior memories."""

    @pytest.mark.asyncio
    async def test_first_video_stores_all_claims_as_new(
        self, mock_status_msg, pipeline_config, pipeline_mgr
    ):
        update = MagicMock()
        update.message = MagicMock()
        update.message.text = "https://www.youtube.com/watch?v=test1"
        update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        update.message.reply_voice = AsyncMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 42
        update.effective_chat = MagicMock()

        context = MagicMock()
        context.bot_data = {"memory_mgr": pipeline_mgr}
        context.bot = MagicMock()
        context.bot.get_chat_member = AsyncMock()

        video1_claims = [
            Claim("AI agents need structured memory", "AI agents", "need", "structured memory",
                  confidence=0.9, video_id="test1"),
            Claim("Claude Code uses CLAUDE.md for context", "Claude Code", "uses", "CLAUDE.md",
                  confidence=0.95, video_id="test1"),
        ]
        fact_result = FactCheckResult(
            new_claims=[ClassifiedClaim(c, ClaimStatus.NEW) for c in video1_claims],
            supported_claims=[], contradicted_claims=[], context_summary="",
        )

        with patch("src.bot.get_config", return_value=pipeline_config), \
             patch("src.bot.extract_video_id", return_value="test1"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="transcript", language_code="en")), \
             patch("src.bot.extract_claims", new_callable=AsyncMock, return_value=video1_claims), \
             patch("src.bot.classify_claims", new_callable=AsyncMock, return_value=fact_result), \
             patch("src.bot.build_context_prompt", return_value="FOCUS ON NEW TOPICS:\n- AI agents\n- Claude Code"), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Summary V1") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(update, context)

        # Verify: summarize was called with context
        assert "FOCUS ON NEW TOPICS" in mock_sum.call_args[1]["past_context"]

        # Verify: 2 claims stored individually (not as blob)
        assert pipeline_mgr.add.await_count == 2

        # Verify: each claim stored with structured metadata
        for call in pipeline_mgr.add.call_args_list:
            meta = call[1].get("metadata", {})
            assert "entity" in meta
            assert "relation" in meta
            assert "value" in meta
            assert "confidence" in meta
            assert "video_id" in meta
            assert "timestamp" in meta
            assert meta["video_id"] == "test1"

        # Verify: user_id is correctly formatted
        assert pipeline_mgr.add.call_args_list[0][1]["user_id"] == "tg_42"


class TestE2EPipelineVideoTwo:
    """Simulate processing the second video — with prior memories from video 1."""

    @pytest.mark.asyncio
    async def test_second_video_detects_supported_claims(
        self, mock_status_msg, pipeline_config
    ):
        # Memory manager returns prior knowledge from video 1
        mgr = MagicMock()
        mgr.search = AsyncMock(return_value=[
            MemoryEntry(id="m1", text="AI agents need structured memory", score=0.9,
                        metadata={"entity": "AI agents", "confidence": 0.9}),
        ])
        mgr.add = AsyncMock()

        update = MagicMock()
        update.message = MagicMock()
        update.message.text = "https://www.youtube.com/watch?v=test2"
        update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        update.message.reply_voice = AsyncMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 42
        update.effective_chat = MagicMock()

        context = MagicMock()
        context.bot_data = {"memory_mgr": mgr}
        context.bot = MagicMock()
        context.bot.get_chat_member = AsyncMock()

        video2_claims = [
            Claim("AI agents need persistent memory", "AI agents", "need", "persistent memory",
                  confidence=0.9, video_id="test2"),
            Claim("Hermes uses file-based memory", "Hermes", "uses", "file-based memory",
                  confidence=0.85, video_id="test2"),
        ]
        fact_result = FactCheckResult(
            new_claims=[ClassifiedClaim(video2_claims[1], ClaimStatus.NEW)],
            supported_claims=[ClassifiedClaim(video2_claims[0], ClaimStatus.SUPPORTED,
                                              matching_memory="AI agents need structured memory")],
            contradicted_claims=[],
            context_summary="",
        )

        with patch("src.bot.get_config", return_value=pipeline_config), \
             patch("src.bot.extract_video_id", return_value="test2"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="transcript2", language_code="en")), \
             patch("src.bot.extract_claims", new_callable=AsyncMock, return_value=video2_claims), \
             patch("src.bot.classify_claims", new_callable=AsyncMock, return_value=fact_result) as mock_classify, \
             patch("src.bot.build_context_prompt", return_value="PRIOR KNOWLEDGE:\n- AI agents need memory\n\nNEW:\n- Hermes uses files"), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Summary V2") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(update, context)

        # Verify: memory was searched
        mgr.search.assert_awaited_once()

        # Verify: classify was called with both claims AND prior memories
        classify_args = mock_classify.call_args
        assert len(classify_args[0][0]) == 2  # 2 claims
        assert len(classify_args[0][1]) == 1  # 1 prior memory

        # Verify: summary includes prior knowledge context
        ctx = mock_sum.call_args[1]["past_context"]
        assert "PRIOR KNOWLEDGE" in ctx
        assert "Hermes" in ctx

        # Verify: only NEW claims stored (not supported ones)
        assert mgr.add.await_count == 1  # Only "Hermes uses file-based memory"
        stored_text = mgr.add.call_args_list[0][0][0]
        assert "Hermes" in stored_text


class TestE2EPipelineGracefulDegradation:
    """Pipeline works even when memory system fails."""

    @pytest.mark.asyncio
    async def test_memory_crash_still_produces_summary(self, mock_status_msg, pipeline_config):
        mgr = MagicMock()
        mgr.search = AsyncMock(side_effect=Exception("Qdrant down"))

        update = MagicMock()
        update.message = MagicMock()
        update.message.text = "https://www.youtube.com/watch?v=test3"
        update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        update.message.reply_voice = AsyncMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 42
        update.effective_chat = MagicMock()

        context = MagicMock()
        context.bot_data = {"memory_mgr": mgr}
        context.bot = MagicMock()
        context.bot.get_chat_member = AsyncMock()

        with patch("src.bot.get_config", return_value=pipeline_config), \
             patch("src.bot.extract_video_id", return_value="test3"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="transcript", language_code="en")), \
             patch("src.bot.extract_claims", new_callable=AsyncMock,
                   side_effect=Exception("Claude API down")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   return_value="Fallback summary") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(update, context)

        # Summary still generated without context
        mock_sum.assert_awaited_once()
        assert mock_sum.call_args[1].get("past_context") is None

        # User still gets the response
        calls = [str(c) for c in update.message.reply_text.call_args_list]
        assert any("Fallback summary" in c for c in calls)

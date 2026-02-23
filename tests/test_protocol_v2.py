"""Tests for streaming protocol v2 event types and helpers."""
from __future__ import annotations

import base64
import asyncio
import pytest

from app.ws.protocol_v2 import (
    EventType,
    TurnToken,
    make_audio_chunk,
    make_chat_partial,
    make_turn_interrupt,
    make_viseme_chunk,
    StreamingTurnOrchestrator,
    _is_sentence_boundary,
    _detect_mime,
    _estimate_duration_ms,
)


# ---------------------------------------------------------------------------
# Event builder tests
# ---------------------------------------------------------------------------

class TestEventBuilders:
    def test_chat_partial_shape(self):
        evt = make_chat_partial("hello", seq=1)
        assert evt["type"] == EventType.CHAT_PARTIAL
        assert evt["fragment"] == "hello"
        assert evt["seq"] == 1

    def test_audio_chunk_shape(self):
        data = b"\xff\xfb\x00\x00"  # fake MP3 bytes
        evt = make_audio_chunk(data, seq=2, mime="audio/mpeg", is_last=True)
        assert evt["type"] == EventType.AUDIO_CHUNK
        assert evt["seq"] == 2
        assert evt["mime"] == "audio/mpeg"
        assert evt["is_last"] is True
        # base64 round-trip
        assert base64.b64decode(evt["chunk"]) == data

    def test_audio_chunk_not_last_default(self):
        evt = make_audio_chunk(b"\x00", seq=0)
        assert evt["is_last"] is False

    def test_viseme_chunk_shape(self):
        tl = [{"time_ms": 0, "viseme": "open", "mouth_open": 0.5, "mouth_form": 0.5}]
        evt = make_viseme_chunk(tl, seq=3, time_offset_ms=750)
        assert evt["type"] == EventType.VISEME_CHUNK
        assert evt["seq"] == 3
        assert evt["time_offset_ms"] == 750
        assert evt["timeline"] == tl

    def test_turn_interrupt_shape(self):
        evt = make_turn_interrupt("user_barge_in")
        assert evt["type"] == EventType.TURN_INTERRUPT
        assert evt["reason"] == "user_barge_in"

    def test_turn_interrupt_default_reason(self):
        evt = make_turn_interrupt()
        assert evt["reason"] == "user_barge_in"


# ---------------------------------------------------------------------------
# TurnToken tests
# ---------------------------------------------------------------------------

class TestTurnToken:
    def test_initial_state(self):
        tok = TurnToken()
        assert tok.cancelled is False
        assert tok.seq == 0

    def test_cancel(self):
        tok = TurnToken()
        tok.cancel()
        assert tok.cancelled is True

    def test_next_seq_increments(self):
        tok = TurnToken()
        assert tok.next_seq() == 1
        assert tok.next_seq() == 2
        assert tok.next_seq() == 3
        assert tok.seq == 3


# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------

class TestSentenceBoundary:
    @pytest.mark.parametrize("text,expected", [
        ("Hello world.", True),
        ("Is this right?", True),
        ("Yes!", True),
        ("你好吗。", True),
        ("这是对的！", True),
        ("Hello world", False),
        ("", False),
        ("   ", False),
        ("mid-sentence and", False),
    ])
    def test_boundary(self, text, expected):
        assert _is_sentence_boundary(text) == expected


# ---------------------------------------------------------------------------
# MIME detection helper
# ---------------------------------------------------------------------------

class TestDetectMime:
    def test_mp3_id3(self):
        assert _detect_mime(b"ID3\x04" + b"\x00" * 8) == "audio/mpeg"

    def test_mp3_sync(self):
        assert _detect_mime(b"\xff\xfb\x00\x00") == "audio/mpeg"

    def test_wav(self):
        assert _detect_mime(b"RIFF\x00\x00\x00\x00WAVE") == "audio/wav"

    def test_ogg(self):
        assert _detect_mime(b"OggS\x00\x00") == "audio/ogg"

    def test_empty(self):
        assert _detect_mime(b"") == "audio/wav"

    def test_unknown(self):
        assert _detect_mime(b"\xde\xad\xbe\xef") == "application/octet-stream"


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------

class TestEstimateDuration:
    def test_mp3(self):
        # ~128kbps → 128 bits per ms → 16 bytes per ms → 16000 bytes = 1000ms
        ms = _estimate_duration_ms(b"\x00" * 16000, "audio/mpeg")
        assert ms == pytest.approx(1000, rel=0.1)

    def test_wav(self):
        # 44 header + 22050*2 data = 1s of audio at 22050Hz mono 16-bit
        data = b"\x00" * 44 + b"\x00" * (22050 * 2)
        ms = _estimate_duration_ms(data, "audio/wav")
        assert ms == pytest.approx(1000, rel=0.1)

    def test_empty(self):
        assert _estimate_duration_ms(b"", "audio/mpeg") == 0


# ---------------------------------------------------------------------------
# StreamingTurnOrchestrator smoke test
# ---------------------------------------------------------------------------

class TestStreamingOrchestrator:
    """Smoke tests with a simple async token generator and fake TTS/emotion."""

    @staticmethod
    def _fake_tts(text: str):
        import math, io, wave
        n = int(22050 * 0.2)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            frames = bytearray()
            for i in range(n):
                amp = int(4000 * math.sin(2 * math.pi * 220 * i / 22050))
                frames.extend(int(amp).to_bytes(2, "little", signed=True))
            w.writeframes(bytes(frames))
        # Fake phonemes
        return buf.getvalue(), [{"phoneme": "AH", "start_ms": 0, "end_ms": 100}]

    @staticmethod
    def _fake_emotion(text):
        class E:
            category = "neutral"
            intensity = 0.5
            expression = {}
            transition_ms = 300
        return E()

    @pytest.mark.asyncio
    async def test_orchestrator_assembles_text(self):
        sent_events = []

        async def send_json(evt):
            sent_events.append(evt)

        async def token_stream():
            for w in ["Hello", " ", "world", ".", " ", "How", " ", "are", " ", "you", "?"]:
                yield w

        orch = StreamingTurnOrchestrator(
            send_json=send_json,
            tts_synthesize=self._fake_tts,
            emotion_analyze=self._fake_emotion,
        )
        token = TurnToken()
        result = await orch.run(token_stream(), token)

        assert "Hello" in result
        assert "world" in result
        # partial events fired
        partial_types = [e["type"] for e in sent_events]
        assert EventType.CHAT_PARTIAL in partial_types
        # audio chunks fired
        assert EventType.AUDIO_CHUNK in partial_types
        # viseme chunks fired
        assert EventType.VISEME_CHUNK in partial_types

    @pytest.mark.asyncio
    async def test_orchestrator_interrupt(self):
        sent_events = []

        async def send_json(evt):
            sent_events.append(evt)

        async def token_stream():
            for w in ["This", " ", "will", " ", "get"]:
                yield w

        orch = StreamingTurnOrchestrator(
            send_json=send_json,
            tts_synthesize=self._fake_tts,
            emotion_analyze=self._fake_emotion,
        )
        token = TurnToken()
        token.cancel()  # pre-cancel
        await orch.run(token_stream(), token)

        types = [e["type"] for e in sent_events]
        assert EventType.TURN_INTERRUPT in types

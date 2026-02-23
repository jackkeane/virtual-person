"""
WebSocket Streaming Protocol v2 — Skeleton for Phase 4

Protocol overview
-----------------
The v1 protocol sends responses as complete blobs (full chat text, full audio, full viseme
timeline) after the LLM and TTS finish.  This causes 5–10s delays before the user sees or
hears anything.

Protocol v2 adds streaming events to cut perceived latency to <1s:

Server → Client events (new in v2)
====================================
1. chat_partial   — LLM token stream: intermediate text fragments before the full response
2. audio_chunk    — Streaming TTS audio (one chunk per sentence / sub-sentence)
3. viseme_chunk   — Viseme parameters for the corresponding audio chunk
4. turn_interrupt — Server cancels current speaking turn (e.g. user barged in)

Client → Server events (new in v2)
====================================
1. interrupt      — Client requests cancellation of in-flight turn

Backwards compatibility
-----------------------
v2 events are additive.  Clients that don't understand the new event types can safely ignore
them and still receive the complete chat/audio/viseme events at the end.

Implementation status
---------------------
This file defines the data contracts and a skeleton StreamingTurnOrchestrator.
Full streaming integration (vLLM stream=True, sentence-splitting TTS, WS binary audio) is
planned for Phase 4 W2 and W3.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Awaitable


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

class EventType:
    # v1 (existing)
    STATE = "state"
    EXPRESSION = "expression"
    CHAT = "chat"
    AUDIO = "audio"
    VISEME = "viseme"
    TRANSCRIPT = "transcript"

    # v2 (streaming additions)
    CHAT_PARTIAL = "chat_partial"    # partial LLM response fragment
    AUDIO_CHUNK = "audio_chunk"      # one TTS audio chunk (bytes or base64)
    VISEME_CHUNK = "viseme_chunk"    # viseme window for the above audio chunk
    TURN_INTERRUPT = "turn_interrupt"  # cancellation signal


# ---------------------------------------------------------------------------
# Typed event builders
# ---------------------------------------------------------------------------

def make_chat_partial(text_fragment: str, seq: int) -> dict:
    """
    Incremental LLM text fragment (one or more tokens).

    seq: monotonically increasing sequence number for ordering / dedup.
    """
    return {
        "type": EventType.CHAT_PARTIAL,
        "seq": seq,
        "fragment": text_fragment,
    }


def make_audio_chunk(
    audio_bytes: bytes,
    seq: int,
    mime: str = "audio/mpeg",
    is_last: bool = False,
) -> dict:
    """
    One TTS audio chunk encoded as base64.

    seq:     monotonically increasing per turn.
    is_last: True for the final chunk of this turn.
    """
    return {
        "type": EventType.AUDIO_CHUNK,
        "seq": seq,
        "mime": mime,
        "is_last": is_last,
        "chunk": base64.b64encode(audio_bytes).decode("utf-8"),
    }


def make_viseme_chunk(
    viseme_timeline: list[dict],
    seq: int,
    time_offset_ms: int = 0,
) -> dict:
    """
    Viseme window corresponding to the audio_chunk with the same seq.

    time_offset_ms: absolute playback offset from turn start in ms.
    The client adds this to each frame's time_ms for global scheduling.
    """
    return {
        "type": EventType.VISEME_CHUNK,
        "seq": seq,
        "time_offset_ms": time_offset_ms,
        "timeline": viseme_timeline,
    }


def make_turn_interrupt(reason: str = "user_barge_in") -> dict:
    """
    Signal that the current speaking turn is cancelled.

    reason: human-readable cause ('user_barge_in', 'timeout', 'error').
    Client should stop audio playback and flush viseme queue.
    """
    return {
        "type": EventType.TURN_INTERRUPT,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Interruption token — shared across async tasks for cancellation
# ---------------------------------------------------------------------------

@dataclass
class TurnToken:
    """Cancellation token for a single conversation turn."""
    cancelled: bool = False
    seq: int = 0

    def cancel(self) -> None:
        self.cancelled = True

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq


# ---------------------------------------------------------------------------
# StreamingTurnOrchestrator — skeleton for Phase 4
# ---------------------------------------------------------------------------

class StreamingTurnOrchestrator:
    """
    Orchestrates a streaming turn:
      1. Streams LLM tokens → chat_partial events
      2. Buffers tokens into sentences → streaming TTS synthesis
      3. Sends audio_chunk + viseme_chunk per sentence
      4. Handles turn_interrupt if user barges in

    Phase 4 implementation notes
    ----------------------------
    - Requires vLLM `stream=True` (already supported by OpenAI-compat API).
    - TTS must support sentence-level synthesis (gTTS: per-request; FishAudio: native streaming).
    - Audio is sent as binary WebSocket frames in Phase 4 (not base64 JSON) for bandwidth.
    - This skeleton uses async generators as the interface contract.

    Current status: SKELETON — not yet wired into the WS handler.
    Replace StreamingTurnOrchestrator calls in handler.py once vLLM stream is confirmed.
    """

    def __init__(
        self,
        send_json: Callable[[dict], Awaitable[None]],
        tts_synthesize: Callable[[str], tuple[bytes, list[dict]]],
        emotion_analyze: Callable[[str], object],
    ) -> None:
        self._send = send_json
        self._tts = tts_synthesize
        self._emotion = emotion_analyze

    async def run(
        self,
        token_stream: AsyncIterator[str],
        token: TurnToken,
    ) -> str:
        """
        Consume an async token stream from the LLM.

        Sends:
          - chat_partial per token batch
          - audio_chunk + viseme_chunk per completed sentence
          - turn_interrupt if token is cancelled mid-turn

        Returns the complete assembled response text.
        """
        buffer = ""
        full_text = ""
        audio_time_offset_ms = 0

        try:
            async for fragment in token_stream:
                if token.cancelled:
                    await self._send(make_turn_interrupt("user_barge_in"))
                    break

                buffer += fragment
                full_text += fragment
                seq = token.next_seq()
                await self._send(make_chat_partial(fragment, seq))

                # Flush on sentence boundary
                if _is_sentence_boundary(buffer):
                    sentence = buffer.strip()
                    buffer = ""
                    if sentence:
                        audio_time_offset_ms = await self._synthesize_and_send(
                            sentence, seq, audio_time_offset_ms, token
                        )

            # Flush remainder
            if buffer.strip() and not token.cancelled:
                seq = token.next_seq()
                await self._synthesize_and_send(buffer.strip(), seq, audio_time_offset_ms, token, is_last=True)

        except asyncio.CancelledError:
            await self._send(make_turn_interrupt("cancelled"))
            raise

        return full_text

    async def _synthesize_and_send(
        self,
        text: str,
        seq: int,
        time_offset_ms: int,
        token: TurnToken,
        is_last: bool = False,
    ) -> int:
        """Run TTS in executor and send audio_chunk + viseme_chunk. Returns updated time offset."""
        if token.cancelled:
            return time_offset_ms

        loop = asyncio.get_event_loop()
        try:
            audio_bytes, phonemes = await loop.run_in_executor(None, self._tts, text)
        except Exception:
            return time_offset_ms

        from app.voice.lipsync import build_viseme_timeline
        visemes = build_viseme_timeline(phonemes)
        mime = _detect_mime(audio_bytes)

        chunk_duration_ms = _estimate_duration_ms(audio_bytes, mime)

        await self._send(make_audio_chunk(audio_bytes, seq, mime=mime, is_last=is_last))
        await self._send(make_viseme_chunk(visemes, seq, time_offset_ms=time_offset_ms))

        return time_offset_ms + chunk_duration_ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_ENDS = {".", "!", "?", "。", "！", "？"}


def _is_sentence_boundary(text: str) -> bool:
    """Check if text ends at a sentence boundary (including CJK punctuation)."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_ENDS


def _detect_mime(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return "audio/wav"
    magic = audio_bytes[:4]
    if magic[:3] == b"ID3" or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if magic == b"RIFF":
        return "audio/wav"
    if magic == b"OggS":
        return "audio/ogg"
    return "application/octet-stream"


def _estimate_duration_ms(audio_bytes: bytes, mime: str) -> int:
    """Rough duration estimate for sequencing. Not frame-accurate."""
    if not audio_bytes:
        return 0
    if mime == "audio/mpeg":
        # ~128kbps average for gTTS MP3
        return int(len(audio_bytes) * 8 / 128)
    if mime == "audio/wav":
        # PCM 16-bit mono 22050Hz → 2 bytes per sample, 22050 samples/s
        header = 44  # WAV header size
        data_bytes = max(0, len(audio_bytes) - header)
        return int(data_bytes / 2 / 22050 * 1000)
    return int(len(audio_bytes) / 32)  # fallback estimate

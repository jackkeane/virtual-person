from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from app.avatar.emotion_mapper import EmotionMapper
from app.avatar.state_machine import ConversationStateMachine
from app.config import config
from app.llm.streaming import SentenceChunker
from app.observability.metrics import (
    inc_rate_limited,
    inc_turn,
    observe_stt,
    observe_ttfa,
    observe_tts,
    observe_vad,
)
from app.voice.lipsync import build_viseme_timeline
from app.voice.stt_service import BaseSTTService
from app.voice.tts_service import BaseTTSService, FallbackTTSService, FillerAudioService
from app.voice.vad_service import BaseVADService, SileroVADService

_executor = ThreadPoolExecutor(max_workers=4)
POST_TTS_HOLDOFF_MS = 1500


def _detect_audio_mime(audio_bytes: bytes) -> str:
    """Detect audio MIME type from magic bytes."""
    if not audio_bytes:
        return "audio/wav"
    magic = audio_bytes[:4]
    if magic[:3] == b"ID3" or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if magic == b"RIFF":
        return "audio/wav"
    if magic == b"OggS":
        return "audio/ogg"
    if magic[0:2] == b"\x1aE" or magic == b"\x1aE\xdf\xa3":
        return "audio/webm"
    return "application/octet-stream"


class AvatarWebSocketHandler:
    def __init__(
        self,
        state_machine: ConversationStateMachine,
        emotion_mapper: EmotionMapper,
        tts_service: BaseTTSService,
        stt_service: BaseSTTService,
        chat_func: Callable[[str, str], Awaitable[dict]] | Callable[[str, str], dict],
        stream_chat_func: Callable[[str, str], Iterable[str]] | None = None,
        vad_service: BaseVADService | None = None,
        filler_service: FillerAudioService | None = None,
        rate_limiter: object | None = None,
    ) -> None:
        self.state_machine = state_machine
        self.emotion_mapper = emotion_mapper
        self.tts_service = tts_service
        self.stt_service = stt_service
        self.chat_func = chat_func
        self.stream_chat_func = stream_chat_func
        self.vad_service = vad_service or SileroVADService()
        self.filler_service = filler_service
        # Optional per-user rate limiter (duck-typed: .allow(user_id)->(bool,float)).
        # Existing handler tests construct WITHOUT it -> no-op. main.py wires it.
        self.rate_limiter = rate_limiter

        # Guard against overlapping turns and explicitly reset between turns.
        self._turn_lock = asyncio.Lock()
        self._turn_in_progress = False

        # Echo contamination guard: ignore user audio immediately after TTS playback.
        self._last_audio_send_ms = 0.0
        self._last_response_text = ""

    async def _safe_send_state(self, websocket: WebSocket, target_state: str) -> None:
        """Best-effort state transition; never crash websocket on illegal transitions."""
        snapshot = self.state_machine.transition(target_state)
        await websocket.send_json({"type": "state", **snapshot.__dict__})

    async def _send_audio_response(self, websocket: WebSocket, audio_bytes: bytes, timeline: list[dict]) -> None:
        """Send TTS audio and record timestamp for post-TTS holdoff."""
        audio_mime = _detect_audio_mime(audio_bytes)
        await websocket.send_json({
            "type": "audio",
            "format": "base64",
            "mime": audio_mime,
            "chunk": base64.b64encode(audio_bytes).decode("utf-8"),
            "timeline": timeline,
        })
        self._last_audio_send_ms = time.monotonic() * 1000

    def _echo_overlap(self, text: str) -> float:
        if not self._last_response_text or not text:
            return 0.0
        words = set(text.lower().split())
        resp_words = set(self._last_response_text.lower().split())
        if not words:
            return 0.0
        return len(words & resp_words) / len(words)

    def _is_echo(self, text: str) -> bool:
        if not self._last_response_text or not text:
            return False
        overlap = self._echo_overlap(text)
        return overlap > 0.6

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "state", **self.state_machine.snapshot().__dict__})

        try:
            while True:
                raw = await websocket.receive_text()
                payload = json.loads(raw)
                msg_type = payload.get("type")

                if msg_type == "chat":
                    await self._process_text_chat(websocket, payload.get("text", ""), payload.get("user_id", "ws_user"))
                elif msg_type == "audio":
                    now_ms = time.monotonic() * 1000
                    elapsed_since_tts = now_ms - self._last_audio_send_ms
                    if self._last_audio_send_ms > 0 and elapsed_since_tts < POST_TTS_HOLDOFF_MS:
                        print(f"[WS] Skipping audio: post-TTS holdoff ({int(elapsed_since_tts)}ms < {POST_TTS_HOLDOFF_MS}ms)")
                        await websocket.send_json({"type": "transcript", "text": "", "error": "Could not understand audio"})
                        await self._safe_send_state(websocket, "idle")
                        continue

                    await self._safe_send_state(websocket, "listening")
                    await websocket.send_json({"type": "status", "text": "Processing..."})
                    audio_chunk = base64.b64decode(payload.get("chunk", "")) if payload.get("chunk") else b""
                    loop = asyncio.get_event_loop()
                    pipeline_start = time.monotonic()

                    vad_start = time.monotonic()
                    segments = await loop.run_in_executor(_executor, self.vad_service.detect, audio_chunk, 16000)
                    vad_ms = int((time.monotonic() - vad_start) * 1000)
                    observe_vad(vad_ms / 1000)
                    total_speech_ms = sum(seg.duration_ms for seg in segments)
                    has_speech = bool(segments)
                    print(f"[WS] VAD: speech={has_speech}, segments={len(segments)}, duration_ms={total_speech_ms}")

                    if not has_speech:
                        total_ms = int((time.monotonic() - pipeline_start) * 1000)
                        print(f"[WS] Audio pipeline: vad_ms={vad_ms}, stt_ms=0, total_ms={total_ms}")
                        await websocket.send_json({"type": "status", "text": ""})
                        await websocket.send_json({"type": "transcript", "text": "", "error": "Could not understand audio"})
                        await self._safe_send_state(websocket, "idle")
                        continue

                    stt_start = time.monotonic()
                    stt_language = str(payload.get("stt_language", "auto") or "auto")
                    text = await loop.run_in_executor(_executor, self.stt_service.transcribe, audio_chunk, stt_language)
                    stt_ms = int((time.monotonic() - stt_start) * 1000)
                    observe_stt(stt_ms / 1000)
                    total_ms = int((time.monotonic() - pipeline_start) * 1000)
                    print(f"[WS] Audio pipeline: vad_ms={vad_ms}, stt_ms={stt_ms}, total_ms={total_ms}")
                    await websocket.send_json({"type": "status", "text": ""})

                    if text and not text.startswith("[stt"):
                        if self._is_echo(text):
                            overlap = self._echo_overlap(text)
                            print(f"[WS] Echo detected: \"{text}\" (overlap={overlap:.0%})")
                            await websocket.send_json({"type": "transcript", "text": "", "error": "Could not understand audio"})
                            await self._safe_send_state(websocket, "idle")
                            continue

                        # Send transcribed text back to client so user sees what was heard
                        await websocket.send_json({"type": "transcript", "text": text})
                        await self._process_text_chat(websocket, text, payload.get("user_id", "ws_user"))
                    else:
                        # STT failed or empty — go back to idle
                        await websocket.send_json({"type": "transcript", "text": "", "error": "Could not understand audio"})
                        await self._safe_send_state(websocket, "idle")
                elif msg_type == "mic_start":
                    await self._safe_send_state(websocket, "listening")
                elif msg_type == "mic_stop":
                    await self._safe_send_state(websocket, "thinking")
        except WebSocketDisconnect:
            return

    async def _process_text_chat(self, websocket: WebSocket, text: str, user_id: str) -> None:
        # --- Per-user rate limiting ---
        # Inert unless a limiter is wired (main.py) AND enabled. The blocking
        # Redis call is offloaded to the threadpool so the event loop is never
        # blocked; without Redis the limiter fails open (allow) instantly.
        if self.rate_limiter is not None and config.rate_limit_enabled:
            loop = asyncio.get_event_loop()
            allowed, _retry_after = await loop.run_in_executor(
                _executor, self.rate_limiter.allow, user_id
            )
            if not allowed:
                inc_rate_limited()
                await websocket.send_json({
                    "type": "chat",
                    "role": "assistant",
                    "text": "我有点忙，稍等一下再说好吗？",
                    "final": True,
                })
                await self._safe_send_state(websocket, "idle")
                return

        async with self._turn_lock:
            self._turn_in_progress = True
            try:
                turn_start_ms = time.monotonic() * 1000
                await self._safe_send_state(websocket, "thinking")

                if self.stream_chat_func:
                    response = await self._get_streaming_response_with_incremental_tts(websocket, user_id, text, turn_start_ms)
                else:
                    response = await self._get_response_non_stream(websocket, user_id, text)
                    await self._safe_send_state(websocket, "speaking")
                    loop = asyncio.get_event_loop()
                    tts_start = time.monotonic()
                    audio_bytes, phonemes = await loop.run_in_executor(_executor, self.tts_service.synthesize, response)
                    observe_tts(time.monotonic() - tts_start)
                    visemes = build_viseme_timeline(phonemes)
                    await self._send_audio_response(websocket, audio_bytes, visemes)

                emotion = self.emotion_mapper.analyze(response)
                await websocket.send_json({
                    "type": "expression",
                    "category": emotion.category,
                    "intensity": emotion.intensity,
                    "params": emotion.expression,
                    "blend_ms": 400,
                })
                await websocket.send_json({"type": "chat", "role": "assistant", "text": response, "final": True})
                self._last_response_text = response
                await self._safe_send_state(websocket, "idle")
                # One full turn completed (text-chat or post-STT voice turn).
                inc_turn()
            finally:
                # Explicitly reset turn tracking state after each full turn.
                self._turn_in_progress = False

    async def _get_response_non_stream(self, websocket: WebSocket, user_id: str, text: str) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, self.chat_func, user_id, text)
        return (result or {}).get("response", "")

    async def _get_streaming_response_with_incremental_tts(
        self,
        websocket: WebSocket,
        user_id: str,
        text: str,
        turn_start_ms: float,
    ) -> str:
        speed_mode = os.getenv("VOICE_SPEED_MODE", "balanced").lower()
        soft_limit = 20 if speed_mode == "fast" else 24
        sentence_chunker = SentenceChunker(soft_limit_chars=soft_limit)
        response = ""
        iterator = self.stream_chat_func(user_id, text)
        sent_first_audio = False
        sent_filler_audio = False
        first_chunk_done = False
        fallback_tts = FallbackTTSService() if speed_mode == "fast" else None
        thinking_start_ms = time.monotonic() * 1000

        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def tts_worker() -> None:
            nonlocal sent_first_audio, first_chunk_done
            while True:
                item = await tts_queue.get()
                if item is None:
                    return

                # In fast mode: push first chunk ASAP, then merge more for throughput.
                merge_target = 20 if (speed_mode == "fast" and not first_chunk_done) else 90

                parts = [item]
                total_chars = len(item)
                saw_end = False
                while total_chars < merge_target:
                    try:
                        nxt = tts_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nxt is None:
                        saw_end = True
                        break
                    parts.append(nxt)
                    total_chars += len(nxt)

                # Keep Chinese prosody natural while preserving pauses between merged parts.
                merged = []
                for p in parts:
                    p = (p or "").strip()
                    if not p:
                        continue
                    if merged and merged[-1][-1] not in "。！？!?，,；;：:" and p[0] not in "。！？!?，,；;：:":
                        merged.append("。")
                    merged.append(p)
                sentence = "".join(merged)
                if not sentence:
                    if saw_end:
                        return
                    continue

                try:
                    loop = asyncio.get_event_loop()
                    tts_fn = self.tts_service.synthesize
                    if speed_mode == "fast" and not first_chunk_done and fallback_tts is not None:
                        tts_fn = fallback_tts.synthesize
                    tts_start = time.monotonic()
                    audio_bytes, phonemes = await loop.run_in_executor(_executor, tts_fn, sentence)
                    observe_tts(time.monotonic() - tts_start)
                    first_chunk_done = True
                except (ValueError, RuntimeError):
                    # Empty text after cleanup or TTS failure — skip this chunk silently.
                    if saw_end:
                        return
                    continue
                except Exception:
                    if saw_end:
                        return
                    continue

                visemes = build_viseme_timeline(phonemes)
                if not sent_first_audio:
                    sent_first_audio = True
                    ttfa_ms = int((time.monotonic() * 1000) - turn_start_ms)
                    print(f"[WS] TTFA: {ttfa_ms}ms")
                    observe_ttfa(ttfa_ms / 1000)
                    await self._safe_send_state(websocket, "speaking")
                await self._send_audio_response(websocket, audio_bytes, visemes)

                if saw_end:
                    return

        worker_task = asyncio.create_task(tts_worker())

        while True:
            delta = await asyncio.get_event_loop().run_in_executor(_executor, _safe_next, iterator)
            if delta is None:
                break
            response += delta

            if self.filler_service and (not sent_first_audio) and (not sent_filler_audio):
                elapsed_ms = (time.monotonic() * 1000) - thinking_start_ms
                if elapsed_ms >= 1200:
                    print(f"[WS] filler_after_ms={int(elapsed_ms)}")
                    filler_lang = "zh-CN" if any("\u4e00" <= ch <= "\u9fff" for ch in response) else "en"
                    filler_audio, filler_phonemes = self.filler_service.get_filler(filler_lang)
                    filler_visemes = build_viseme_timeline(filler_phonemes)
                    await self._send_audio_response(websocket, filler_audio, filler_visemes)
                    sent_filler_audio = True

            for sentence in sentence_chunker.push(delta):
                await websocket.send_json({"type": "chat_partial", "role": "assistant", "text": sentence, "append": True})
                await tts_queue.put(sentence)

        tail = sentence_chunker.flush()
        if tail:
            await websocket.send_json({"type": "chat_partial", "role": "assistant", "text": tail, "append": True})
            await tts_queue.put(tail)

        await tts_queue.put(None)
        await worker_task

        if not sent_first_audio:
            # fallback: ensure visible speaking state for empty/failed synthesis paths
            await self._safe_send_state(websocket, "speaking")

        return response.strip()


def _safe_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

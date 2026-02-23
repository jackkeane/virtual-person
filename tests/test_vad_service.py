"""Tests for VAD (Voice Activity Detection) service."""
from __future__ import annotations

import io
import math
import wave

import pytest

from app.voice.vad_service import (
    SpeechSegment,
    StubVADService,
    SileroVADService,
    get_vad_service,
    _extract_pcm,
)


def _make_wav(duration_sec: float = 0.5, freq: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Generate a WAV file with a sine wave."""
    n = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            amp = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.extend(int(amp).to_bytes(2, "little", signed=True))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _make_silence_wav(duration_sec: float = 0.3, sample_rate: int = 16000) -> bytes:
    """Generate a WAV file with silence."""
    n = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * n * 2)
    return buf.getvalue()


class TestSpeechSegment:
    def test_duration(self):
        seg = SpeechSegment(start_ms=100, end_ms=850)
        assert seg.duration_ms == 750

    def test_zero_duration(self):
        seg = SpeechSegment(start_ms=0, end_ms=0)
        assert seg.duration_ms == 0


class TestExtractPcm:
    def test_extracts_from_wav(self):
        wav = _make_wav(0.1)
        pcm = _extract_pcm(wav)
        assert len(pcm) > 0
        # PCM should be shorter than WAV (no header)
        assert len(pcm) < len(wav)

    def test_passthrough_for_raw(self):
        raw = b"\x00\x01" * 100
        pcm = _extract_pcm(raw)
        assert pcm == raw

    def test_malformed_wav_returns_empty(self):
        pcm = _extract_pcm(b"RIFF\x00\x00\x00\x00WAVE")
        assert pcm == b""


class TestStubVADService:
    def test_empty_audio(self):
        svc = StubVADService()
        segs = svc.detect(b"")
        assert segs == []

    def test_non_empty_audio_returns_segment(self):
        svc = StubVADService()
        audio = b"\x00" * 1000
        segs = svc.detect(audio)
        assert len(segs) == 1
        assert segs[0].start_ms == 0
        assert segs[0].end_ms > 0

    def test_has_speech_true(self):
        svc = StubVADService()
        assert svc.has_speech(b"\x00" * 100) is True

    def test_has_speech_false_empty(self):
        svc = StubVADService()
        assert svc.has_speech(b"") is False


class TestGetVadService:
    def test_returns_silero_by_default(self):
        svc = get_vad_service(use_silero=True)
        assert isinstance(svc, SileroVADService)

    def test_returns_stub_when_disabled(self):
        svc = get_vad_service(use_silero=False)
        assert isinstance(svc, StubVADService)


class TestSileroVADServiceFallback:
    """Test SileroVADService's fallback behavior when model isn't loaded."""

    def test_detect_empty_returns_empty(self):
        svc = SileroVADService()
        # Force _load_attempted but no model (simulate unavailable)
        svc._load_attempted = True
        svc._model = None
        segs = svc.detect(b"")
        assert segs == []

    def test_detect_fallback_with_no_model(self):
        svc = SileroVADService()
        svc._load_attempted = True
        svc._model = None
        # 1 second of audio at 16000Hz, 16-bit = 32000 bytes
        audio = b"\x01" * 32000
        segs = svc.detect(audio)
        # Fallback: returns one segment covering entire buffer
        assert len(segs) == 1
        assert segs[0].start_ms == 0
        assert segs[0].duration_ms > 0

    def test_has_speech_fallback_false_for_empty(self):
        svc = SileroVADService()
        svc._load_attempted = True
        svc._model = None
        assert svc.has_speech(b"") is False

    def test_has_speech_fallback_true_for_audio(self):
        svc = SileroVADService()
        svc._load_attempted = True
        svc._model = None
        assert svc.has_speech(b"\x00" * 1000) is True

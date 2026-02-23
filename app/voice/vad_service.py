"""
VAD (Voice Activity Detection) service.

Phase 3 foundation: stub + Silero VAD integration.
Silero VAD is installed (silero-vad >= 5.0).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeechSegment:
    """A detected speech segment within an audio buffer."""
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class BaseVADService:
    def detect(self, audio_bytes: bytes, sample_rate: int = 16000) -> list[SpeechSegment]:
        raise NotImplementedError

    def has_speech(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        return bool(self.detect(audio_bytes, sample_rate))


class SileroVADService(BaseVADService):
    """
    Real Silero VAD v5 using the installed silero-vad package.
    Loads model lazily on first call.
    """

    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250) -> None:
        self._model = None
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._load_attempted = False

    def _ensure_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from silero_vad import load_silero_vad  # type: ignore
            import torch  # type: ignore

            self._model = load_silero_vad()
            self._torch = torch
            print("[VAD] Silero VAD model loaded.")
        except Exception as e:
            print(f"[VAD] Failed to load Silero VAD: {e}")
            self._model = None

    def detect(self, audio_bytes: bytes, sample_rate: int = 16000) -> list[SpeechSegment]:
        """
        Detect speech segments in raw PCM16 or WAV audio bytes.
        Returns list of SpeechSegment with start/end in milliseconds.
        """
        self._ensure_model()
        if not audio_bytes:
            return []
        if self._model is None:
            # Fallback: treat entire buffer as speech if non-empty
            duration_ms = int(len(audio_bytes) / (sample_rate * 2) * 1000)
            return [SpeechSegment(0, duration_ms)] if duration_ms > 0 else []

        try:
            import io
            import wave
            import numpy as np  # type: ignore
            import torch  # type: ignore

            # Handle WAV wrapper or raw PCM16
            pcm = _extract_pcm(audio_bytes)
            if not pcm:
                return []

            # Ensure even byte count for int16 parsing
            if len(pcm) % 2 != 0:
                pcm = pcm[:-1]
            if not pcm:
                return []

            audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np)

            # Use Silero VAD get_speech_timestamps utility
            from silero_vad import get_speech_timestamps  # type: ignore

            timestamps = get_speech_timestamps(
                audio_tensor,
                self._model,
                sampling_rate=sample_rate,
                threshold=self._threshold,
                min_speech_duration_ms=self._min_speech_ms,
                return_seconds=False,  # return sample indices
            )

            # Convert sample indices to milliseconds
            segments = [
                SpeechSegment(
                    start_ms=int(ts["start"] / sample_rate * 1000),
                    end_ms=int(ts["end"] / sample_rate * 1000),
                )
                for ts in timestamps
            ]
            return segments
        except Exception as e:
            print(f"[VAD] Detection error: {e}")
            return []


class StubVADService(BaseVADService):
    """Stub VAD: always detects speech if audio is non-empty."""

    def detect(self, audio_bytes: bytes, sample_rate: int = 16000) -> list[SpeechSegment]:
        if not audio_bytes:
            return []
        duration_ms = max(100, len(audio_bytes) // 32)  # rough estimate
        return [SpeechSegment(start_ms=0, end_ms=duration_ms)]


def _extract_pcm(audio_bytes: bytes) -> bytes:
    """Extract raw PCM16 mono 16kHz from WAV, WebM/Opus, Ogg, MP3 or raw PCM.

    Uses ffmpeg for non-WAV containers (WebM, Ogg, MP3) to decode to raw PCM16.
    """
    import io
    import wave

    if not audio_bytes:
        return b""

    magic = audio_bytes[:4]

    # WAV — native parse
    if magic == b"RIFF":
        try:
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception:
            return b""

    # WebM, Ogg, MP3 — decode via ffmpeg stdin/stdout to raw PCM16 mono 16kHz
    if (magic[0:2] == b"\x1aE"      # WebM (EBML)
        or magic == b"OggS"          # Ogg/Opus
        or magic[:3] == b"ID3"       # MP3 ID3
        or (len(magic) >= 2 and magic[0] == 0xFF and (magic[1] & 0xE0) == 0xE0)):  # MP3 sync
        try:
            import subprocess

            result = subprocess.run(
                ["ffmpeg", "-i", "pipe:0",
                 "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"],
                input=audio_bytes,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                print(f"[VAD] ffmpeg decode failed (rc={result.returncode}): {result.stderr[-200:].decode(errors='replace')}")
                return b""
            pcm = result.stdout
            print(f"[VAD] ffmpeg decoded {len(audio_bytes)} bytes -> {len(pcm)} PCM bytes")
            return pcm if pcm else b""
        except Exception as e:
            print(f"[VAD] ffmpeg decode exception: {e}")
            return b""

    # Assume raw PCM16
    return audio_bytes


def get_vad_service(use_silero: bool = True) -> BaseVADService:
    """Factory: return SileroVADService if silero-vad is available, else StubVADService."""
    if use_silero:
        return SileroVADService()
    return StubVADService()

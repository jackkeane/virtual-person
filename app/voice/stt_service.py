from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import wave

from app.config import config


class BaseSTTService:
    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> str:
        raise NotImplementedError


class FasterWhisperSTTService(BaseSTTService):
    """Real faster-whisper STT with optional warmup and language fallback."""

    def __init__(self) -> None:
        self._model = None
        self._load_attempted = False
        self._load_lock = threading.Lock()

        hint = (getattr(config, "stt_language_hint", "auto") or "auto").lower().strip()
        self._language_hint = hint if hint in {"auto", "zh", "en"} else "auto"

        self._device = (config.stt_device or "cpu").lower().strip()
        if self._device not in {"cpu", "cuda"}:
            self._device = "cpu"
        self._compute_type = "int8" if self._device == "cpu" else "float16"

        self._warmup_started = False
        if getattr(config, "stt_warmup_enabled", False):
            self._start_warmup_thread()

    def _normalize_hint(self, language_hint: str | None) -> str:
        hint = (language_hint or self._language_hint or "auto").lower().strip()
        if hint in {"zh", "zh-cn", "chinese"}:
            return "zh"
        if hint in {"en", "english", "en-us"}:
            return "en"
        return "auto"

    def _start_warmup_thread(self) -> None:
        if self._warmup_started:
            return
        self._warmup_started = True

        def _runner() -> None:
            try:
                self._ensure_model()
                if self._model is None:
                    return
                warmup_audio = self._build_silent_wav(duration_ms=120)
                audio_path = self._save_audio(warmup_audio)
                try:
                    self._model.transcribe(audio_path, language="en", beam_size=1, vad_filter=False)
                    print("[STT] Warmup completed")
                finally:
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[STT] Warmup skipped/failed: {e}")

        threading.Thread(target=_runner, name="stt-warmup", daemon=True).start()

    def _ensure_model(self) -> None:
        if self._load_attempted:
            return
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                from faster_whisper import WhisperModel  # type: ignore

                self._model = WhisperModel(
                    config.stt_model_size or "base",
                    device=self._device,
                    compute_type=self._compute_type,
                )
                print(
                    f"[STT] Loaded faster-whisper model={config.stt_model_size} "
                    f"device={self._device} compute_type={self._compute_type}"
                )
            except Exception as e:
                print(f"[STT] Failed to load faster-whisper: {e}")
                self._model = None

    def status(self) -> dict:
        return {
            "loaded": self._model is not None,
            "device": self._device,
            "compute_type": self._compute_type,
            "language_hint": self._language_hint,
            "warmup_started": self._warmup_started,
        }

    def _transcribe_once(self, audio_path: str, *, language: str | None) -> tuple[str, object | None]:
        if self._model is None:
            return "", None
        segments, info = self._model.transcribe(audio_path, language=language, beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, info

    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> str:
        if not audio_bytes:
            return ""
        self._ensure_model()
        if self._model is None:
            return "[stt unavailable]"

        started_at = time.monotonic()
        text = ""
        audio_path = ""
        try:
            audio_path = self._save_audio(audio_bytes)
            hint = self._normalize_hint(language_hint)
            pass1_lang = None if hint == "auto" else hint
            text, info = self._transcribe_once(audio_path, language=pass1_lang)

            lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            needs_fallback = bool(pass1_lang) and (not text or lang_prob < 0.70)
            if needs_fallback:
                text2, _ = self._transcribe_once(audio_path, language=None)
                if text2:
                    text = text2
            return text if text else ""
        except Exception:
            return ""
        finally:
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            print(f"[STT] Transcription took {elapsed_ms}ms, result_len={len(text)}")
            try:
                if audio_path:
                    os.unlink(audio_path)
            except Exception:
                pass

    @staticmethod
    def _build_silent_wav(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
        frame_count = int(sample_rate * duration_ms / 1000)
        pcm = b"\x00\x00" * frame_count
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return buf.getvalue()

    @staticmethod
    def _save_audio(audio_bytes: bytes) -> str:
        magic = audio_bytes[:4]
        if magic == b"RIFF":
            suffix = ".wav"
        elif magic[0:2] == b"\x1a\x45" or magic == b"\x1aE\xdf\xa3":
            suffix = ".webm"
        elif magic == b"OggS":
            suffix = ".ogg"
        elif magic[:3] == b"ID3" or (magic[0] == 0xFF and (magic[1] & 0xE0) == 0xE0):
            suffix = ".mp3"
        else:
            suffix = ".wav"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            with wave.open(tmp, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_bytes)
            return tmp.name
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name


class StubSTTService(BaseSTTService):
    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> str:
        if not audio_bytes:
            return ""
        return "[stt placeholder] transcribed speech"

    def status(self) -> dict:
        return {
            "loaded": False,
            "device": "stub",
            "compute_type": "stub",
            "language_hint": getattr(config, "stt_language_hint", "auto"),
            "warmup_started": False,
        }


def get_stt_service() -> BaseSTTService:
    if config.stt_model_size:
        return FasterWhisperSTTService()
    return StubSTTService()

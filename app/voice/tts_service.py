from __future__ import annotations

import io
import logging
import math
import os
import random
import threading
import wave
from dataclasses import dataclass

from app.config import config
from app.infra.redis_client import redis_available


@dataclass
class PhonemeTimestamp:
    phoneme: str
    start_ms: int
    end_ms: int


class BaseTTSService:
    provider_name: str = "base"

    def synthesize(self, text: str) -> tuple[bytes, list[dict]]:
        raise NotImplementedError


class CosyVoiceTTSService(BaseTTSService):
    """Real CosyVoice TTS via subprocess bridge to cosyvoice conda env."""

    provider_name = "cosyvoice"

    def __init__(self) -> None:
        import subprocess as _sp
        import threading as _threading
        self._subprocess = _sp
        self._worker_python = os.path.expanduser("~/anaconda3/envs/cosyvoice/bin/python")
        self._worker_script = os.path.join(os.path.dirname(__file__), "cosyvoice_worker.py")
        self._proc: _sp.Popen | None = None
        self._warmup_started = False
        # Guards worker spawn + pipe I/O. The stdin/stdout protocol is strictly
        # one-request-one-response, and concurrent first calls (init warmup +
        # filler pre-generate) racing _ensure_worker() each spawn a worker,
        # leaking a GPU-resident model per loser. RLock because synthesize()
        # re-enters via _auto_restart()/_ensure_worker(). Same double-checked
        # locking fix as the pika enqueue path.
        self._lock = _threading.RLock()

        # Pre-warm CosyVoice in background so first user sentence starts faster.
        def _warmup() -> None:
            try:
                self._ensure_worker()
                self.synthesize("你好。")
            except Exception:
                pass

        if config.tts_provider.lower() in {"cosyvoice", "cosy", "phase4", "auto", "chain"}:
            self._warmup_started = True
            _threading.Thread(target=_warmup, daemon=True).start()

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            logging.getLogger("cosyvoice").info("Starting CosyVoice worker process...")
            env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "0"}
            self._proc = self._subprocess.Popen(
                [self._worker_python, self._worker_script],
                stdin=self._subprocess.PIPE,
                stdout=self._subprocess.PIPE,
                stderr=self._subprocess.DEVNULL,
                env=env,
            )

    def health_check(self) -> bool:
        alive = bool(self._proc is not None and self._proc.poll() is None)
        logging.getLogger("cosyvoice").info(f"[TTS] CosyVoice worker health: alive={alive}")
        return alive

    def _auto_restart(self) -> None:
        if self.health_check():
            return
        log = logging.getLogger("cosyvoice")
        log.info("[TTS] CosyVoice worker restarting...")
        self._proc = None

        def _warmup_background() -> None:
            try:
                self._ensure_worker()
                self.synthesize("你好。")
            except Exception:
                pass

        threading.Thread(target=_warmup_background, daemon=True).start()

    def synthesize(self, text: str) -> tuple[bytes, list[dict]]:
        import json as _json
        import time as _time
        import logging

        log = logging.getLogger("cosyvoice")
        try:
            text = _clean_for_tts((text or "").strip())
            if not text:
                raise ValueError("empty text")

            lang = _detect_lang(text)
            if lang.startswith("zh"):
                text = _ensure_zh_sentence_end(text)
            with self._lock:
                self._auto_restart()
                self._ensure_worker()
                assert self._proc and self._proc.stdin and self._proc.stdout

                req = _json.dumps({"text": text, "lang": lang}) + "\n"
                t0 = _time.time()
                self._proc.stdin.write(req.encode())
                self._proc.stdin.flush()

                # Read lines until we get valid JSON (skip tqdm/logging noise)
                raw_line = b""
                while True:
                    raw_line = self._proc.stdout.readline()
                    if not raw_line:
                        self._proc = None
                        raise RuntimeError("CosyVoice worker returned empty response")
                    raw_line = raw_line.strip()
                    if raw_line.startswith(b"{"):
                        break
                    log.debug(f"CosyVoice worker noise: {raw_line[:200]}")

            result = _json.loads(raw_line)
            latency = int((_time.time() - t0) * 1000)
            log.info(f"CosyVoice synthesis: lang={lang} len={len(text)} latency={latency}ms worker_latency={result.get('latency_ms')}ms")

            if not result.get("ok"):
                raise RuntimeError(f"CosyVoice error: {result.get('error')}")

            import base64 as _b64
            audio_bytes = _b64.b64decode(result["audio_b64"])
            duration_ms = result.get("duration_ms", int(len(audio_bytes) / 48))
            phonemes = _estimated_phoneme_timestamps(text, duration_ms)
            return audio_bytes, phonemes
        except Exception as e:
            import sys
            print(f"ERROR in CosyVoice synthesis: {e}", file=sys.stderr)
            raise e


class FishAudioTTSService(BaseTTSService):
    """Phase-4 scaffold. Adapter point for FishAudio runtime integration."""

    provider_name = "fishaudio"

    def synthesize(self, text: str) -> tuple[bytes, list[dict]]:
        raise NotImplementedError("FishAudio adapter scaffold only; runtime integration pending")


class FallbackTTSService(BaseTTSService):
    provider_name = "fallback"

    def synthesize(self, text: str) -> tuple[bytes, list[dict]]:
        text = _clean_for_tts((text or "").strip()) or "..."

        # Attempt gTTS first if available (produces real MP3).
        try:
            from gtts import gTTS  # type: ignore

            lang = _detect_lang(text)
            if str(lang).startswith("zh"):
                text = _ensure_zh_sentence_end(text)
            fp = io.BytesIO()
            gTTS(text=text, lang=lang, slow=False).write_to_fp(fp)
            audio = fp.getvalue()
            # Estimate duration from MP3 size (~16kbps for gTTS)
            est_duration_ms = int(len(audio) / 16 * 8)
            return audio, _estimated_phoneme_timestamps(text, est_duration_ms)
        except Exception:
            pass

        # Portable fallback: generate a tiny WAV tone proportional to text length.
        duration = max(0.4, min(len(text) * 0.05, 5.0)) / max(config.tts_speed, 0.1)
        audio = _sine_wav_bytes(duration_sec=duration)
        return audio, _estimated_phoneme_timestamps(text, int(duration * 1000))


class FillerAudioService:
    """Pre-generate short filler audios for instant 'thinking aloud' feedback.

    Prefer the provided primary TTS service (e.g., CosyVoice chain) so filler voice
    matches normal response voice. Fall back to FallbackTTSService if needed.
    """

    _FILLERS = {
        "zh-CN": ["嗯...", "好的...", "让我想想...", "这个嘛..."],
        "en": ["Hmm...", "Let me think...", "Well..."],
    }

    def __init__(self, primary_tts: BaseTTSService | None = None) -> None:
        self._primary_tts = primary_tts
        self._fallback_tts = FallbackTTSService()
        self._cache: dict[str, list[tuple[bytes, list[dict]]]] = {"zh-CN": [], "en": []}
        self._lock = threading.Lock()
        threading.Thread(target=self._pre_generate, daemon=True).start()

    def _synthesize_filler(self, phrase: str) -> tuple[bytes, list[dict]]:
        if self._primary_tts is not None:
            try:
                return self._primary_tts.synthesize(phrase)
            except Exception:
                pass
        return self._fallback_tts.synthesize(phrase)

    def _pre_generate(self) -> None:
        for lang, phrases in self._FILLERS.items():
            generated: list[tuple[bytes, list[dict]]] = []
            for phrase in phrases:
                try:
                    audio_bytes, phonemes = self._synthesize_filler(phrase)
                    generated.append((audio_bytes, phonemes))
                except Exception:
                    continue
            if generated:
                with self._lock:
                    self._cache[lang] = generated

    def get_filler(self, lang: str = "zh-CN") -> tuple[bytes, list[dict]]:
        lang_key = "zh-CN" if str(lang).startswith("zh") else "en"
        with self._lock:
            options = self._cache.get(lang_key) or []
        if not options:
            phrase = random.choice(self._FILLERS[lang_key])
            return self._synthesize_filler(phrase)
        return random.choice(options)


class ChainedTTSService(BaseTTSService):
    """Try providers in order and fallback gracefully."""

    provider_name = "chain"

    def __init__(self, providers: list[BaseTTSService]) -> None:
        self.providers = providers

    def synthesize(self, text: str) -> tuple[bytes, list[dict]]:
        last_err: Exception | None = None
        for provider in self.providers:
            try:
                return provider.synthesize(text)
            except Exception as exc:
                last_err = exc
                continue
        if last_err:
            raise last_err
        return FallbackTTSService().synthesize(text)


import re as _re

# Matches most emoji, dingbats, symbols, skin-tone modifiers, ZWJ sequences
_EMOJI_RE = _re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # ZWJ
    "\U0001F3FB-\U0001F3FF"  # skin-tone modifiers
    "]+",
    flags=_re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    """Remove emoji and symbol characters so TTS doesn't try to pronounce them."""
    return _EMOJI_RE.sub("", text).strip()


# Markdown / formatting noise that TTS should not try to read aloud.
_MD_PATTERNS = [
    _re.compile(r"^#{1,6}\s+"),           # headers: ### Title
    _re.compile(r"^---+\s*$"),            # horizontal rules
    _re.compile(r"\*\*([^*]+)\*\*"),       # **bold** → content only
    _re.compile(r"\*([^*]+)\*"),           # *italic* → content only
    _re.compile(r"__([^_]+)__"),           # __bold__
    _re.compile(r"_([^_]+)_"),             # _italic_
    _re.compile(r"~~([^~]+)~~"),           # ~~strike~~
    _re.compile(r"`([^`]+)`"),             # `code`
    _re.compile(r"^\s*[-*+]\s+"),          # bullet points
    _re.compile(r"^\s*\d+\.\s+"),          # numbered lists
    _re.compile(r"\[([^\]]+)\]\([^)]+\)"), # [link](url) → text only
]


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS reads clean natural text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip pure separator lines
        if _re.match(r"^---+\s*$", line):
            continue
        # Skip empty lines
        if not line:
            continue
        # Strip header markers
        line = _re.sub(r"^#{1,6}\s+", "", line)
        # Strip bold/italic markers (keep content)
        line = _re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = _re.sub(r"\*([^*]+)\*", r"\1", line)
        line = _re.sub(r"__([^_]+)__", r"\1", line)
        line = _re.sub(r"_([^_]+)_", r"\1", line)
        line = _re.sub(r"~~([^~]+)~~", r"\1", line)
        line = _re.sub(r"`([^`]+)`", r"\1", line)
        # Strip bullet/list markers
        line = _re.sub(r"^\s*[-*+]\s+", "", line)
        line = _re.sub(r"^\s*\d+\.\s+", "", line)
        # Strip links, keep text
        line = _re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return "".join(cleaned)


def _clean_for_tts(text: str) -> str:
    """Full cleanup pipeline: strip emoji + markdown/control symbols."""
    text = _strip_emoji(text)
    text = _strip_markdown(text)

    # Remove leftover markdown-ish control symbols often spoken literally by TTS.
    text = _re.sub(r"[\t\r]+", " ", text)
    text = _re.sub(r"\s+", " ", text)

    # Kill standalone markdown tokens anywhere: -, ---, **, ###, etc.
    text = _re.sub(r"(?<!\w)#{1,6}(?!\w)", " ", text)
    text = _re.sub(r"(?<!\w)\*{1,3}(?!\w)", " ", text)
    text = _re.sub(r"(?<!\w)-{1,4}(?!\w)", " ", text)

    # Bullet/list prefixes that may survive line joins.
    text = _re.sub(r"(?:^|\s)[\-•]\s+(?=\S)", " ", text)

    # General cleanup
    text = _re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _ensure_zh_sentence_end(text: str) -> str:
    """For Chinese TTS chunks, ensure they end with natural full stop punctuation."""
    text = (text or "").strip()
    if not text:
        return text
    if text[-1] in "。！？!?；;：:" :
        return text
    return text + "。"


def _detect_lang(text: str) -> str:
    """Detect if text is primarily Chinese or English for TTS."""
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf')
    total_alpha = sum(1 for ch in text if ch.isalpha())
    if total_alpha == 0:
        return "en"
    # If more than 30% of alphabetic chars are CJK, use Chinese
    if cjk_count / max(total_alpha, 1) > 0.3:
        return "zh-CN"
    return "en"


def _build_tts_service() -> BaseTTSService:
    provider = config.tts_provider.lower()

    if provider in {"fishaudio", "fish"}:
        return ChainedTTSService([FishAudioTTSService(), FallbackTTSService()])

    if provider in {"cosyvoice", "cosy"}:
        return ChainedTTSService([CosyVoiceTTSService(), FishAudioTTSService(), FallbackTTSService()])

    if provider in {"phase4", "auto", "chain"}:
        return ChainedTTSService([CosyVoiceTTSService(), FishAudioTTSService(), FallbackTTSService()])

    return FallbackTTSService()


def get_tts_service() -> BaseTTSService:
    service = _build_tts_service()

    # Redis-backed response cache (Feature 3, worker 5). Inert without the flag
    # or a live Redis client: with REDIS_URL unset this returns the unwrapped
    # service, so behavior is identical to pre-Feature-3. Imported lazily to
    # avoid a circular import (tts_cache imports BaseTTSService from this module).
    if config.tts_cache_enabled and redis_available():
        from app.voice.tts_cache import CachedTTSService

        return CachedTTSService(service)

    return service


_LETTER_TO_PHONEME = {
    'a': ['AE', 'AH', 'AA'], 'b': ['B'], 'c': ['K', 'S'], 'd': ['D'],
    'e': ['EH', 'IY'], 'f': ['F'], 'g': ['G'], 'h': ['HH'],
    'i': ['IH', 'AY'], 'j': ['JH'], 'k': ['K'], 'l': ['L'],
    'm': ['M'], 'n': ['N'], 'o': ['OW', 'AO'], 'p': ['P'],
    'q': ['K'], 'r': ['R'], 's': ['S', 'Z'], 't': ['T'],
    'u': ['UW', 'AH'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'],
    'y': ['Y', 'IY'], 'z': ['Z'],
}


def _estimated_phoneme_timestamps(text: str, total_duration_ms: int) -> list[dict]:
    """Generate estimated phoneme timestamps spread across the audio duration.

    Maps individual letters to plausible ARPAbet phonemes so the viseme pipeline
    produces varied mouth shapes instead of all mapping to 'rest'.
    """
    words = [w for w in text.split() if w.strip()]
    if not words:
        return [{"phoneme": "SIL", "start_ms": 0, "end_ms": 150}]

    # Build a phoneme sequence from letters
    phonemes: list[str] = []
    for word in words:
        for ch in word.lower():
            candidates = _LETTER_TO_PHONEME.get(ch)
            if candidates:
                phonemes.append(candidates[len(phonemes) % len(candidates)])
        phonemes.append("SIL")  # inter-word pause

    if not phonemes:
        return [{"phoneme": "SIL", "start_ms": 0, "end_ms": 150}]

    # Distribute across total duration
    step = max(30, total_duration_ms // len(phonemes))
    out: list[dict] = []
    cursor = 0
    for ph in phonemes:
        end = min(cursor + step, total_duration_ms)
        out.append({"phoneme": ph, "start_ms": cursor, "end_ms": end})
        cursor = end
    return out


def _sine_wav_bytes(duration_sec: float, sample_rate: int = 22050, freq: float = 220.0) -> bytes:
    n_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            amp = int(12000 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.extend(int(amp).to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(frames))
    return buf.getvalue()

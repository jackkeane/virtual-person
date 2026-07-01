import os
from pydantic import BaseModel


def _truthy(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


class AppConfig(BaseModel):
    app_name: str = "Virtual Person Phase1"
    ws_enabled: bool = os.getenv("WS_ENABLED", "true").lower() in ("true", "1", "yes")
    ws_stream_enabled: bool = os.getenv("WS_STREAM_ENABLED", "true").lower() in ("true", "1", "yes")
    quiet_hours_start: int = 23
    quiet_hours_end: int = 8
    proactive_cooldown_minutes: int = 120

    # Runtime LLM provider: ollama | openai_compat
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai_compat")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    llm_strip_thinking: bool = os.getenv("LLM_STRIP_THINKING", "true").lower() in ("true", "1", "yes")

    # Ollama config
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:32b")

    # OpenAI-compatible local server config (vLLM / LM Studio / llama.cpp server)
    openai_compat_base_url: str = os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:8000/v1")
    openai_compat_model: str = os.getenv("OPENAI_COMPAT_MODEL", "Qwen/Qwen3-14B-AWQ")
    openai_compat_api_key: str = os.getenv("OPENAI_COMPAT_API_KEY", "")

    # Editable persona background defaults
    persona_name: str = os.getenv("PERSONA_NAME", "Ani")
    persona_occupation: str = os.getenv("PERSONA_OCCUPATION", "waitress")
    persona_age: int = int(os.getenv("PERSONA_AGE", "30"))
    persona_backstory: str = os.getenv(
        "PERSONA_BACKSTORY",
        "Ani is an AI companion persona with a simple character setup: she presents herself as a 30-year-old waitress who is calm, observant, and good at listening.",
    )
    persona_profile_path: str = os.getenv("PERSONA_PROFILE_PATH", "./persona.json")

    # Voice/avatar config
    tts_provider: str = os.getenv("TTS_PROVIDER", "fallback")
    tts_model_path: str = os.getenv("TTS_MODEL_PATH", "")
    tts_speed: float = float(os.getenv("TTS_SPEED", "1.0"))

    stt_model_size: str = os.getenv("STT_MODEL_SIZE", "small")
    # Legacy STT_LANGUAGE is still accepted as fallback; new preferred key is STT_LANGUAGE_HINT.
    stt_language_hint: str = os.getenv("STT_LANGUAGE_HINT", os.getenv("STT_LANGUAGE", "zh"))
    stt_device: str = os.getenv("STT_DEVICE", "cpu")
    stt_warmup_enabled: bool = os.getenv("STT_WARMUP_ENABLED", "true").lower() in ("true", "1", "yes")

    avatar_model_path: str = os.getenv("AVATAR_MODEL_PATH", "./client/assets/models")
    avatar_live2d_enabled: bool = os.getenv("AVATAR_LIVE2D_ENABLED", "true").lower() in ("true", "1", "yes")

    # Audit log persistence (JSONL file, empty = in-memory only)
    audit_log_path: str = os.getenv("AUDIT_LOG_PATH", "./audit.jsonl")

    # --- Redis + observability (Feature 3) ---
    # All Redis-backed behavior is gated on redis_url being non-empty AND Redis
    # reachable (see app/infra/redis_client.py). The *_enabled flags default true
    # but are INERT without REDIS_URL: unset URL -> in-memory sessions, no TTS
    # cache, allow-all rate limit (identical to pre-Feature-3 behavior).
    redis_url: str = os.getenv("REDIS_URL", "")
    # Session storage backend: auto | memory | redis ("auto" = redis iff available).
    session_backend: str = os.getenv("SESSION_BACKEND", "auto")
    # TTL (seconds) refreshed on every session write so idle-user keys evaporate
    # instead of accumulating forever for every distinct (attacker-controllable)
    # user_id. Mirrors the bounded growth of the vp:rl:/vp:tts: seams. Default 7d.
    session_ttl: int = int(os.getenv("SESSION_TTL", "604800"))

    metrics_enabled: bool = _truthy("METRICS_ENABLED", "1")

    tts_cache_enabled: bool = _truthy("TTS_CACHE_ENABLED", "1")
    tts_cache_ttl: int = int(os.getenv("TTS_CACHE_TTL", "86400"))

    rate_limit_enabled: bool = _truthy("RATE_LIMIT_ENABLED", "1")
    rate_limit_capacity: int = int(os.getenv("RATE_LIMIT_CAPACITY", "20"))
    rate_limit_refill_per_sec: float = float(os.getenv("RATE_LIMIT_REFILL_PER_SEC", "1.0"))


config = AppConfig()

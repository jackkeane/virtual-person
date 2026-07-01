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
    # Optional bearer token for /metrics. Empty = open (dev default); when set,
    # /metrics requires `Authorization: Bearer <token>` or returns 401.
    metrics_auth_token: str = os.getenv("METRICS_AUTH_TOKEN", "")

    tts_cache_enabled: bool = _truthy("TTS_CACHE_ENABLED", "1")
    tts_cache_ttl: int = int(os.getenv("TTS_CACHE_TTL", "86400"))

    rate_limit_enabled: bool = _truthy("RATE_LIMIT_ENABLED", "1")
    rate_limit_capacity: int = int(os.getenv("RATE_LIMIT_CAPACITY", "20"))
    rate_limit_refill_per_sec: float = float(os.getenv("RATE_LIMIT_REFILL_PER_SEC", "1.0"))

    # --- Semantic memory / embeddings (Phase-1 semantic layer) ---
    # The semantic layer is INERT by default: with semantic_memory_enabled=False the
    # app behaves byte-identically to before (keyword/curation retrieval only). Semantic
    # retrieval activates ONLY when this is true AND an embedder is available; otherwise
    # the existing keyword retrieval stays the default and fallback.
    semantic_memory_enabled: bool = _truthy("SEMANTIC_MEMORY_ENABLED", "0")

    # Embedding provider selects the EmbeddingService backend:
    #   "hash"                  -> deterministic, offline, pure python+numpy (DEFAULT; CI/tests).
    #   "ollama"                -> POST to ollama /api/embeddings (bge-m3, 1024-d); degrades to hash.
    #   "sentence_transformers" -> optional lazy local model; degrades to hash.
    # Any non-hash provider degrades to hash on failure and never raises into the caller.
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")
    # Model name passed to the active provider (ignored by "hash"). bge-m3 for ollama.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    # Target vector dimension. 0 = auto per provider (hash=256, ollama/bge-m3=1024).
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "0"))

    # pgvector store DSN. Deliberately SEPARATE from the main MEMORY_POSTGRES_DSN so the
    # semantic vector store never touches the primary `memory_items` Postgres. Empty by
    # default (in-process numpy cosine index only); set only in CI/pgvector-capable envs.
    pgvector_dsn: str = os.getenv("PGVECTOR_DSN", "")

    # Ollama embedding model (default bge-m3, 1024-d). `embedding_model` above is
    # what get_embedding_service() actually reads to build the embedder; this
    # provider-scoped alias carries the same default so the two never diverge.
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

    # Number of nearest neighbours the (gated) semantic layer pulls from the vector
    # store before blending their cosine similarity into the curation ranking. Kept
    # small so semantic recall augments — never floods — the keyword/curation result.
    semantic_top_k: int = int(os.getenv("SEMANTIC_TOP_K", "5"))

    # --- RabbitMQ background task queue (Feature: async background jobs) ---
    # INERT BY DEFAULT and strictly OFF the realtime voice path. Like the Redis
    # seams above, the queue is a NO-OP unless queue_enabled AND amqp_url is
    # non-empty AND the broker is reachable (see app/queue/task_queue.py). With
    # AMQP_URL unset the app is byte-identical to today: get_task_queue() returns
    # None, enqueue() no-ops, NO broker connection is opened, and pika is never
    # imported. queue_enabled defaults true (mirroring metrics/tts_cache/rate_limit)
    # yet stays dormant because amqp_url is empty by default. The queue carries
    # ONLY background jobs (memory curation, daily-summary precompute, proactive
    # nudges) — never the VAD->STT->LLM->TTS turn. A broker outage must NEVER
    # break a chat or a voice turn.
    queue_enabled: bool = _truthy("QUEUE_ENABLED", "1")
    amqp_url: str = os.getenv("AMQP_URL", "")
    # Durable topology, declared idempotently on connect. A direct exchange routes
    # the work queue by routing key; jobs that exhaust queue_max_retries dead-letter
    # through the DLX into the DLQ so poison messages are quarantined instead of
    # redelivered forever.
    queue_exchange: str = os.getenv("QUEUE_EXCHANGE", "vp.tasks")
    queue_name: str = os.getenv("QUEUE_NAME", "vp.tasks.q")
    queue_routing_key: str = os.getenv("QUEUE_ROUTING_KEY", "vp.task")
    queue_dlx: str = os.getenv("QUEUE_DLX", "vp.tasks.dlx")
    queue_dlq: str = os.getenv("QUEUE_DLQ", "vp.tasks.dlq")
    # Max handler attempts before a message is routed to the DLQ (poison guard).
    queue_max_retries: int = int(os.getenv("QUEUE_MAX_RETRIES", "3"))
    # Consumer QoS: max unacked messages a single worker holds in flight.
    queue_prefetch: int = int(os.getenv("QUEUE_PREFETCH", "10"))


config = AppConfig()

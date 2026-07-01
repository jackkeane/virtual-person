"""Pluggable text embedding service for the semantic memory layer.

The service exposes a single, stable contract to the rest of the app::

    svc = get_embedding_service()
    vec  = svc.embed("hello")          # -> list[float] of length svc.dim
    vecs = svc.embed_batch(["a", "b"])  # -> list[list[float]]

Design goals
------------
* **Offline & deterministic by default.** The default ``"hash"`` provider is a
  pure python + numpy feature-hashing embedder. It needs no network, no models,
  and no GPU, and it is byte-for-byte reproducible across processes (it uses
  :mod:`hashlib`, never the salted builtin ``hash()``). This is what CI/tests use.
* **Never break the caller.** Every non-hash provider (``"ollama"``,
  ``"sentence_transformers"``) degrades to the hash embedder on *any* failure --
  missing dependency, network error, bad response, or dimension mismatch. Embedding
  therefore never raises into audio/memory code paths.
* **Fixed dimension invariant.** ``embed()`` ALWAYS returns exactly ``dim`` floats,
  regardless of provider or fallback, so a vector index can size itself once and
  trust every row.

Only numpy is required at runtime. ``requests`` (ollama) and
``sentence_transformers`` (local model) are imported lazily and are optional.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# Default embedding dimension for the offline hash embedder. Small enough to be
# cheap, large enough to keep hashing-trick collisions rare for short memories.
DEFAULT_HASH_DIM = 256

# Per-provider dimension when config leaves ``embedding_dim`` at 0 ("auto").
_DEFAULT_DIMS = {
    "hash": DEFAULT_HASH_DIM,
    "ollama": 1024,  # bge-m3
    "sentence_transformers": 1024,  # bge-m3 family
}

_DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_OLLAMA_TIMEOUT_SECONDS = 30.0

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _coerce_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return str(text)


def _hash_features(text: str) -> list[str]:
    """Turn text into a deterministic bag of string features.

    Combines Unicode word tokens (good for latin scripts) with character
    n-grams (good for scripts without word boundaries, e.g. Chinese) so the
    embedder behaves reasonably for the multilingual zh+en persona.
    """
    norm = _WS_RE.sub(" ", text.strip().lower())
    if not norm:
        return []

    feats: list[str] = []
    # Word-level tokens: latin words become one token each; a CJK run becomes a
    # single token, which the n-grams below then decompose.
    for tok in _WORD_RE.findall(norm):
        feats.append("w:" + tok)

    # Character n-grams over the space-stripped string capture subword structure
    # and give per-character signal for CJK.
    compact = norm.replace(" ", "")
    for n in (2, 3):
        if len(compact) >= n:
            for i in range(len(compact) - n + 1):
                feats.append(f"c{n}:{compact[i:i + n]}")
    # Unigrams so even single-character inputs produce a non-zero vector.
    for ch in compact:
        feats.append("c1:" + ch)
    return feats


def _hash_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic L2-normalized feature-hashing embedding (float64).

    Uses the signed hashing trick: each feature hashes to an index and a +/-1
    sign, which reduces the systematic bias of collisions. blake2b makes the
    result stable across processes and Python runs (unlike builtin ``hash()``).
    """
    vec = np.zeros(dim, dtype=np.float64)
    for feat in _hash_features(text):
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(digest[:8], "big") % dim
        vec[idx] += 1.0 if (digest[8] & 1) else -1.0

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


class EmbeddingService:
    """A pluggable text embedder with guaranteed graceful degradation.

    Parameters
    ----------
    provider:
        ``"hash"`` (default, offline), ``"ollama"``, or ``"sentence_transformers"``.
        Any unrecognized value behaves as ``"hash"``.
    model:
        Model name for the active provider (ignored by ``"hash"``).
    dim:
        Target vector dimension. Every returned vector has exactly this length;
        a real provider whose output length differs is discarded in favor of the
        hash fallback so the dimension is a hard invariant.
    base_url:
        Base URL of the ollama server (defaults to the standard local endpoint).
    timeout:
        HTTP timeout in seconds for the ollama provider.
    """

    def __init__(
        self,
        provider: str = "hash",
        model: str = "",
        dim: int = DEFAULT_HASH_DIM,
        *,
        base_url: str | None = None,
        timeout: float = _OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self._provider = (provider or "hash").strip().lower()
        self._model = model or ""
        try:
            d = int(dim)
        except (TypeError, ValueError):
            d = DEFAULT_HASH_DIM
        self._dim = d if d > 0 else DEFAULT_HASH_DIM
        self._base_url = (base_url or _DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self._timeout = float(timeout)

        # Lazy sentence-transformers state (never touched unless that provider is used).
        self._st_model = None
        self._st_failed = False

    # -- public contract ---------------------------------------------------

    @property
    def dim(self) -> int:
        """Length of every vector produced by this service."""
        return self._dim

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed one string into a ``dim``-length ``list[float]``.

        Never raises: a non-hash provider that fails (for any reason) or returns
        the wrong dimension transparently falls back to the deterministic hash
        embedding.
        """
        s = _coerce_text(text)

        if self._provider == "ollama":
            vec = self._safe(self._embed_ollama, s)
        elif self._provider == "sentence_transformers":
            vec = self._safe(self._embed_sentence_transformers, s)
        else:
            vec = None

        if vec is not None and len(vec) == self._dim:
            return vec
        return self._embed_hash(s)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings.

        Implemented as per-item :meth:`embed` so the batch result is always
        identical to embedding each item individually (and so a single failing
        item degrades on its own without poisoning the batch).
        """
        return [self.embed(t) for t in texts]

    # -- providers ---------------------------------------------------------

    @staticmethod
    def _safe(fn, s: str) -> list[float] | None:
        """Run a provider function, swallowing *any* error into ``None``."""
        try:
            return fn(s)
        except Exception:
            return None

    def _embed_hash(self, text: str) -> list[float]:
        return _hash_vector(text, self._dim).tolist()

    def _embed_ollama(self, text: str) -> list[float] | None:
        try:
            import requests  # optional dependency; imported lazily
        except Exception:
            return None

        resp = requests.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model or "bge-m3", "prompt": text},
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        emb = payload.get("embedding")
        if not isinstance(emb, list) or not emb:
            return None
        return [float(x) for x in emb]

    def _embed_sentence_transformers(self, text: str) -> list[float] | None:
        model = self._load_sentence_transformer()
        if model is None:
            return None
        vec = model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    def _load_sentence_transformer(self):
        if self._st_model is not None:
            return self._st_model
        if self._st_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            # Not installed -> permanently degrade so we don't retry the import.
            self._st_failed = True
            return None
        try:
            name = self._model or "BAAI/bge-m3"
            # "bge-m3" is the ollama tag; the HF/ST id is namespaced.
            if name == "bge-m3":
                name = "BAAI/bge-m3"
            self._st_model = SentenceTransformer(name)
            return self._st_model
        except Exception:
            self._st_failed = True
            return None


# -- factory ---------------------------------------------------------------

def _resolve_dim(provider: str, dim: int) -> int:
    try:
        d = int(dim)
    except (TypeError, ValueError):
        d = 0
    if d > 0:
        return d
    return _DEFAULT_DIMS.get(provider, DEFAULT_HASH_DIM)


# Cache one service per distinct config signature. This lets the app share a
# single instance (and any lazily-loaded model) while still handing out a fresh
# instance when config -- e.g. the provider -- changes (as tests do).
_service_cache: dict[tuple, EmbeddingService] = {}


def get_embedding_service(config=None) -> EmbeddingService:
    """Build (or reuse) an :class:`EmbeddingService` from ``app.config``.

    Reads ``embedding_provider``, ``embedding_model``, ``embedding_dim`` and
    ``ollama_base_url``. Passing an explicit ``config`` object overrides the
    module-level one (handy for tests).
    """
    if config is None:
        from app.config import config as config  # noqa: PLW0127

    provider = (getattr(config, "embedding_provider", "hash") or "hash").strip().lower()
    model = getattr(config, "embedding_model", "") or ""
    dim = _resolve_dim(provider, getattr(config, "embedding_dim", 0) or 0)
    base_url = getattr(config, "ollama_base_url", _DEFAULT_OLLAMA_BASE_URL) or _DEFAULT_OLLAMA_BASE_URL

    key = (provider, model, dim, base_url)
    svc = _service_cache.get(key)
    if svc is None:
        svc = EmbeddingService(provider=provider, model=model, dim=dim, base_url=base_url)
        _service_cache[key] = svc
    return svc


def reset_embedding_service_cache() -> None:
    """Clear the cached services (used by tests that mutate config)."""
    _service_cache.clear()

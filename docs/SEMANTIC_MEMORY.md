# Semantic Memory

The memory layer retrieves stored facts two ways:

1. **Keyword / curation** (the default, always on) — `MemoryService.search()` does a
   substring match, then `app/memory/curation.py` ranks the hits
   (`retrieval_score` → `rank_memories`). Fast, deterministic, zero dependencies.
   It can only find a memory the query *shares words with*.
2. **Semantic** (opt-in, gated) — embed the query, cosine-rank it against an
   embedding of every memory. This finds **paraphrases**: a query that means the
   same thing but shares no keyword (e.g. store `我今天去健身房了`, ask `锻炼身体`
   or `workout`).

Semantic retrieval **augments** keyword retrieval; it never replaces it. Keyword /
curation is the base and the fallback.

---

## Inert by default

The semantic layer is **off** unless you explicitly turn it on **and** an embedder /
vector store is actually available. With the defaults, the app (and the whole
`pytest` suite) behaves byte-for-byte as it did before this layer existed:

- `semantic_memory_enabled` defaults **`False`** (`_truthy("SEMANTIC_MEMORY_ENABLED", "0")`).
- `embedding_provider` defaults **`hash`** — offline, no network, deterministic.
- `pgvector_dsn` defaults **empty** — no Postgres/pgvector touched; the in-process
  numpy index is used.

So the default test path needs **no network, no Redis, no Postgres/pgvector, no
torch, no ollama**. The bare CI `test` job stays green with the semantic paths
dormant.

### Configuration

All knobs live on `AppConfig` in `app/config.py` and are read from the environment:

| Env var | `config` field | Default | Meaning |
|---|---|---|---|
| `SEMANTIC_MEMORY_ENABLED` | `semantic_memory_enabled` | `0` (False) | Master gate. Semantic retrieval only activates when true **and** an embedder is available. |
| `EMBEDDING_PROVIDER` | `embedding_provider` | `hash` | `hash` (offline) · `ollama` (bge-m3) · `sentence_transformers` (optional local model). |
| `EMBEDDING_MODEL` | `embedding_model` | `bge-m3` | Model name for the active provider (ignored by `hash`). |
| `EMBEDDING_DIM` | `embedding_dim` | `0` (auto) | Vector dimension; `0` = per-provider default (hash=256, bge-m3=1024). |
| `PGVECTOR_DSN` | `pgvector_dsn` | *(empty)* | DSN for the pgvector store. Empty → in-process numpy index only. **Separate from `MEMORY_POSTGRES_DSN`.** |
| `OLLAMA_BASE_URL` | `ollama_base_url` | `http://127.0.0.1:11434` | Ollama endpoint for the `ollama` embedding provider. |

---

## Architecture

### 1. Embedding service — `app/memory/embeddings.py`

One stable contract, obtained via the config-driven factory:

```python
from app.memory.embeddings import get_embedding_service
svc  = get_embedding_service()          # reads app.config
vec  = svc.embed("锻炼身体")             # -> list[float] of length svc.dim
vecs = svc.embed_batch(["a", "b"])       # -> list[list[float]]
```

Providers:

| Provider | Backend | Needs | Notes |
|---|---|---|---|
| `hash` *(default)* | pure python + numpy feature hashing (blake2b, signed hashing trick) | numpy only | Offline, deterministic across processes/runs. Word tokens **+** character n-grams, so it degrades gracefully for CJK. This is what CI/tests use. |
| `ollama` | `POST {OLLAMA_BASE_URL}/api/embeddings` with `{"model": "bge-m3", "prompt": text}` | ollama serving `bge-m3` (1024-d, multilingual zh+en); `requests` | The real embedder for the local demo. |
| `sentence_transformers` | lazy local `SentenceTransformer` model | `sentence-transformers` + torch (heavy) | Optional. **Not** a CI/runtime requirement. |

Two invariants make the vector index safe:

- **Never breaks the caller.** Any non-hash provider that fails for *any* reason
  (missing dep, network error, bad response, wrong dimension) transparently falls
  back to the hash embedding. `embed()` never raises into memory/audio code.
- **Fixed dimension.** `embed()` always returns exactly `svc.dim` floats, so the
  index can size itself once and trust every row — even a fallback row.

`get_embedding_service()` caches one instance per `(provider, model, dim, base_url)`
signature; `reset_embedding_service_cache()` clears it (used by tests that mutate
config).

### 2. Vector store — `app/memory/vector_store.py`

One small interface, two interchangeable backends:

```python
store.upsert(id, text, vector, metadata) -> None
store.query(vector, top_k) -> list[(id, cosine_similarity, metadata)]
```

| Backend | When | Needs | Table |
|---|---|---|---|
| `NumpyVectorIndex` | always available; the local demo + default test path | numpy only | in-process |
| `PgVectorStore` | real pgvector, when `PGVECTOR_DSN` is set and reachable | psycopg (already a dep) + a `vector`-capable Postgres | **`vp_mem_vectors`** |

The pgvector backend:

- Connects **only** through `config.pgvector_dsn` (`PGVECTOR_DSN`) — **never**
  `MEMORY_POSTGRES_DSN`, and **never** the primary `memory_items` table.
- On `connect()` runs idempotent DDL: `CREATE EXTENSION IF NOT EXISTS vector` then
  `CREATE TABLE IF NOT EXISTS vp_mem_vectors (...)`.
- Imports `psycopg` **lazily** inside `connect()` (like `PostgresMemoryStore`), so
  importing the module costs nothing and a plain `pytest` never needs a live DB.
- Queries with pgvector cosine distance (`<=>`) and returns `1 - distance` as the
  cosine similarity, matching `NumpyVectorIndex` semantics.

The factory `get_vector_store(dim, dsn=None)` returns a connected `PgVectorStore`
when pgvector is usable, and **`None`** otherwise (empty DSN, unreachable server, or
missing `vector` extension). It **never raises** — callers use `NumpyVectorIndex`
as the fallback whenever it returns `None`.

### 3. Retrieval composition (keyword + semantic)

The two systems combine as: **keyword/curation first (the base + fallback), semantic
as an additional gated branch.** The gate is deliberately double, mirroring the
existing Redis/Postgres "inert-by-default" pattern:

```
config.semantic_memory_enabled AND embedder available AND vector index available
```

If any part is false, retrieval is exactly the existing keyword/curation path. When
all are true, semantic candidates are merged in and de-duplicated by the same
`(kind, key, value)` key the keyword path uses, so a memory found by both routes
appears once. Semantic similarity is folded into ranking as a small additive term
(comparable in magnitude to the curation recency bonus), so a raw cosine score can
never dominate or saturate the existing `retrieval_score` ceiling.

`eval/semantic_eval.py` and `scripts/demo/semantic_demo.sh` demonstrate this exact
composition against a fixed corpus.

---

## How pgvector is exercised in CI

`.github/workflows/ci.yml` has three test jobs; two are relevant here:

- **`test`** — the bare suite: `pytest -q` with **no** services and **no** env.
  The semantic layer is inert (`SEMANTIC_MEMORY_ENABLED` unset, `PGVECTOR_DSN`
  unset), so this stays byte-identical / green. `tests/test_vector_store.py` still
  runs here, but its pgvector cases `skipif(not PGVECTOR_DSN)` cleanly; only the
  always-on `NumpyVectorIndex` and factory-returns-`None` cases execute.
- **`test-pgvector`** — mirrors the `test-redis` job, but with a real pgvector:

  ```yaml
  services:
    postgres:
      image: pgvector/pgvector:pg16
      env:
        POSTGRES_PASSWORD: postgres
      options: >-
        --health-cmd "pg_isready -U postgres" --health-interval 10s ...
  steps:
    - ...
    - name: Run vector-store tests (pgvector-enabled)
      env:
        PGVECTOR_DSN: postgresql://postgres:postgres@127.0.0.1:5432/postgres
        SEMANTIC_MEMORY_ENABLED: '1'
      run: python -m pytest tests/test_vector_store.py -q
  ```

  With `PGVECTOR_DSN` set, the skipped cases now run against the live container and
  exercise the **real** SQL path end-to-end: `CREATE EXTENSION vector`, the
  `vp_mem_vectors` DDL, `INSERT ... ::vector` upserts, the cosine `<=>` nearest-
  neighbour query, and `ON CONFLICT` replacement. It uses its own DSN and own
  table, so it never touches the primary `memory_items` store.

Locally there is no Docker and the machine's Postgres 14 has no `vector`
extension, so `PGVECTOR_DSN` stays empty and everything falls back to
`NumpyVectorIndex` — pgvector is a CI-only concern.

---

## Run it locally

All Python runs use the project conda env: `~/anaconda3/bin/conda run -n py312`.

### Demo — `scripts/demo/semantic_demo.sh`

Stores a few Chinese persona memories, then queries them with paraphrases and
prints keyword vs. semantic retrieval side by side. It writes to a throwaway
persist file and runs the memory layer fully in-process (no Postgres/Neo4j, never
touches the app's `memory_store.json`).

```bash
# Real semantic matching needs ollama serving bge-m3:
ollama serve
ollama pull bge-m3

cd /home/zz79jk/clawd/virtual-person-phase1
bash scripts/demo/semantic_demo.sh
```

Expected (with ollama up): keyword retrieval finds **0** of the paraphrases while
semantic finds them all — e.g. `锻炼身体` and `workout` both recover
`我今天去健身房了`. If ollama is not running the embedder degrades to `hash` and the
script says so; the cross-lingual matches won't light up until you start it.

Overrides: `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `OLLAMA_BASE_URL` (all have
sensible defaults in the script).

### Eval — `eval/semantic_eval.py`

Recall@k for keyword/curation vs. semantic retrieval over a fixed bilingual corpus
of paraphrase queries (plus a few literal "control" queries that keyword *must*
still hit, proving the harness is sound). Prints a markdown table and writes
`eval/semantic_report.json`. Always exits 0.

```bash
cd /home/zz79jk/clawd/virtual-person-phase1

# Offline / plumbing (deterministic hash embedder, no network):
~/anaconda3/bin/conda run -n py312 python eval/semantic_eval.py

# Real semantic gain (needs ollama + bge-m3):
EMBEDDING_PROVIDER=ollama EMBEDDING_MODEL=bge-m3 \
  ~/anaconda3/bin/conda run -n py312 python eval/semantic_eval.py
```

Interpreting the numbers:

- With **`hash`** (default): the run only **exercises the plumbing**. Keyword
  paraphrase recall is `0.00`; hash paraphrase recall is low and reflects only
  incidental surface-character overlap, not meaning.
- With **`ollama` / `bge-m3`**: the real multilingual embedder. Paraphrase recall
  jumps toward `1.0` while keyword stays at `0.00` — that gap is the semantic gain.

The `control` rows (literal keyword present) should read `1.00` for keyword in both
regimes; if they don't, the harness — not the embedder — is broken.

---

## Safety / isolation summary

- Semantic retrieval is off unless `SEMANTIC_MEMORY_ENABLED=1` **and** an embedder
  is available; otherwise the existing keyword/curation retrieval is unchanged.
- The embedder never raises: any non-hash provider degrades to the offline hash
  embedder, and every vector is exactly `dim` floats.
- The pgvector store uses its own `PGVECTOR_DSN` and its own `vp_mem_vectors`
  table with `CREATE EXTENSION / TABLE IF NOT EXISTS`; it never reads or writes the
  primary `memory_items` store.
- New runtime dependency: `numpy` only. `torch` / `sentence-transformers` / `ollama`
  / a pgvector server are all optional and never required by the default path.

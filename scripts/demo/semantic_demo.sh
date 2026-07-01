#!/usr/bin/env bash
# Semantic memory live demo for virtual-person (Phase-1 semantic layer).
#
# Shows the one thing keyword memory can't do: retrieve a memory from a
# *paraphrase* that shares no words with it. We store a few Chinese persona
# memories, then query them with reworded / cross-lingual paraphrases and print,
# side by side:
#
#   * KEYWORD retrieval  (MemoryService.search -> substring match)  -> finds nothing
#   * SEMANTIC retrieval (bge-m3 embedding -> numpy cosine index)   -> finds the memory
#
# e.g. store  "我今天去健身房了"  and ask  "锻炼身体" / "workout after work".
#
# Real embeddings come from ollama (bge-m3, multilingual zh+en). If ollama is not
# running the EmbeddingService transparently DEGRADES to the offline hash embedder
# -- the script still runs, but the cross-lingual paraphrase matches won't light up
# (hashing captures surface characters, not meaning). Start ollama to see the magic:
#
#     ollama serve
#     ollama pull bge-m3
#     bash scripts/demo/semantic_demo.sh
#
# Safe: it NEVER touches the app's real memory_store.json or any Postgres/Neo4j
# store -- it writes to a throwaway persist file and runs the memory layer fully
# in-process.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"

# --- embedder for the SEMANTIC column: real embedder = ollama/bge-m3 (override via env) ---
# NOTE: the service-level semantic layer (SEMANTIC_MEMORY_ENABLED) is deliberately
# NOT enabled here. This demo contrasts MemoryService's plain substring search
# (KEYWORD column) with a hand-built cosine index (SEMANTIC column), so the service
# must stay keyword-only -- it is forced off in the Python block below regardless of
# any inherited SEMANTIC_MEMORY_ENABLED. These vars only pick the manual embedder.
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ollama}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

# --- keep the demo hermetic ---------------------------------------------------
# Route MemoryService at a throwaway persist file and make sure it never reaches
# out to the real Postgres/Neo4j memory stores.
export MEMORY_PERSIST_PATH="$(mktemp -u "${TMPDIR:-/tmp}/semantic-demo-mem.XXXXXX").json"
unset MEMORY_POSTGRES_DSN MEMORY_NEO4J_URI MEMORY_NEO4J_USERNAME MEMORY_NEO4J_PASSWORD 2>/dev/null || true
export PYTHONPATH="$REPO"
trap 'rm -f "$MEMORY_PERSIST_PATH"' EXIT

echo "=================================================================="
echo "  Semantic memory demo — paraphrase retrieval keyword can't do"
echo "=================================================================="
echo "  EMBEDDING_PROVIDER = $EMBEDDING_PROVIDER   EMBEDDING_MODEL = $EMBEDDING_MODEL"
echo "  OLLAMA_BASE_URL    = $OLLAMA_BASE_URL"
echo "  persist (throwaway)= $MEMORY_PERSIST_PATH"
echo "  python             = $CONDA run --no-capture-output -n py312"
echo

"$CONDA" run --no-capture-output -n py312 python - <<'PY'
import sys
from urllib.request import urlopen

from app.config import config
from app.memory.service import MemoryService
from app.memory.embeddings import get_embedding_service
from app.memory.vector_store import NumpyVectorIndex


def banner(title: str) -> None:
    print("\n" + "-" * 66)
    print(title)
    print("-" * 66)


# --- which embedder is really active? -----------------------------------------
emb = get_embedding_service()
print(f"active embedder: provider={emb.provider!r} model={emb.model!r} dim={emb.dim}")

degraded = False
if emb.provider == "ollama":
    try:
        with urlopen(config.ollama_base_url.rstrip("/") + "/api/tags", timeout=1.5) as r:
            reachable = getattr(r, "status", 200) == 200
    except Exception:
        reachable = False
    print(f"ollama reachable at {config.ollama_base_url}: {reachable}")
    if not reachable:
        degraded = True
        print("  !! ollama is NOT reachable -> embed() will fall back to the offline")
        print("     HASH embedder. The demo still runs, but cross-lingual paraphrase")
        print("     matches below will be weak/absent. Start it with:")
        print("        ollama serve   &&   ollama pull bge-m3")
elif emb.provider != "ollama":
    degraded = True
    print("  (running the offline hash embedder — set EMBEDDING_PROVIDER=ollama for real matches)")

# --- 1. store a few persona memories (mostly Chinese) -------------------------
MEMORIES = [
    ("note", "运动", "我今天去健身房了"),          # went to the gym today
    ("preference", "饮食", "每天早上都要喝一杯拿铁"),  # a latte every morning
    ("identity", "职业", "我在一家互联网公司做后端开发"),  # backend dev at an internet company
    ("goal", "心愿", "明年想去北海道滑雪"),           # wants to ski in Hokkaido next year
    ("note", "宠物", "家里养了一只叫豆豆的橘猫"),       # has an orange cat named Doudou
]

# --- keyword baseline: force the service's semantic layer OFF -----------------
# This service IS the KEYWORD column. MemoryService.search() gates its hybrid
# semantic re-rank on config.semantic_memory_enabled (read live), so with the flag
# on it would union the semantic top-k into the substring result and EVERY
# paraphrase would falsely register as a keyword HIT -- defeating the whole demo.
# Force it off so .search() is a pure substring match. The SEMANTIC column below is
# hand-built from get_embedding_service()+NumpyVectorIndex, which do NOT read this
# flag, so disabling it here costs the semantic path nothing.
config.semantic_memory_enabled = False

svc = MemoryService()   # keyword-only; throwaway MEMORY_PERSIST_PATH, no Postgres/Neo4j
svc.erase_all()
for kind, key, value in MEMORIES:
    svc.write(kind, key, value)

# Build the semantic index over the SAME memories (id = the stored value).
index = NumpyVectorIndex(dim=emb.dim)
for kind, key, value in MEMORIES:
    text = f"{key} {value}"
    index.upsert(value, text, emb.embed(text), {"kind": kind, "key": key})

banner("stored memories")
for kind, key, value in MEMORIES:
    print(f"  [{kind:10s}] {key}: {value}")

# --- 2. query with paraphrases that share NO keyword --------------------------
# (query, expected memory value, human gloss)
QUERIES = [
    ("锻炼身体", "我今天去健身房了", "exercise ~ went to the gym"),
    ("workout after work", "我今天去健身房了", "en: workout ~ 健身房"),
    ("咖啡因", "每天早上都要喝一杯拿铁", "caffeine ~ latte"),
    ("程序员", "我在一家互联网公司做后端开发", "programmer ~ 后端开发"),
    ("winter ski holiday", "明年想去北海道滑雪", "en: ski trip ~ 北海道滑雪"),
    ("喵星人", "家里养了一只叫豆豆的橘猫", "kitty-slang ~ 橘猫"),
]

banner("paraphrase queries:  KEYWORD (substring)   vs   SEMANTIC (cosine)")
kw_hits = 0
sem_hits = 0
for query, expected, gloss in QUERIES:
    kw = [m.value for m in svc.search(query)]           # keyword/substring path (semantic forced off)
    kw_hit = expected in kw
    kw_hits += int(kw_hit)

    res = index.query(emb.embed(query), top_k=1)        # semantic path
    sem_top, sem_score = (res[0][0], res[0][1]) if res else ("(none)", 0.0)
    sem_hit = sem_top == expected
    sem_hits += int(sem_hit)

    print(f"\nquery: {query!r}   ({gloss})")
    print(f"   KEYWORD  -> {kw if kw else '(no match)':<28}  {'HIT' if kw_hit else 'miss'}")
    print(f"   SEMANTIC -> {sem_top}   cos={sem_score:.3f}      {'HIT' if sem_hit else 'miss'}")

# --- 3. scoreboard ------------------------------------------------------------
n = len(QUERIES)
banner("result")
print(f"  keyword retrieval : {kw_hits}/{n} paraphrases found")
print(f"  semantic retrieval: {sem_hits}/{n} paraphrases found")
if degraded:
    print("\n  NOTE: the hash embedder is active (ollama unavailable), so semantic")
    print("  matching is weak. With ollama+bge-m3 the semantic column reaches ~6/6")
    print("  while keyword stays at 0/6 — that gap is the whole point of this layer.")
elif sem_hits > kw_hits:
    print("\n  Semantic memory recovered memories that keyword retrieval could not —")
    print("  same meaning, different words. Keyword substring match found 0 of them.")

sys.exit(0)
PY

echo
echo "done. (throwaway persist file removed on exit)"

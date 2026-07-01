#!/usr/bin/env python3
"""Semantic-vs-keyword retrieval eval for the memory layer.

The point this harness makes precise: the existing **keyword/curation** retrieval
can only find a memory when the query *lexically overlaps* it (substring match,
then ``retrieval_score`` ranking). It is blind to **paraphrases** -- a query that
means the same thing but shares no words. The **semantic** path (an embedding of
the query, cosine-ranked over an embedding of every memory) is designed to close
exactly that gap.

To show it, we run a small fixed, bilingual (zh+en) corpus of persona memories
against two kinds of query:

* **paraphrase** queries -- reworded / cross-lingual, with **no** shared keyword
  (e.g. store ``我今天去健身房了`` / "I went to the gym today", ask ``锻炼身体`` /
  "workout after work"). Keyword retrieval must miss these; a real semantic
  embedder should catch them.
* **control** queries -- contain a literal term from the memory (e.g. ``拿铁``).
  Keyword retrieval *must* still hit these -- they prove the harness and the
  keyword path are sound, and isolate the paraphrase gap as the real signal.

For each system we report ``recall@k`` (fraction of queries whose gold memory is
in the top-k). With one relevant memory per query this equals hit-rate@k.

Embedder choice (READ THIS to interpret the numbers)
----------------------------------------------------
The embedder comes from ``app.config`` via ``get_embedding_service()`` -- i.e. the
same real code path the app uses. Two regimes:

* ``EMBEDDING_PROVIDER=hash`` (the DEFAULT / CI / offline path): a deterministic
  feature-hashing embedder. It captures *surface* character/word features, NOT
  meaning, so paraphrases with no shared characters stay far apart. Here the eval
  just **exercises the plumbing** end-to-end and the semantic paraphrase recall is
  near the keyword baseline. It runs with no network and is fully reproducible.
* ``EMBEDDING_PROVIDER=ollama`` with ``EMBEDDING_MODEL=bge-m3`` (a running ollama):
  the real multilingual embedder. This is where the semantic path shows its true
  gain -- paraphrase recall jumps toward 1.0 while keyword stays flat.

Run it::

    cd /home/zz79jk/clawd/virtual-person-phase1

    # Plumbing / offline (deterministic hash embedder):
    ~/anaconda3/bin/conda run -n py312 python eval/semantic_eval.py

    # Real semantic gain (needs `ollama serve` + `ollama pull bge-m3`):
    EMBEDDING_PROVIDER=ollama EMBEDDING_MODEL=bge-m3 \
      ~/anaconda3/bin/conda run -n py312 python eval/semantic_eval.py

It ALWAYS exits 0. The machine-readable artifact is ``eval/semantic_report.json``;
the human-readable artifact is the markdown table printed to stdout.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime, timedelta

# --- make the package root importable regardless of cwd -----------------------
# `python eval/semantic_eval.py` puts eval/ (not the repo root) on sys.path[0],
# so the `app` package would not otherwise be importable.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lightweight, deterministic-by-default modules only (no app.main, no DB, no LLM).
from app.config import config  # noqa: E402
from app.memory.curation import (  # noqa: E402
    ScoredMemory,
    rank_memories,
    retrieval_score,
)
from app.memory.embeddings import get_embedding_service  # noqa: E402
from app.memory.vector_store import NumpyVectorIndex  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPORT_PATH = HERE / "semantic_report.json"

K_VALUES = (1, 3, 5)

# Fixed freshness for every memory so the curation recency term is uniform (never
# decides a case) and the run is wall-clock independent -- same rationale as
# run_eval.py's DEFAULT_AGE_DAYS. 3d < the smallest fresh-kind TTL (ephemeral=7d).
DEFAULT_AGE_DAYS = 3


# --- fixed corpus -------------------------------------------------------------
# Persona memories (mostly Chinese, the persona's language). id is used as the
# vector-store key and as the gold label.
CORPUS: list[dict] = [
    {"id": "m_job", "kind": "identity", "key": "职业", "value": "我在一家互联网公司做后端开发"},
    {"id": "m_coffee", "kind": "preference", "key": "饮食", "value": "每天早上都要喝一杯拿铁"},
    {"id": "m_gym", "kind": "note", "key": "运动", "value": "我今天去健身房了"},
    {"id": "m_cat", "kind": "note", "key": "宠物", "value": "家里养了一只叫豆豆的橘猫"},
    {"id": "m_ski", "kind": "goal", "key": "心愿", "value": "明年想去北海道滑雪"},
    {"id": "m_guitar", "kind": "preference", "key": "爱好", "value": "周末喜欢弹吉他放松"},
    {"id": "m_sleep", "kind": "note", "key": "作息", "value": "最近总是熬夜到凌晨两三点"},
    {"id": "m_jp", "kind": "project", "key": "学习", "value": "正在备考日语能力考试"},
]

# Queries. `type` = "paraphrase" (no shared keyword -> keyword MUST miss) or
# "control" (a literal term is present -> keyword MUST hit). `gloss` is a human note.
QUERIES: list[dict] = [
    # -- paraphrases: reworded and/or cross-lingual, zero lexical overlap --------
    {"query": "锻炼身体", "gold": "m_gym", "type": "paraphrase", "gloss": "exercise ~ 去健身房"},
    {"query": "workout after work", "gold": "m_gym", "type": "paraphrase", "gloss": "en: workout ~ 健身房"},
    {"query": "咖啡因", "gold": "m_coffee", "type": "paraphrase", "gloss": "caffeine ~ 拿铁"},
    {"query": "morning coffee habit", "gold": "m_coffee", "type": "paraphrase", "gloss": "en ~ 拿铁"},
    {"query": "程序员", "gold": "m_job", "type": "paraphrase", "gloss": "programmer ~ 后端开发"},
    {"query": "software engineer job", "gold": "m_job", "type": "paraphrase", "gloss": "en ~ 后端开发"},
    {"query": "喵星人", "gold": "m_cat", "type": "paraphrase", "gloss": "kitty-slang ~ 橘猫"},
    {"query": "winter ski holiday", "gold": "m_ski", "type": "paraphrase", "gloss": "en ~ 北海道滑雪"},
    {"query": "弹奏乐器", "gold": "m_guitar", "type": "paraphrase", "gloss": "play an instrument ~ 弹吉他"},
    {"query": "失眠睡不着", "gold": "m_sleep", "type": "paraphrase", "gloss": "insomnia ~ 熬夜到凌晨"},
    {"query": "Japanese exam prep", "gold": "m_jp", "type": "paraphrase", "gloss": "en ~ 日语能力考试"},
    # -- controls: literal keyword present -> keyword retrieval must still hit ----
    {"query": "拿铁", "gold": "m_coffee", "type": "control", "gloss": "literal 拿铁"},
    {"query": "吉他", "gold": "m_guitar", "type": "control", "gloss": "literal 吉他"},
    {"query": "健身房", "gold": "m_gym", "type": "control", "gloss": "literal 健身房"},
]


# --- keyword / curation retrieval (the existing, deployed path) ---------------
def _keyword_candidates(query: str) -> list[dict]:
    """Substring candidate filter -- mirrors MemoryService.search() exactly.

    A paraphrase with no shared keyword yields an EMPTY candidate set, which is
    the whole reason the semantic path exists.
    """
    q = (query or "").lower().strip()
    return [
        m
        for m in CORPUS
        if not q or q in m["key"].lower() or q in m["value"].lower() or q in m["kind"].lower()
    ]


def keyword_topk(query: str, k: int, created_at: str) -> list[str]:
    """Rank keyword candidates with the real curation pipeline; return top-k ids."""
    scored = [
        ScoredMemory(
            kind=m["kind"],
            key=m["key"],
            value=m["value"],
            created_at=created_at,
            score=retrieval_score(m["kind"], m["key"], m["value"], created_at, query),
            source=m["id"],  # stash id so we can map the ranked winner back
        )
        for m in _keyword_candidates(query)
    ]
    ranked = rank_memories(scored, limit=max(k, 1))
    return [r.source for r in ranked][:k]


# --- semantic retrieval (embedding + numpy cosine index) ----------------------
def build_semantic_index(embedder) -> NumpyVectorIndex:
    """Embed ``f'{key} {value}'`` for every memory into a numpy cosine index."""
    idx = NumpyVectorIndex(dim=embedder.dim)
    for m in CORPUS:
        text = f'{m["key"]} {m["value"]}'
        idx.upsert(m["id"], text, embedder.embed(text), {"id": m["id"]})
    return idx


# --- evaluation ---------------------------------------------------------------
def evaluate(embedder, idx: NumpyVectorIndex) -> list[dict]:
    """Compute, per query, the keyword and semantic top-`max(K)` id lists."""
    maxk = max(K_VALUES)
    created_at = (datetime.now(UTC) - timedelta(days=DEFAULT_AGE_DAYS)).isoformat()
    per_query: list[dict] = []
    for qc in QUERIES:
        kw = keyword_topk(qc["query"], maxk, created_at)
        sem_res = idx.query(embedder.embed(qc["query"]), top_k=maxk)
        sem = [r[0] for r in sem_res]
        per_query.append(
            {
                "query": qc["query"],
                "gold": qc["gold"],
                "type": qc["type"],
                "gloss": qc["gloss"],
                "keyword_topk": kw,
                "semantic_topk": sem,
                "semantic_top_score": round(sem_res[0][1], 4) if sem_res else None,
                "keyword_hit_at_1": bool(kw[:1] == [qc["gold"]]),
                "semantic_hit_at_1": bool(sem[:1] == [qc["gold"]]),
            }
        )
    return per_query


def _recall(per_query: list[dict], system: str, k: int, subset: str | None = None) -> float | None:
    """recall@k for a system ("keyword"/"semantic"), optionally within a subset."""
    rows = [p for p in per_query if subset is None or p["type"] == subset]
    if not rows:
        return None
    hits = sum(1 for p in rows if p["gold"] in p[f"{system}_topk"][:k])
    return round(hits / len(rows), 4)


def _recall_block(per_query: list[dict], system: str) -> dict:
    return {
        "overall": {f"recall@{k}": _recall(per_query, system, k) for k in K_VALUES},
        "paraphrase": {f"recall@{k}": _recall(per_query, system, k, "paraphrase") for k in K_VALUES},
        "control": {f"recall@{k}": _recall(per_query, system, k, "control") for k in K_VALUES},
    }


def _ollama_reachable(base_url: str) -> bool:
    try:
        from urllib.request import urlopen

        with urlopen(base_url.rstrip("/") + "/api/tags", timeout=1.5) as resp:  # noqa: S310
            return getattr(resp, "status", 200) == 200
    except Exception:
        return False


# --- report -------------------------------------------------------------------
def build_report() -> dict:
    embedder = get_embedding_service()
    provider = embedder.provider

    # A non-hash provider silently degrades to hash on any failure; surface the
    # *effective* embedder honestly so a "provider=ollama" run against a dead
    # server isn't mistaken for real semantic quality.
    reachable = None
    effective = provider
    if provider == "ollama":
        reachable = _ollama_reachable(config.ollama_base_url)
        if not reachable:
            effective = "hash (ollama unreachable -> degraded)"

    idx = build_semantic_index(embedder)
    per_query = evaluate(embedder, idx)

    keyword = _recall_block(per_query, "keyword")
    semantic = _recall_block(per_query, "semantic")

    # Headline: paraphrase recall@3 is where the semantic gain lives.
    gain = None
    kw_para = keyword["paraphrase"]["recall@3"]
    sem_para = semantic["paraphrase"]["recall@3"]
    if kw_para is not None and sem_para is not None:
        gain = round(sem_para - kw_para, 4)

    is_real = provider == "ollama" and reachable
    note = (
        "Real multilingual embedder (ollama/bge-m3): paraphrase recall reflects "
        "true semantic retrieval quality."
        if is_real
        else "Offline hash embedder: this run only EXERCISES THE PLUMBING end-to-end. "
        "Hashed features capture surface form (character n-grams), not meaning, so any "
        "lift over the keyword baseline is incidental surface overlap and stays well below "
        "what a real embedder reaches. Re-run with EMBEDDING_PROVIDER=ollama "
        "EMBEDDING_MODEL=bge-m3 (ollama serving bge-m3) to see the real semantic gain."
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding": {
            "provider": provider,
            "effective_provider": effective,
            "model": embedder.model,
            "dim": embedder.dim,
            "ollama_base_url": config.ollama_base_url if provider == "ollama" else None,
            "ollama_reachable": reachable,
            "semantic_memory_enabled": config.semantic_memory_enabled,
        },
        "k_values": list(K_VALUES),
        "n_memories": len(CORPUS),
        "n_queries": len(QUERIES),
        "n_paraphrase": sum(1 for q in QUERIES if q["type"] == "paraphrase"),
        "n_control": sum(1 for q in QUERIES if q["type"] == "control"),
        "systems": {"keyword": keyword, "semantic": semantic},
        "paraphrase_recall_at_3_gain": gain,
        "note": note,
        "queries": per_query,
    }


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def render_markdown(report: dict) -> str:
    emb = report["embedding"]
    kw = report["systems"]["keyword"]
    sem = report["systems"]["semantic"]
    prov = f"{emb['effective_provider']} · dim={emb['dim']}"

    def row(label: str, block: dict, section: str) -> str:
        b = block[section]
        return (
            f"| {label} | {_fmt(b['recall@1'])} | {_fmt(b['recall@3'])} | {_fmt(b['recall@5'])} |"
        )

    lines = [
        "# Semantic vs. Keyword Retrieval Eval",
        "",
        f"_Generated: {report['generated_at']}_",
        "",
        f"- Embedder: **{prov}**  (provider requested: `{emb['provider']}`"
        + (f", ollama_reachable={emb['ollama_reachable']}" if emb["provider"] == "ollama" else "")
        + ")",
        f"- Corpus: {report['n_memories']} memories · Queries: {report['n_queries']} "
        f"({report['n_paraphrase']} paraphrase, {report['n_control']} control)",
        "",
        f"> {report['note']}",
        "",
        "## Overall recall@k (all queries)",
        "",
        "| System | recall@1 | recall@3 | recall@5 |",
        "|--------|----------|----------|----------|",
        row("keyword / curation", kw, "overall"),
        row(f"semantic ({emb['effective_provider']})", sem, "overall"),
        "",
        "## Paraphrase queries only  ← the real test (no shared keyword)",
        "",
        "| System | recall@1 | recall@3 | recall@5 |",
        "|--------|----------|----------|----------|",
        row("keyword / curation", kw, "paraphrase"),
        row(f"semantic ({emb['effective_provider']})", sem, "paraphrase"),
        "",
        f"**Paraphrase recall@3 gain (semantic − keyword): {_fmt(report['paraphrase_recall_at_3_gain'])}**",
        "",
        "## Control queries only  (literal keyword present — sanity check)",
        "",
        "| System | recall@1 | recall@3 | recall@5 |",
        "|--------|----------|----------|----------|",
        row("keyword / curation", kw, "control"),
        row(f"semantic ({emb['effective_provider']})", sem, "control"),
        "",
        "## Per paraphrase query (keyword top-1 vs semantic top-1)",
        "",
        "| Query | means | gold | keyword→ | semantic→ (cos) | winner |",
        "|-------|-------|------|----------|-----------------|--------|",
    ]
    for p in report["queries"]:
        if p["type"] != "paraphrase":
            continue
        kw_top = p["keyword_topk"][0] if p["keyword_topk"] else "—"
        sem_top = p["semantic_topk"][0] if p["semantic_topk"] else "—"
        score = "" if p["semantic_top_score"] is None else f" {p['semantic_top_score']:.2f}"
        if p["semantic_hit_at_1"] and not p["keyword_hit_at_1"]:
            winner = "semantic"
        elif p["keyword_hit_at_1"] and not p["semantic_hit_at_1"]:
            winner = "keyword"
        elif p["keyword_hit_at_1"] and p["semantic_hit_at_1"]:
            winner = "both"
        else:
            winner = "neither"
        lines.append(
            f"| `{p['query']}` | {p['gloss']} | {p['gold']} | {kw_top} | {sem_top}{score} | {winner} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    print(render_markdown(report))
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report -> {REPORT_PATH}")
    # The report is the artifact; always exit 0 so this never blocks a pipeline.
    return 0


if __name__ == "__main__":
    sys.exit(main())

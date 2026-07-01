# Evaluation

Two complementary evals guard different things. The deterministic harness is the
CI gate for correctness; the LLM-judge eval is a local, subjective quality check.

| Eval | Entry point | In CI? | Requires | Measures |
|------|-------------|:------:|----------|----------|
| **Deterministic harness** | `eval/run_eval.py` + `tests/test_eval_harness.py` | yes | nothing (pure in-memory) | safety-gate accuracy/precision/recall, memory-retrieval top-1 |
| **LLM-judge quality** | `eval/run_llm_eval.py` | no | app on `:8090` + ollama on `:11434` | persona-consistency + helpfulness of *live* replies |

All Python runs use the project's conda env: `~/anaconda3/bin/conda run -n py312`.

---

## 1. Deterministic harness (CI gate, no LLM)

Exercises the pure service logic behind `/chat/turn` and asserts fixed golden
metrics. It imports only the lightweight modules (`app.safety.service`,
`app.memory.curation`) — never `app.main` — so there is **no network, no LLM, no
Redis, and no database**. It is fast (single-digit milliseconds) and fully
reproducible, which is why it runs on every commit.

### Run

```bash
cd /home/zz79jk/clawd/virtual-person-phase1

# One-off report: prints a markdown table and writes eval/report.json.
# ALWAYS exits 0 — the report is the artifact, gating is done by pytest below.
~/anaconda3/bin/conda run -n py312 python eval/run_eval.py

# CI gate: asserts the thresholds. Deterministic; needs no REDIS_URL.
~/anaconda3/bin/conda run -n py312 python -m pytest tests/test_eval_harness.py -q
```

### Suites & metrics

| Suite | Data | Metric | Threshold |
|-------|------|--------|-----------|
| `safety` (input gate) | `eval/datasets/safety.jsonl` (16 cases) | accuracy, precision, recall, F1 of refusal | accuracy / precision / recall ≥ **0.85** |
| `output_gate` | 8 inline cases in `run_eval.py` | accuracy | ≥ **0.85** |
| `retrieval` | `eval/datasets/retrieval.jsonl` (10 cases) | top-1 ranking accuracy | ≥ **0.70** |

- **safety (input gate)** — treats `SafetyService.check()` as a binary
  "refused / allowed" classifier. Positive class is *refused*. It reports a full
  confusion matrix (tp/fp/tn/fn); the test requires both high **recall** (don't
  miss unsafe prompts) and high **precision** (don't over-refuse benign ones), so
  a gate that trivially always- or never-refuses cannot pass.
- **output_gate** — same idea for `SafetyService.check_output()`, the
  moderation pass applied to generated replies.
- **retrieval** — scores each candidate memory with `retrieval_score()`, ranks
  with `rank_memories()` (stale-drop → dedup → sort → truncate), and checks that
  the top-ranked item is the expected one. Candidate ages are chosen so the
  recency term is uniform, making the outcome wall-clock independent.

Golden datasets currently score **1.00 / 1.00 / 1.00**. `tests/test_eval_harness.py`
asserts the thresholds and also verifies two back-to-back runs are byte-identical
(`test_build_report_is_deterministic`).

### Artifacts

- `eval/report.json` — machine-readable metrics, confusion matrix, and per-case
  retrieval detail / any failures.
- stdout — a human-readable markdown summary table.

---

## 2. LLM-judge quality eval (local, needs ollama)

Sends a small fixed set of 8 user prompts to a **running** app and asks a local
LLM (ollama) to grade each reply. This measures subjective conversational
quality that the deterministic harness cannot: does the reply *sound like the
persona* and is it *actually helpful*? It is **not** run in CI — it needs live
services and an LLM, and its scores are not bit-reproducible.

The 8 prompts probe: self-introduction, turning a messy week into a plan,
empathy on a bad day, explaining reasoning, concrete next steps, gentle advice,
values/boundaries, and one safety-boundary prompt (the app refuses it
deterministically, and the judge rates how gracefully it declines in persona).

### Prerequisites

```bash
# ollama up with the judge model pulled
ollama serve                      # http://127.0.0.1:11434
ollama pull qwen3:32b

# the app on :8090 (ollama-backed, no vLLM/GPU needed)
cd /home/zz79jk/clawd/virtual-person-phase1
bash scripts/demo/serve.sh        # serves http://127.0.0.1:8090
```

### Run

```bash
cd /home/zz79jk/clawd/virtual-person-phase1
~/anaconda3/bin/conda run -n py312 python eval/run_llm_eval.py

# With overrides (all optional):
BASE=http://127.0.0.1:8090 OLLAMA_MODEL=qwen3:32b \
  ~/anaconda3/bin/conda run -n py312 python eval/run_llm_eval.py
```

It prints per-case scores plus averages, and writes `eval/llm_report.json`
(full prompt, app reply, per-axis scores, and one-line judge rationale per case).

### Metrics — each scored 1 (poor) to 5 (excellent)

- **persona_consistency** — does the reply match the configured persona: warm,
  empathetic, a good listener, concise, with healthy boundaries? The judge is
  grounded in the app's *actual* persona, fetched live from `GET /persona/profile`.
- **helpfulness** — does the reply address the user's need with useful,
  accurate, actionable content? A polite in-persona refusal of an unsafe request
  counts as correct, helpful behavior.
- **overall** — the mean of the two axes, reported per case and as an aggregate
  average across all scored cases.

> Note: by default the judge model equals the app's model (`qwen3:32b`). That is
> fine for a local smoke check; to reduce self-preference bias, set `OLLAMA_MODEL`
> to a *different* model than the one the app generates with.

### Exit codes

| Code | Meaning |
|:----:|---------|
| `0` | Eval ran and produced a report (any quality score). |
| `2` | App or ollama unreachable / misconfigured — prints a clear, actionable message. |
| `3` | `MIN_AVG` was set and the overall average fell below it (optional quality gate). |

### Configuration (env or flags; see `--help`)

| Env | Flag | Default | Purpose |
|-----|------|---------|---------|
| `BASE` | `--base` | `http://127.0.0.1:8090` | App base URL. |
| `OLLAMA_URL` | `--ollama-url` | `http://127.0.0.1:11434` | Ollama base URL. |
| `OLLAMA_MODEL` | `--model` | `qwen3:32b` | Judge model. |
| `EVAL_USER_ID` | `--user-id` | `llm-eval` | `user_id` sent to `/chat/turn`. |
| `MIN_AVG` | `--min-avg` | *(unset)* | Optional pass gate on the overall average. |
| `CHAT_TIMEOUT` | `--chat-timeout` | `120` | Per-turn timeout (the app's own LLM can be slow). |
| `JUDGE_TIMEOUT` | `--judge-timeout` | `180` | Per-judge-call timeout (large models are slow). |

Dependencies: Python stdlib + `requests` (already a project dependency). The
script is a pure HTTP client — it does not import the `app` package.

---

## Files

```
eval/
  run_eval.py          # deterministic harness (CI gate, no LLM)
  run_llm_eval.py      # LLM-judge quality eval (local, needs app + ollama)
  datasets/
    safety.jsonl       # 16 labelled input-safety cases
    retrieval.jsonl    # 10 memory-ranking cases
  report.json          # deterministic harness output
  llm_report.json      # LLM-judge output (created on first run)
tests/
  test_eval_harness.py # pytest wrapper that gates on the deterministic thresholds
```

#!/usr/bin/env bash
# LLM-as-judge quality eval on the PRODUCTION stack.
#
# The app under test runs on vLLM (Qwen/Qwen3-14B-AWQ), so the replies being
# scored are REAL production output. The judge is an INDEPENDENT model
# (ollama qwen3:32b) on purpose: best practice is judge != system-under-test, so
# a model doesn't grade its own output. The report records both (app_llm_model +
# judge_model) for auditability.
#
# Requires: app on :8090 via scripts/demo/serve_vllm.sh (vLLM), ollama up with
# the judge model pulled. This is NOT a CI job (the deterministic, LLM-free gate
# is eval/run_eval.py). Pass-through args/flags go to run_llm_eval.py.
set -euo pipefail
cd "$(dirname "$0")/../.."

export BASE="${BASE:-http://127.0.0.1:8090}"      # app under test (expect vLLM)
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:32b}"   # the JUDGE model (independent)

echo "[llm-eval/vllm] SUT app = $BASE  (expect app_llm_model = Qwen/Qwen3-14B-AWQ)"
echo "[llm-eval/vllm] judge   = ollama:$OLLAMA_MODEL  (independent of the SUT)"
exec ~/anaconda3/bin/conda run --no-capture-output -n py312 python eval/run_llm_eval.py "$@"

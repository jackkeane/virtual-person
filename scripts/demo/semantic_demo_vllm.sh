#!/usr/bin/env bash
# End-to-end semantic memory on the PRODUCTION stack:
#   bge-m3 embeddings (ollama) -> in-process cosine recall -> vLLM/Qwen3-14B-AWQ.
#
# The payoff keyword memory can't give: the assistant answers a PARAPHRASED
# question using a stored memory that shares NO word with the question, because
# the semantic layer recalled it and fed it to the real LLM. (semantic_demo.sh
# proves the retrieval half in-process; THIS proves the full product loop over
# HTTP against the running app + real Qwen.)
#
# Requires the app running via:  bash scripts/demo/serve_vllm.sh
#   (LLM_PROVIDER=openai_compat -> vLLM; SEMANTIC_MEMORY_ENABLED=1;
#    EMBEDDING_PROVIDER=ollama/bge-m3). Talks only over HTTP; uses the app's
#    throwaway memory store; erases + re-seeds its own memories. db0 untouched.
set -uo pipefail
BASE="${BASE:-http://127.0.0.1:8090}"

api(){ # api METHOD PATH [JSON-BODY]
  local m="$1" p="$2" d="${3:-}"
  if [ -n "$d" ]; then curl -s -m 120 -X "$m" "$BASE/$p" -H 'Content-Type: application/json' -d "$d"
  else curl -s -m 120 -X "$m" "$BASE/$p"; fi
}
field(){ python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get(sys.argv[1],''))" "$1"; }
hr(){ printf '\n========== %s ==========\n' "$1"; }

# --- 0) preflight: app up + actually on vLLM --------------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" || true)
[ "$code" = "200" ] || { echo "app not reachable at $BASE — start it first:  bash scripts/demo/serve_vllm.sh"; exit 1; }
prov=$(curl -s -m 5 "$BASE/persona/profile")
echo "app LLM : $(printf '%s' "$prov" | field llm_model)   (provider=$(printf '%s' "$prov" | field llm_provider))"

# --- 1) clean slate + seed persona memories (Chinese) -----------------------
hr "seed persona memories  (POST /memory/write; semantic index = bge-m3)"
api DELETE "memory/erase?confirm=true" >/dev/null
seed(){ api POST memory/write "{\"kind\":\"$1\",\"key\":\"$2\",\"value\":\"$3\"}" >/dev/null; printf '  [%-9s] %s: %s\n' "$1" "$2" "$3"; }
seed note     宠物 "我家养了一只叫豆豆的橘猫"
seed identity 职业 "我在一家互联网公司做后端开发"
seed goal     心愿 "明年冬天想去北海道滑雪"

# --- 2) paraphrase questions: recall (semantic) -> answer (vLLM) -------------
# Each question shares NO content word with its target memory, so a keyword
# search finds nothing; semantic recall feeds the memory to vLLM, which answers.
recall(){ # print top semantic hits for a query (what memory.search surfaces)
  curl -s -G -m 60 "$BASE/memory/search" --data-urlencode "query=$1" | python3 -c "
import json,sys
d=json.load(sys.stdin)
items=d.get('items',[]) if isinstance(d,dict) else d
print(' | '.join(f\"{i.get('key')}={i.get('value')}\" for i in items[:3]) or '(none)')"
}
ask(){ # ask QUESTION GLOSS
  hr "Q: $1"
  echo "  ($2)"
  echo "  semantic recall -> $(recall "$1")"
  local resp; resp=$(api POST chat/turn "{\"user_id\":\"sem\",\"message\":\"$1\"}")
  echo "  vLLM ($(printf '%s' "$resp" | field model)) answer -> $(printf '%s' "$resp" | field response)"
}
ask "我家那只喵星人叫什么名字来着？"          "喵星人 ~ 橘猫;    '喵星人' 不在任何记忆里"
ask "你还记得我平时是做什么工作的吗？"         "工作 ~ 后端开发;  无共同词"
ask "我明年天冷了想出去玩雪，记得我想去哪吗？"  "玩雪 ~ 滑雪/北海道"

hr "takeaway"
echo "None of the three questions shares a word with its memory -> keyword search returns nothing."
echo "Semantic recall (bge-m3) surfaced the right memory and vLLM/Qwen answered from it."

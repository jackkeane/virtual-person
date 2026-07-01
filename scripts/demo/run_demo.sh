#!/usr/bin/env bash
# Redis + observability live demo for virtual-person (Feature 3).
#
# Exercises all four Redis-gated features against a running app and prints the
# real numbers an interviewer would ask about:
#   1. Prometheus metrics move on the HTTP path (turn counter + latency)
#   2. Sessions persist to Redis (durable, TTL-bounded, survives restart)
#   3. TTS response cache: miss -> hit latency collapse
#   4. Per-user token-bucket rate limiting rejects a burst
#
# Prereqs: app running with REDIS_URL set (see scripts/demo/serve.sh), Redis up.
# Safe: uses a dedicated Redis DB (default 15), never touches db0.
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8090}"
DB="${DB:-15}"
R() { redis-cli -n "$DB" "$@"; }

hr() { printf '\n========== %s ==========\n' "$1"; }
post() { curl -s -m 60 -X POST "$BASE/chat/turn" -H 'Content-Type: application/json' -d "$1"; }
# value of an unlabeled exposition line "<name> <value>"
metric() { curl -s "$BASE/metrics" | awk -v n="$1" '$1==n{print $2; exit}'; }

hr "clean slate: FLUSHDB db$DB (never db0)"
R FLUSHDB >/dev/null
echo "db$DB size now: $(R DBSIZE)"

hr "SECTION 1 — HTTP metrics + Redis session store"
tb=$(metric vp_turns_total); tb=${tb:-0}
for _ in 1 2 3; do post '{"user_id":"alice","message":"who are you"}' >/dev/null; done
ta=$(metric vp_turns_total); ta=${ta:-0}
echo "vp_turns_total (HTTP path): ${tb} -> ${ta}   (was flat before this change)"
echo "chat-latency samples recorded: $(metric vp_chat_seconds_count)"
echo "session key(s) in Redis:"; R KEYS 'vp:sess:*'
echo "alice conversation history (Redis LIST, role|text):"; R LRANGE vp:sess:alice 0 -1
echo "alice session TTL: $(R TTL vp:sess:alice) s  (7-day bound -> idle keys evaporate)"

hr "SECTION 1B — REAL LLM turn (open-ended message -> the app's real LLM)"
# The 'who are you' turns above hit the deterministic persona branch (~2ms,
# model=rule-persona, no LLM). An open-ended message with no trigger keyword
# flows through the REAL provider: on serve_vllm.sh that's vLLM/Qwen3-14B-AWQ.
# The model name is echoed from the response body; vp_chat_seconds now captures
# true inference latency (contrast the stub's ~2ms).
csb=$(metric vp_chat_seconds_sum); csb=${csb:-0}
rt=$(curl -s -o /tmp/vp_real_turn.json -w '%{time_total}' -m 120 -X POST "$BASE/chat/turn" \
       -H 'Content-Type: application/json' \
       -d '{"user_id":"real","message":"我今天工作压力有点大，用一句话鼓励我一下"}')
csa=$(metric vp_chat_seconds_sum); csa=${csa:-0}
echo "  end-to-end latency: ${rt}s"
python3 -c "import json; d=json.load(open('/tmp/vp_real_turn.json')); print('  model :', d.get('model'), '  ok:', d.get('ok')); print('  reply :', d.get('response'))"
awk -v a="$csb" -v b="$csa" 'BEGIN{printf "  server-side inference latency (delta vp_chat_seconds_sum): %.3f s   (stub branch was ~0.002s)\n", b-a}'
echo "  real reply persisted to Redis session vp:sess:real:"; R LRANGE vp:sess:real 0 -1

hr "SECTION 2 — TTS response cache: miss (full synth) vs hit (Redis GET)"
cache_ct() { curl -s "$BASE/metrics" | grep -F "vp_tts_cache_total{result=\"$1\"}" | awk '{print $2}'; }
mb=$(cache_ct miss); mb=${mb:-0}; hb=$(cache_ct hit); hb=${hb:-0}
sum_miss=0; sum_hit=0; n=0
for phrase in "缓存演示第一句话" "缓存演示第二句不一样" "缓存演示第三句再不同" "缓存演示第四句继续变"; do
  body=$(printf '{"text":"%s"}' "$phrase")
  miss=$(curl -s -o /dev/null -w '%{time_total}' -m 20 -X POST "$BASE/voice/synthesize" -H 'Content-Type: application/json' -d "$body")
  hit=$(curl -s -o /dev/null -w '%{time_total}' -m 20 -X POST "$BASE/voice/synthesize" -H 'Content-Type: application/json' -d "$body")
  printf "  miss=%ss  hit=%ss\n" "$miss" "$hit"
  sum_miss=$(awk -v a="$sum_miss" -v b="$miss" 'BEGIN{print a+b}')
  sum_hit=$(awk -v a="$sum_hit" -v b="$hit" 'BEGIN{print a+b}')
  n=$((n+1))
done
ma=$(cache_ct miss); ma=${ma:-0}; ha=$(cache_ct hit); ha=${ha:-0}
awk -v tm="$sum_miss" -v th="$sum_hit" -v n="$n" 'BEGIN{
  am=tm/n*1000; ah=th/n*1000;
  printf "  --> avg MISS=%.1f ms | avg HIT=%.2f ms | %.0fx faster | %.2f%% latency cut\n", am, ah, (ah>0?am/ah:0), (am>0?(am-ah)/am*100:0)
}'
echo "  cache counter delta this run: miss +$(awk -v a="$mb" -v b="$ma" 'BEGIN{print b-a}'), hit +$(awk -v a="$hb" -v b="$ha" 'BEGIN{print b-a}')"
echo "  (baseline misses = filler audio pre-warmed into the cache at app startup)"
k=$(R KEYS 'vp:tts:*' | head -1)
echo "  cache keys: $(R KEYS 'vp:tts:*' | wc -l) total; sample key TTL: $(R TTL "$k") s"

hr "SECTION 3 — per-caller rate limiting (keyed on client IP, capacity=5)"
for k in $(R KEYS 'vp:rl:*'); do R DEL "$k" >/dev/null; done   # reset limiter for a clean burst
rb=$(metric vp_rate_limited_total); rb=${rb:-0}
ok=0; limited=0
for _ in $(seq 1 10); do
  resp=$(post '{"user_id":"burst","message":"who are you"}')
  if printf '%s' "$resp" | grep -q '"error":"rate_limited"'; then limited=$((limited+1)); else ok=$((ok+1)); fi
done
ra=$(metric vp_rate_limited_total); ra=${ra:-0}
echo "burst of 10 rapid requests from one client IP -> allowed=${ok}, rejected=${limited}"
echo "vp_rate_limited_total: ${rb} -> ${ra}"
rlkey=$(R KEYS 'vp:rl:*' | head -1)
echo "bucket key: ${rlkey}  TTL=$(R TTL "$rlkey") s  (keyed on caller IP, not the spoofable user_id)"

hr "FINAL — app metric families (/metrics)"
mkdir -p "$(dirname "$0")/../../docs/demo"
OUT="$(cd "$(dirname "$0")/../../" && pwd)/docs/demo/metrics_snapshot.txt"
curl -s "$BASE/metrics" > "$OUT"
grep -E '^vp_' "$OUT" | grep -vE '_bucket|_created'
echo
echo "full /metrics saved -> docs/demo/metrics_snapshot.txt"
echo "all Redis keys created by this demo (db$DB):"; R KEYS '*' | sort

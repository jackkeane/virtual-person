import json
import re
import os
import base64
import time
from datetime import datetime
from typing import Iterable
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import config
from app.logging.audit import AuditLogger
from app.memory.service import MemoryService
from app.memory.curation import ScoredMemory, rank_memories, retrieval_score
from app.persona.service import PersonaService
from app.proactivity.service import ProactivityService
from app.safety.service import SafetyService
from app.session.service import SessionService
from app.llm.ollama_client import OllamaClient
from app.llm.openai_compat_client import OpenAICompatClient
from app.tools.registry import ToolRegistry
from app.tools.builtins import register_builtins
from app.avatar.state_machine import ConversationStateMachine
from app.avatar.emotion_mapper import EmotionMapper
from app.voice.tts_service import FillerAudioService, get_tts_service
from app.voice.stt_service import get_stt_service
from app.voice.lipsync import build_viseme_timeline
from app.ws.handler import AvatarWebSocketHandler
from app.infra.rate_limit import TokenBucketLimiter
from app.observability import metrics as obs_metrics


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3-style reasoning output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def maybe_capture_user_name(user_id: str, message: str) -> None:
    """Best-effort name capture from natural user utterances."""
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\- '\.]{0,39})",
        r"\bi am\s+([A-Za-z][A-Za-z\- '\.]{0,39})",
        r"\bi'm\s+([A-Za-z][A-Za-z\- '\.]{0,39})",
        r"\bcall me\s+([A-Za-z][A-Za-z\- '\.]{0,39})",
        r"我是\s*([\u4e00-\u9fffA-Za-z0-9·•]{1,20})",
        r"我叫\s*([\u4e00-\u9fffA-Za-z0-9·•]{1,20})",
    ]

    name: str | None = None
    for p in patterns:
        m = re.search(p, message, flags=re.IGNORECASE)
        if m:
            name = (m.group(1) or "").strip("\"'，。,.!！?？:： ")
            break

    if not name:
        return

    invalid_names = {
        "什么", "什么名字", "谁", "你谁", "我", "你", "unknown", "name", "名字",
        "what", "who", "my", "your"
    }
    if name.lower() in invalid_names or any(tok in name for tok in ["什么", "？", "?"]):
        return

    item = memory.write(
        "profile",
        "name",
        name,
        metadata={"source": "chat_auto_extract", "user_id": user_id, "field": "name"},
    )
    if not item.metadata.get("filtered"):
        audit.log("memory_write_auto", f"profile:name={name}|user={user_id}")


def recall_user_name(user_id: str) -> str | None:
    """Get most relevant stored user name, preferring entries for this user_id."""
    hits = memory.search("name")
    candidates = [h for h in hits if h.key.lower() == "name"]
    if not candidates:
        return None

    invalid_names = {"什么", "什么名字", "谁", "你谁", "我", "你", "unknown", "name", "名字", "what", "who"}

    def _valid(v: str) -> bool:
        t = (v or "").strip().lower()
        if not t or t in invalid_names:
            return False
        if "什么" in t or "?" in t or "？" in t:
            return False
        return True

    user_specific = [h for h in candidates if (h.metadata or {}).get("user_id") == user_id and _valid(h.value)]
    global_valid = [h for h in candidates if _valid(h.value)]

    pool = user_specific if user_specific else global_valid
    pool = sorted(pool, key=lambda x: x.created_at)
    return (pool[-1].value or "").strip() if pool else None


app = FastAPI(title=config.app_name)

persona = PersonaService(
    name=config.persona_name,
    occupation=config.persona_occupation,
    age=config.persona_age,
    backstory=config.persona_backstory,
    profile_path=config.persona_profile_path,
)
memory = MemoryService()
safety = SafetyService()
audit = AuditLogger(persist_path=config.audit_log_path if config.audit_log_path else None)
sessions = SessionService(max_turns=20)
# Per-user token-bucket rate limiter. Inert (allow-all) unless REDIS_URL is set
# AND Redis is reachable — see app/infra/rate_limit.py.
limiter = TokenBucketLimiter(
    capacity=config.rate_limit_capacity,
    refill_per_sec=config.rate_limit_refill_per_sec,
)
proactivity = ProactivityService(
    quiet_start=config.quiet_hours_start,
    quiet_end=config.quiet_hours_end,
    cooldown_minutes=config.proactive_cooldown_minutes,
)
ollama = OllamaClient(
    base_url=config.ollama_base_url,
    model=config.ollama_model,
    timeout_seconds=config.llm_timeout_seconds,
)
openai_compat = OpenAICompatClient(
    base_url=config.openai_compat_base_url,
    model=config.openai_compat_model,
    api_key=config.openai_compat_api_key,
    timeout_seconds=config.llm_timeout_seconds,
)

# --- Tool registry ---
tools = ToolRegistry()
register_builtins(tools, memory_svc=memory, proactivity_svc=proactivity)

# --- Avatar/voice runtime ---
state_machine = ConversationStateMachine()
emotion_mapper = EmotionMapper()
tts_service = get_tts_service()
stt_service = get_stt_service()
filler_service = FillerAudioService(primary_tts=tts_service)

# Static client
app.mount("/client", StaticFiles(directory="client", html=True), name="client")


# --- Helpers ---

def _llm_call(system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> tuple[str, str]:
    """Call the configured LLM and return (text, model_used)."""
    if config.llm_provider == "ollama":
        raw = ollama.chat(system_prompt=system_prompt, user_prompt=user_prompt, history=history)
        model_used = config.ollama_model
    else:
        raw = openai_compat.chat(system_prompt=system_prompt, user_prompt=user_prompt, history=history)
        model_used = config.openai_compat_model
    if config.llm_strip_thinking:
        raw = strip_thinking(raw or "")
    return raw or "", model_used


def _tool_call_loop(user_message: str, memory_context: str, history: list[dict]) -> tuple[str, str, list[dict]]:
    """
    Let the LLM decide whether to call tools. If it emits a JSON tool call block,
    execute the tool and feed the result back for a final answer.
    Returns (final_response, model_used, tool_actions).
    """
    tool_defs = tools.list_tools()
    tool_names = [t["name"] for t in tool_defs]
    tool_desc = "\n".join([f"- {t['name']} ({t['risk']}): {t['description']}" for t in tool_defs])

    prompt = (
        f"User message: {user_message}\n"
        f"Known memory (ranked):\n{memory_context if memory_context else '- none'}\n\n"
        f"Available tools:\n{tool_desc}\n\n"
        "If a tool would help, respond with ONLY a JSON block like:\n"
        '{"tool_call": {"name": "tool_name", "params": {"key": "value"}}}\n'
        "Otherwise, just answer naturally and helpfully."
    )

    raw, model_used = _llm_call(persona.system_prompt(), prompt, history)
    tool_actions: list[dict] = []

    # Check if LLM wants to call a tool
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "tool_call" in parsed:
            tc = parsed["tool_call"]
            tool_name = tc.get("name", "")
            tool_params = tc.get("params", {})
            if tool_name in tool_names:
                result = tools.execute(tool_name, tool_params)
                tool_actions.append({"tool": tool_name, "params": tool_params, "result": result})
                audit.log("tool_auto_exec", f"{tool_name}")

                # Second LLM call with tool result
                followup = (
                    f"User message: {user_message}\n"
                    f"You called tool '{tool_name}' and got:\n{json.dumps(result, default=str)}\n"
                    "Now answer the user naturally using this result."
                )
                raw2, _ = _llm_call(persona.system_prompt(), followup, history)
                return raw2, model_used, tool_actions
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return raw, model_used, tool_actions


def _chat_via_pipeline(user_id: str, message: str) -> dict:
    # Internal entrypoint (used by the WS path, which already rate-limits the
    # turn in AvatarWebSocketHandler._process_text_chat). Call the shared core
    # directly so the endpoint-level limiter does NOT consume a second token.
    return _run_chat_turn(ChatTurnIn(user_id=user_id, message=message))


def _stream_chat_via_pipeline(user_id: str, message: str) -> Iterable[str]:
    """Streaming path for WS: LLM deltas -> progressive sentence updates on client."""
    # Reuse existing non-stream pipeline when streaming is disabled/provider unsupported.
    if (not config.ws_stream_enabled) or config.llm_provider != "openai_compat":
        result = _chat_via_pipeline(user_id, message)
        yield result.get("response", "")
        return

    # Safety checks (same intent as /chat/turn)
    safe, refusal = safety.check(message)
    if not safe:
        yield refusal
        return

    maybe_capture_user_name(user_id, message)

    message_l = message.lower()
    if any(k in message_l for k in ["my name", "who am i", "do you remember my name", "我的名字", "我叫什么"]):
        remembered_name = recall_user_name(user_id)
        if remembered_name:
            response = persona.enforce(f"Your name is {remembered_name}.")
            sessions.add(user_id, "user", message)
            sessions.add(user_id, "assistant", response)
            yield response
            return

    history = sessions.get(user_id, limit=10)
    system_prompt = persona.system_prompt()

    streamed_text = ""
    try:
        for delta in openai_compat.chat_stream(system_prompt=system_prompt, user_prompt=message, history=history):
            streamed_text += delta
            yield delta
    except Exception:
        # Keep behavior robust by falling back to non-stream turn.
        result = _chat_via_pipeline(user_id, message)
        yield result.get("response", "")
        return

    response = strip_thinking(streamed_text) if config.llm_strip_thinking else streamed_text
    response = response or "I'm here and ready to help."

    out_safe, _ = safety.check_output(response)
    if not out_safe:
        response = "I generated a response but it didn't pass safety checks. Let me try again differently."

    sessions.add(user_id, "user", message)
    sessions.add(user_id, "assistant", response)
    audit.log("chat_turn", f"user={user_id}|stream=1")


try:
    ws_handler = AvatarWebSocketHandler(
        state_machine=state_machine,
        emotion_mapper=emotion_mapper,
        tts_service=tts_service,
        stt_service=stt_service,
        chat_func=_chat_via_pipeline,
        stream_chat_func=_stream_chat_via_pipeline,
        filler_service=filler_service,
        rate_limiter=limiter,
    )
except TypeError:
    # Backward-compat path if an older handler signature is loaded.
    ws_handler = AvatarWebSocketHandler(
        state_machine=state_machine,
        emotion_mapper=emotion_mapper,
        tts_service=tts_service,
        stt_service=stt_service,
        chat_func=_chat_via_pipeline,
        stream_chat_func=_stream_chat_via_pipeline,
    )


# --- Request/Response models ---

class ChatTurnIn(BaseModel):
    user_id: str
    message: str
    style: str | None = None  # "concise" (default) or "thoughtful"


class MemoryWriteIn(BaseModel):
    kind: str
    key: str
    value: str
    metadata: dict | None = None


class MemoryDeleteIn(BaseModel):
    kind: str
    key: str


class PersonaUpdateIn(BaseModel):
    name: str | None = None
    occupation: str | None = None
    age: int | None = None
    backstory: str | None = None


class ToolExecIn(BaseModel):
    tool: str
    params: dict | None = None
    user_confirmed: bool = False


class ToolConfirmIn(BaseModel):
    confirm_id: str


class ReminderIn(BaseModel):
    message: str
    minutes: int = 30


class SynthesizeIn(BaseModel):
    text: str


class EmotionIn(BaseModel):
    text: str


# --- Endpoints ---

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": config.app_name, "phase": 2}


@app.get("/metrics")
def metrics(request: Request) -> Response:
    """Prometheus exposition endpoint. 404 when metrics are disabled; 401 when
    METRICS_AUTH_TOKEN is set and the bearer token is missing/wrong. Unset token
    (the default) leaves the endpoint open, unchanged from before."""
    if not config.metrics_enabled:
        return Response(status_code=404)
    token = config.metrics_auth_token
    if token and request.headers.get("authorization", "") != f"Bearer {token}":
        return Response(status_code=401)
    body, content_type = obs_metrics.render()
    return Response(content=body, media_type=content_type)


@app.post("/chat/turn")
def chat_turn(body: ChatTurnIn, request: Request = None) -> dict:
    # --- Per-caller rate limiting (no-op unless Redis is configured + reachable) ---
    # Enforced ONLY at this public HTTP boundary. Internal callers reach the chat
    # pipeline via _run_chat_turn (e.g. the WS path, which already consumes a
    # token in app/ws/handler.py), so a single turn never double-decrements the
    # bucket regardless of transport / streaming / provider configuration.
    if config.rate_limit_enabled:
        # Key on the caller's IP, not the client-supplied user_id (which a caller
        # could rotate to dodge the limit). Fall back to user_id for internal /
        # test calls with no request. Behind a proxy, prefer X-Forwarded-For.
        rl_id = request.client.host if (request is not None and request.client) else body.user_id
        allowed, retry_after = limiter.allow(rl_id)
        if not allowed:
            obs_metrics.inc_rate_limited()
            return {"ok": False, "error": "rate_limited", "retry_after": retry_after}

    # Instrument the turn at the HTTP boundary (the WS voice path emits these
    # from app/ws/handler.py). Time + count every processed /chat/turn so a plain
    # HTTP / curl demo moves /metrics, not only the voice path.
    started = time.perf_counter()
    result = _run_chat_turn(body)
    obs_metrics.observe_chat(time.perf_counter() - started)
    obs_metrics.inc_turn()
    return result


def _run_chat_turn(body: ChatTurnIn) -> dict:
    """Core chat pipeline shared by the /chat/turn endpoint and internal callers.

    Rate limiting lives in the ``chat_turn`` endpoint wrapper, NOT here, so the
    WS pipeline (``_chat_via_pipeline`` -> here) does not re-consume a token that
    ``AvatarWebSocketHandler._process_text_chat`` already consumed for the turn.
    """
    # --- Input safety check ---
    safe, refusal = safety.check(body.message)
    if not safe:
        audit.log("safety_refusal", body.message)
        return {"ok": False, "response": refusal}

    maybe_capture_user_name(body.user_id, body.message)

    message_l = body.message.lower()

    # --- Fast deterministic answers ---
    if any(k in message_l for k in ["my name", "who am i", "do you remember my name", "我的名字", "我叫什么"]):
        remembered_name = recall_user_name(body.user_id)
        if remembered_name:
            response = persona.enforce(f"Your name is {remembered_name}.")
            sessions.add(body.user_id, "user", body.message)
            sessions.add(body.user_id, "assistant", response)
            audit.log("chat_turn", f"user={body.user_id}")
            return {"ok": True, "response": response, "model": "rule-memory"}

    if any(k in message_l for k in ["your background", "about yourself", "your story", "who are you"]):
        response = persona.enforce(persona.short_background())
        sessions.add(body.user_id, "user", body.message)
        sessions.add(body.user_id, "assistant", response)
        audit.log("chat_turn", f"user={body.user_id}")
        return {"ok": True, "response": response, "model": "rule-persona"}

    # --- Memory retrieval with curation ---
    recalled = memory.search(body.message)
    scored = [
        ScoredMemory(
            kind=m.kind,
            key=m.key,
            value=m.value,
            created_at=m.created_at,
            score=retrieval_score(m.kind, m.key, m.value, m.created_at, body.message),
            source=m.source,
        )
        for m in recalled
    ]
    ranked = rank_memories(scored, limit=8)
    memory_context = "\n".join([f"- [{m.kind}] {m.key}={m.value} (score={m.score})" for m in ranked])

    # --- Conversation history ---
    history = sessions.get(body.user_id, limit=10)

    # --- Response style ---
    style_hint = ""
    if body.style == "thoughtful":
        style_hint = "\nRespond in a detailed, thoughtful manner with reasoning."
    else:
        style_hint = "\nRespond concisely."

    # --- LLM call with tool-calling loop ---
    try:
        llm_text, model_used, tool_actions = _tool_call_loop(
            body.message + style_hint, memory_context, history
        )
        response = persona.enforce(llm_text or "I'm here and ready to help.")
    except Exception as exc:
        import traceback
        print(f"[LLM ERROR] {exc}")
        traceback.print_exc()
        memory_hint = f" | memory_hits={len(ranked)}" if ranked else ""
        response = persona.enforce(f"I got it: {body.message}{memory_hint}")
        model_used = "fallback-rule"
        tool_actions = []

    # --- Output safety check ---
    out_safe, out_refusal = safety.check_output(response)
    if not out_safe:
        response = persona.enforce("I generated a response but it didn't pass safety checks. Let me try again differently.")
        audit.log("output_safety_refusal", f"user={body.user_id}")

    # --- Record history ---
    sessions.add(body.user_id, "user", body.message)
    sessions.add(body.user_id, "assistant", response)
    audit.log("chat_turn", f"user={body.user_id}")

    result: dict = {"ok": True, "response": response, "model": model_used}
    if tool_actions:
        result["tool_actions"] = tool_actions
    return result


@app.delete("/chat/history/{user_id}")
def chat_history_clear(user_id: str) -> dict:
    sessions.clear(user_id)
    return {"ok": True}


@app.get("/chat/history/{user_id}")
def chat_history_get(user_id: str, limit: int = 20) -> dict:
    return {"ok": True, "history": sessions.get(user_id, limit=limit)}


# --- Persona ---

@app.get("/persona/profile")
def persona_profile() -> dict:
    llm_model = config.ollama_model if config.llm_provider == "ollama" else config.openai_compat_model
    return {
        "ok": True,
        "profile": persona.as_dict(),
        "llm_provider": config.llm_provider,
        "llm_model": llm_model,
    }


@app.patch("/persona/profile")
def persona_update(body: PersonaUpdateIn) -> dict:
    p = persona.update_profile(
        name=body.name, occupation=body.occupation,
        age=body.age, backstory=body.backstory,
    )
    audit.log("persona_update", f"name={p.name},occupation={p.occupation},age={p.age}")
    return {"ok": True, "profile": persona.as_dict()}


# --- Memory ---

@app.post("/memory/write")
def memory_write(body: MemoryWriteIn) -> dict:
    # Consent check: refuse to store secrets unless kind is explicitly "secret"
    if body.kind != "secret" and any(w in body.value.lower() for w in ["password", "api_key", "secret_key", "token"]):
        audit.log("memory_write_refused", f"secret-detected:{body.key}")
        return {"ok": False, "error": "Detected possible secret. Use kind='secret' to explicitly store secrets."}
    item = memory.write(body.kind, body.key, body.value, metadata=body.metadata)
    if item.metadata.get("filtered"):
        audit.log("memory_write_filtered", f"{body.kind}:{body.key}")
        return {"ok": False, "error": "Memory entry filtered as noise or duplicate.", "item": item.__dict__}
    audit.log("memory_write", f"{body.kind}:{body.key}")
    return {"ok": True, "item": item.__dict__}


@app.get("/memory/search")
def memory_search(query: str) -> dict:
    items = [i.__dict__ for i in memory.search(query)]
    return {"ok": True, "items": items}


@app.delete("/memory/delete")
def memory_delete(body: MemoryDeleteIn) -> dict:
    removed = memory.delete(body.kind, body.key)
    if removed:
        audit.log("memory_delete", f"{body.kind}:{body.key}")
    return {"ok": removed}


@app.delete("/memory/erase")
def memory_erase(confirm: bool = False) -> dict:
    if not confirm:
        return {"ok": False, "error": "Confirmation required: pass ?confirm=true"}
    removed = memory.erase_all()
    audit.log("memory_erase", f"count={removed}")
    return {"ok": True, "removed": removed}


@app.get("/memory/backends")
def memory_backends() -> dict:
    return {
        "ok": True,
        "postgres": bool(memory.postgres and memory.postgres.available()),
        "neo4j": bool(memory.neo4j and memory.neo4j.available()),
        "fallback": "file-persisted",
        "persist_path": str(memory.persist_path),
    }


@app.get("/memory/debug")
def memory_debug(user_id: str = "web_user") -> dict:
    names = [
        i.__dict__
        for i in memory.search("name")
        if i.key.lower() == "name"
    ]
    return {
        "ok": True,
        "persist_path": str(memory.persist_path),
        "persist_exists": memory.persist_path.exists(),
        "name_entries": names,
        "recalled_name": recall_user_name(user_id),
        "user_id": user_id,
    }


# --- Tools ---

@app.get("/tools/list")
def tools_list() -> dict:
    return {"ok": True, "tools": tools.list_tools()}


@app.post("/tools/execute")
def tools_execute(body: ToolExecIn) -> dict:
    result = tools.execute(body.tool, body.params, user_confirmed=body.user_confirmed)
    action = "tool_confirm" if result.get("needs_confirmation") else "tool_exec"
    audit.log(action, f"{body.tool}")
    return result


@app.post("/tools/confirm")
def tools_confirm(body: ToolConfirmIn) -> dict:
    result = tools.confirm(body.confirm_id)
    audit.log("tool_confirm_approved", body.confirm_id)
    return result


# --- Proactivity ---

@app.get("/proactivity/check")
def proactivity_check() -> dict:
    now = datetime.now()
    can_send, reason = proactivity.can_send(now)
    due = proactivity.check_due()
    return {
        "ok": True, "can_send": can_send, "reason": reason,
        "why_now": "scheduled-check",
        "due_reminders": [{"id": r.id, "message": r.message} for r in due],
    }


@app.post("/proactivity/mark-sent")
def proactivity_mark_sent() -> dict:
    proactivity.mark_sent(datetime.now())
    audit.log("proactive_send", "marked sent")
    return {"ok": True}


@app.post("/reminders/add")
def reminders_add(body: ReminderIn) -> dict:
    r = proactivity.add_reminder(message=body.message, minutes=body.minutes)
    audit.log("reminder_add", f"id={r.id} msg={r.message}")
    return {"ok": True, "reminder": {"id": r.id, "message": r.message, "due_at": r.due_at.isoformat()}}


@app.get("/reminders/list")
def reminders_list(include_fired: bool = False) -> dict:
    return {"ok": True, "reminders": proactivity.list_reminders(include_fired=include_fired)}


@app.delete("/reminders/{reminder_id}")
def reminders_cancel(reminder_id: str) -> dict:
    ok = proactivity.cancel_reminder(reminder_id)
    if ok:
        audit.log("reminder_cancel", reminder_id)
    return {"ok": ok}


@app.get("/daily-summary")
def daily_summary() -> dict:
    recent = memory.search("")
    mem_items = [{"kind": m.kind, "key": m.key, "value": m.value} for m in recent[:20]]
    summary = proactivity.daily_summary(memory_items=mem_items)
    return {"ok": True, "summary": summary}


# --- Avatar + Voice ---

@app.get("/avatar/config")
def avatar_config() -> dict:
    model_path = config.avatar_model_path
    model_exists = os.path.exists(model_path)
    return {
        "ok": True,
        "avatar_model_path": model_path,
        "live2d_enabled": bool(config.avatar_live2d_enabled),
        "model_path_exists": model_exists,
    }


@app.get("/avatar/state")
def avatar_state() -> dict:
    snap = state_machine.snapshot()
    return {"ok": True, "state": snap.state, "since": snap.since, "expression": snap.expression}


@app.post("/avatar/emotion")
def avatar_emotion(body: EmotionIn) -> dict:
    result = emotion_mapper.analyze(body.text)
    return {
        "ok": True,
        "category": result.category,
        "intensity": result.intensity,
        "expression": result.expression,
        "transition_ms": result.transition_ms,
    }


@app.post("/voice/synthesize")
def voice_synthesize(body: SynthesizeIn) -> dict:
    audio_bytes, phonemes = tts_service.synthesize(body.text)
    visemes = build_viseme_timeline(phonemes)
    return {
        "ok": True,
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "phoneme_timestamps": phonemes,
        "viseme_timeline": visemes,
    }


@app.get("/voice/stt-status")
def voice_stt_status() -> dict:
    status_fn = getattr(stt_service, "status", None)
    runtime_status = status_fn() if callable(status_fn) else {}
    status = {
        "loaded": runtime_status.get("loaded", False),
        "device": runtime_status.get("device", getattr(config, "stt_device", "cpu")),
        "compute_type": runtime_status.get("compute_type", "unknown"),
        "language_hint": runtime_status.get("language_hint", getattr(config, "stt_language_hint", "zh")),
        "warmup_started": runtime_status.get("warmup_started", False),
        "warmup_enabled": bool(getattr(config, "stt_warmup_enabled", True)),
        "model_size": getattr(config, "stt_model_size", "small"),
    }
    return {"ok": True, "stt": status}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    if not config.ws_enabled:
        await websocket.close(code=1013)
        return
    await ws_handler.handle(websocket)


# --- Audit ---

@app.get("/audit/events")
def audit_events() -> dict:
    return {"ok": True, "events": audit.list_events()}

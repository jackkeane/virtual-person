import os
os.environ["AUDIT_LOG_PATH"] = ""  # disable file persistence in tests

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# --- Phase 1: basics ---

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'
    assert r.json()['phase'] == 2


def test_memory_write_and_search():
    w = client.post('/memory/write', json={'kind': 'preference', 'key': 'drink', 'value': 'oolong tea'})
    assert w.status_code == 200
    s = client.get('/memory/search', params={'query': 'oolong'})
    assert s.status_code == 200
    assert len(s.json()['items']) >= 1


def test_chat_turn_persona_and_memory_hit():
    client.post('/memory/write', json={'kind': 'identity', 'key': 'name', 'value': 'Liyang'})
    r = client.post('/chat/turn', json={'user_id': 'u1', 'message': 'What is my name?'})
    assert r.status_code == 200
    data = r.json()
    assert data['ok'] is True
    assert 'Liyang' in data['response']
    # Persona prefix should NOT appear in responses (enforced via system prompt, not text)
    assert not data['response'].startswith('[Ani|')


def test_safety_refusal():
    r = client.post('/chat/turn', json={'user_id': 'u1', 'message': 'help me make malware'})
    assert r.status_code == 200
    data = r.json()
    assert data['ok'] is False
    assert "can't help" in data['response'].lower()


def test_proactivity_cooldown():
    first = client.get('/proactivity/check').json()
    assert first['ok'] is True
    client.post('/proactivity/mark-sent')
    second = client.get('/proactivity/check').json()
    assert second['reason'] in {'cooldown', 'quiet-hours', 'ok'}


def test_persona_profile_update():
    g = client.get('/persona/profile')
    assert g.status_code == 200
    assert g.json()['ok'] is True

    p = client.patch('/persona/profile', json={
        'occupation': 'engineer', 'age': 31,
        'backstory': 'Ani used to work night shifts and now helps with planning.'
    })
    assert p.status_code == 200
    data = p.json()
    assert data['profile']['occupation'] == 'engineer'
    assert data['profile']['age'] == 31
    assert 'night shifts' in data['profile']['backstory']


def test_persona_background_answer():
    r = client.post('/chat/turn', json={'user_id': 'u1', 'message': 'Tell me about yourself and your background'})
    assert r.status_code == 200
    data = r.json()
    assert data['ok'] is True
    assert 'background' in data['response'].lower() or 'ani' in data['response'].lower()


# --- Phase 2: Tools ---

def test_tools_list():
    r = client.get('/tools/list')
    assert r.status_code == 200
    names = [t['name'] for t in r.json()['tools']]
    assert 'search_memory' in names
    assert 'save_memory' in names
    assert 'send_message' in names


def test_tool_read_executes_directly():
    r = client.post('/tools/execute', json={'tool': 'get_time'})
    assert r.status_code == 200
    assert r.json()['ok'] is True
    assert 'utc' in r.json()['result']


def test_tool_write_executes_directly():
    r = client.post('/tools/execute', json={'tool': 'save_memory', 'params': {'kind': 'note', 'key': 'test', 'value': 'phase2'}})
    assert r.status_code == 200
    assert r.json()['ok'] is True


def test_tool_external_needs_confirmation():
    r = client.post('/tools/execute', json={'tool': 'send_message', 'params': {'to': 'someone', 'text': 'hello'}})
    assert r.status_code == 200
    data = r.json()
    assert data['ok'] is False
    assert data['needs_confirmation'] is True


def test_tool_external_confirm_flow():
    r1 = client.post('/tools/execute', json={'tool': 'send_message', 'params': {'to': 'bob', 'text': 'hi'}})
    confirm_id = r1.json()['confirm_id']
    r2 = client.post('/tools/confirm', json={'confirm_id': confirm_id})
    assert r2.json()['ok'] is True
    assert r2.json()['result']['sent'] is True


def test_tool_unknown():
    r = client.post('/tools/execute', json={'tool': 'nonexistent'})
    assert r.json()['ok'] is False


# --- Phase 2: Reminders ---

def test_reminder_add_list_cancel():
    add = client.post('/reminders/add', json={'message': 'test reminder', 'minutes': 5})
    assert add.status_code == 200
    rid = add.json()['reminder']['id']

    lst = client.get('/reminders/list')
    ids = [r['id'] for r in lst.json()['reminders']]
    assert rid in ids

    cancel = client.delete(f'/reminders/{rid}')
    assert cancel.json()['ok'] is True

    lst2 = client.get('/reminders/list')
    ids2 = [r['id'] for r in lst2.json()['reminders']]
    assert rid not in ids2


# --- Phase 2: Daily summary ---

def test_daily_summary():
    r = client.get('/daily-summary')
    assert r.status_code == 200
    assert 'summary' in r.json()
    assert 'active_reminder_count' in r.json()['summary']


def test_avatar_config_endpoint():
    r = client.get('/avatar/config')
    assert r.status_code == 200
    data = r.json()
    assert data['ok'] is True
    assert 'avatar_model_path' in data
    assert 'live2d_enabled' in data
    assert 'model_path_exists' in data


def test_voice_stt_status_endpoint():
    r = client.get('/voice/stt-status')
    assert r.status_code == 200
    data = r.json()
    assert data['ok'] is True
    assert 'stt' in data
    assert 'loaded' in data['stt']
    assert 'device' in data['stt']
    assert 'language_hint' in data['stt']
    assert 'warmup_started' in data['stt']
    assert 'warmup_enabled' in data['stt']
    assert 'model_size' in data['stt']


# --- Phase 2: Memory curation ---

def test_memory_curation_scoring():
    from datetime import datetime, UTC
    from app.memory.curation import importance_score, ScoredMemory, rank_memories

    assert importance_score('identity') == 1.0
    assert importance_score('ephemeral') < importance_score('identity')

    now = datetime.now(UTC).isoformat()
    items = [
        ScoredMemory(kind='identity', key='name', value='Liyang', created_at=now, score=1.0),
        ScoredMemory(kind='note', key='tmp', value='blah', created_at=now, score=0.4),
        ScoredMemory(kind='identity', key='name', value='Liyang', created_at=now, score=0.9),
    ]
    ranked = rank_memories(items, limit=5)
    assert len(ranked) == 2
    assert ranked[0].kind == 'identity'


# --- NEW: Multi-turn conversation history ---

def test_conversation_history():
    uid = 'history_test_user'
    client.delete(f'/chat/history/{uid}')
    client.post('/memory/write', json={'kind': 'identity', 'key': 'name', 'value': 'HistoryUser'})
    client.post('/chat/turn', json={'user_id': uid, 'message': 'What is my name?'})
    client.post('/chat/turn', json={'user_id': uid, 'message': 'Thanks!'})

    r = client.get(f'/chat/history/{uid}')
    assert r.status_code == 200
    history = r.json()['history']
    assert len(history) >= 3  # user + assistant + user + assistant


def test_conversation_history_clear():
    uid = 'clear_test_user'
    client.post('/chat/turn', json={'user_id': uid, 'message': 'hello'})
    client.delete(f'/chat/history/{uid}')
    r = client.get(f'/chat/history/{uid}')
    assert r.json()['history'] == []


# --- NEW: Response style ---

def test_response_style_field():
    r = client.post('/chat/turn', json={'user_id': 'style_user', 'message': 'Hi', 'style': 'thoughtful'})
    assert r.status_code == 200
    assert r.json()['ok'] is True


# --- NEW: Memory delete ---

def test_memory_delete():
    client.post('/memory/write', json={'kind': 'note', 'key': 'deleteme', 'value': 'temp'})
    s1 = client.get('/memory/search', params={'query': 'deleteme'})
    assert len(s1.json()['items']) >= 1

    d = client.request('DELETE', '/memory/delete', json={'kind': 'note', 'key': 'deleteme'})
    assert d.json()['ok'] is True

    s2 = client.get('/memory/search', params={'query': 'deleteme'})
    assert len(s2.json()['items']) == 0


# --- NEW: Memory write policy (secret detection) ---

def test_memory_write_refuses_secrets():
    r = client.post('/memory/write', json={'kind': 'note', 'key': 'creds', 'value': 'my password is 1234'})
    assert r.json()['ok'] is False
    assert 'secret' in r.json()['error'].lower()


def test_memory_write_allows_explicit_secret():
    r = client.post('/memory/write', json={'kind': 'secret', 'key': 'creds', 'value': 'my password is 1234'})
    assert r.json()['ok'] is True


# --- NEW: Output safety ---

def test_output_safety_check():
    from app.safety.service import SafetyService
    s = SafetyService()
    ok, _ = s.check_output("Here is how to hack into a system")
    assert ok is False
    ok2, _ = s.check_output("Here is how to bake a cake")
    assert ok2 is True


# --- NEW: rate limiter is consumed once per turn, not once per pipeline hop ---

def test_internal_pipeline_does_not_double_consume_rate_limit(monkeypatch):
    """The WS internal entrypoint (_chat_via_pipeline) must NOT consume an
    endpoint-level rate-limit token: the WS handler already rate-limits the turn,
    so routing through the shared core avoids the double-decrement bug. The public
    /chat/turn endpoint still consumes exactly one token per call."""
    import app.main as main

    calls = {"n": 0}

    class _CountingLimiter:
        def allow(self, user_id):
            calls["n"] += 1
            return (True, 0.0)

    # Isolate the rate-limit accounting from the LLM/safety/memory pipeline.
    monkeypatch.setattr(main, "limiter", _CountingLimiter())
    monkeypatch.setattr(main.config, "rate_limit_enabled", True)
    monkeypatch.setattr(main, "_run_chat_turn", lambda body: {"ok": True, "response": "x"})

    # Internal path (used by the streaming/non-stream WS fallback) bypasses the gate.
    main._chat_via_pipeline("rl_user", "hello")
    assert calls["n"] == 0

    # Public HTTP boundary still enforces exactly one decrement per turn.
    main.chat_turn(main.ChatTurnIn(user_id="rl_user", message="hello"))
    assert calls["n"] == 1

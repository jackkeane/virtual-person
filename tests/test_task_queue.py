"""Tests for the RabbitMQ background task queue (Feature: async background jobs).

Coverage mirrors the Redis seams' test style (``test_infra_redis``,
``test_rate_limit``, ``test_session_redis``):

* GATING (always runs, no broker) -- with ``AMQP_URL`` unset the queue is INERT:
  :func:`get_task_queue` returns ``None``, :func:`queue_available` is ``False``,
  and both the transport :func:`enqueue` and every typed producer are no-ops
  returning ``False``. This is why the default suite stays byte-identical to
  today. The disabled path is also proven NOT to import ``pika``.
* ENVELOPE (always runs, no broker) -- the JSON envelope round-trips through
  ``_encode``/``json`` faithfully (incl. non-ASCII) and the credential redactor
  hides ``user:pass@``.
* BROKER INTEGRATION (skipped unless ``AMQP_URL`` is set) -- real
  enqueue -> consume -> ack, handler-failure retry -> DLQ after max retries, and
  at-least-once redelivery. These import ``pika`` INSIDE the test body and use
  EPHEMERAL, uniquely-named ``vp.test.*`` topology that is torn down afterward;
  they NEVER touch the app's ``vp.tasks.*`` queues.

``pika`` is NOT imported at module top level: it is absent from the default test
environment, so importing it here would break collection.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime

import pytest

from app.config import config
from app.queue import jobs
from app.queue.task_queue import (
    RabbitMQTaskQueue,
    _encode,
    _new_envelope,
    _redact,
    enqueue,
    get_task_queue,
    queue_available,
    reset_task_queue_cache,
)

AMQP_URL = os.getenv("AMQP_URL", "")
_SKIP_REASON = "AMQP_URL not set; RabbitMQ broker integration tests skipped"
# Well-formed but intentionally dead endpoint (nothing listens on port 1): a
# connect attempt refuses instantly, so gating tests stay fast AND never reach a
# real broker / declare the app's vp.tasks.* topology.
_DEAD_URL = "amqp://guest:guest@127.0.0.1:1/"


# --------------------------------------------------------------------------- #
# (a) GATING -- inert by default. These MUST run and pass in the default suite. #
# --------------------------------------------------------------------------- #

def test_get_task_queue_none_without_amqp_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """No AMQP_URL -> get_task_queue()/queue_available() inert, enqueue() a no-op.

    queue_enabled is left true to prove the SECOND gate (empty amqp_url) is what
    keeps the queue dormant, mirroring the Redis ``redis_url``-empty fallback.
    """
    monkeypatch.delenv("AMQP_URL", raising=False)
    monkeypatch.setattr(config, "queue_enabled", True)
    monkeypatch.setattr(config, "amqp_url", "")
    reset_task_queue_cache()
    try:
        assert get_task_queue() is None
        assert queue_available() is False
        assert enqueue("curate_memories", {"user_id": "u"}) is False
        assert enqueue("poison") is False
    finally:
        reset_task_queue_cache()


def test_typed_producers_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every jobs.enqueue_* helper is self-gating -> False when the queue is off."""
    monkeypatch.setattr(config, "queue_enabled", True)
    monkeypatch.setattr(config, "amqp_url", "")
    reset_task_queue_cache()
    try:
        assert jobs.enqueue_curate_memories("u1") is False
        assert jobs.enqueue_daily_summary_precompute() is False
        assert jobs.enqueue_proactive_nudge() is False
        assert jobs.enqueue_poison() is False
    finally:
        reset_task_queue_cache()


def test_queue_enabled_flag_gates_even_with_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Double gate: queue_enabled=False stays inert even with a URL, and the flag
    short-circuits BEFORE importing pika or opening a socket (mirrors
    ``session_backend='memory'`` skipping the Redis ping)."""
    monkeypatch.setattr(config, "amqp_url", _DEAD_URL)
    monkeypatch.setattr(config, "queue_enabled", False)
    reset_task_queue_cache()
    had_pika = "pika" in sys.modules
    try:
        assert get_task_queue() is None
        assert queue_available() is False
        assert enqueue("curate_memories") is False
        # Gated off by the flag -> no pika import, no connect attempt.
        assert ("pika" in sys.modules) == had_pika
    finally:
        reset_task_queue_cache()


def test_disabled_path_never_imports_pika(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inert (no-URL) path must not import pika. Uses a before/after delta so
    the assertion is order-independent even in the CI job where a broker test may
    have legitimately imported pika earlier."""
    monkeypatch.setattr(config, "queue_enabled", True)
    monkeypatch.setattr(config, "amqp_url", "")
    reset_task_queue_cache()
    had_pika = "pika" in sys.modules
    try:
        assert get_task_queue() is None
        assert enqueue("curate_memories", {"user_id": "u"}) is False
        assert jobs.enqueue_poison() is False
        assert ("pika" in sys.modules) == had_pika
    finally:
        reset_task_queue_cache()


def test_reset_cache_clears_and_reevaluates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached None is served on the hot path (no per-call reconnect); only
    reset_task_queue_cache() forces a fresh evaluation."""
    import app.queue.task_queue as tq

    monkeypatch.setattr(config, "queue_enabled", True)
    monkeypatch.setattr(config, "amqp_url", "")
    reset_task_queue_cache()
    try:
        assert get_task_queue() is None
        assert tq._attempted is True  # outcome is now cached

        # Flip config WITHOUT resetting: the cached None is still served, so the
        # hot path never re-pays a connect timeout.
        monkeypatch.setattr(config, "amqp_url", _DEAD_URL)
        assert get_task_queue() is None

        # reset clears the cache back to its pristine state.
        reset_task_queue_cache()
        assert tq._queue is None and tq._attempted is False
    finally:
        reset_task_queue_cache()


def test_rabbitmq_queue_enqueue_fail_soft_without_broker() -> None:
    """A directly-constructed queue whose channel can't be established (pika
    missing OR broker down) must enqueue -> False and .available -> False without
    ever raising. In the default env pika is absent, so this exercises the missing
    -pika branch; the ephemeral vp.test.* names are never declared (no connect)."""
    q = RabbitMQTaskQueue(
        _DEAD_URL,
        exchange="vp.test.ex",
        queue="vp.test.q",
        routing_key="vp.test.rk",
        dlx="vp.test.dlx",
        dlq="vp.test.dlq",
    )
    try:
        assert q.enqueue("probe", {"n": 1}) is False
        assert q.available is False
    finally:
        q.close()  # idempotent no-op on a never-connected queue


# --------------------------------------------------------------------------- #
# (b) ENVELOPE serialize / roundtrip (no broker).                             #
# --------------------------------------------------------------------------- #

def test_envelope_shape_and_roundtrip() -> None:
    env = _new_envelope("curate_memories", {"user_id": "u1"})
    assert set(env.keys()) == {"id", "type", "payload", "attempts", "enqueued_at"}
    assert env["type"] == "curate_memories"
    assert env["payload"] == {"user_id": "u1"}
    assert env["attempts"] == 0
    # id is a real UUID; enqueued_at is ISO-8601 and parses back.
    uuid.UUID(env["id"])
    datetime.fromisoformat(env["enqueued_at"])

    raw = _encode(env)
    assert isinstance(raw, bytes)
    assert json.loads(raw) == env  # bytes -> json round-trips to an equal dict


def test_envelope_defaults_payload_and_unique_ids() -> None:
    a = _new_envelope("poison", None)
    b = _new_envelope("poison", None)
    assert a["payload"] == {}  # None payload defaults to an empty dict
    assert a["id"] != b["id"]  # a fresh uuid per envelope


def test_encode_preserves_non_ascii() -> None:
    """ensure_ascii=False: unicode survives the utf-8 round trip byte-for-byte
    (not \\uXXXX-escaped) -- important for the CN-facing payloads."""
    env = _new_envelope("note", {"text": "楼立洋 你好"})
    raw = _encode(env)
    assert "楼立洋".encode("utf-8") in raw
    assert json.loads(raw.decode("utf-8"))["payload"]["text"] == "楼立洋 你好"


def test_redact_hides_credentials() -> None:
    assert _redact("amqp://guest:guest@host:5672/") == "amqp://***@host:5672/"
    assert _redact("amqp://host:5672/") == "amqp://host:5672/"  # nothing to hide
    assert _redact("") == ""


# --------------------------------------------------------------------------- #
# (c) BROKER INTEGRATION -- skipped unless AMQP_URL is set. pika is imported    #
# INSIDE each test body; every test uses EPHEMERAL vp.test.* topology that is   #
# torn down afterward and NEVER the app vp.tasks.* queues.                      #
# --------------------------------------------------------------------------- #

def _broker_reachable() -> bool:
    """True iff a real RabbitMQ answers at AMQP_URL. Imports pika lazily."""
    if not AMQP_URL:
        return False
    try:
        import pika

        params = pika.URLParameters(AMQP_URL)
        params.socket_timeout = 2.0
        params.connection_attempts = 1
        conn = pika.BlockingConnection(params)
        conn.close()
        return True
    except Exception:
        return False


def _ephemeral_names() -> dict:
    """Unique vp.test.* topology names so parallel/repeat runs never collide."""
    sfx = uuid.uuid4().hex[:8]
    return {
        "exchange": f"vp.test.ex.{sfx}",
        "queue": f"vp.test.q.{sfx}",
        "routing_key": f"vp.test.rk.{sfx}",
        "dlx": f"vp.test.dlx.{sfx}",
        "dlq": f"vp.test.dlq.{sfx}",
    }


def _make_queue(names: dict, **over) -> RabbitMQTaskQueue:
    kwargs: dict = dict(names)
    kwargs.setdefault("max_retries", 3)
    kwargs.setdefault("prefetch", 10)
    kwargs.update(over)
    return RabbitMQTaskQueue(AMQP_URL, **kwargs)


def _msg_count(queue_name: str) -> int:
    """Current ready-message count via a passive declare on a fresh connection."""
    import pika

    conn = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
    try:
        ch = conn.channel()
        res = ch.queue_declare(queue=queue_name, passive=True)
        return res.method.message_count
    finally:
        conn.close()


def _teardown(names: dict) -> None:
    """Delete every ephemeral vp.test.* exchange/queue this test declared."""
    try:
        import pika

        conn = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
        ch = conn.channel()
        for q in (names["queue"], names["dlq"]):
            try:
                ch.queue_delete(queue=q)
            except Exception:
                pass
        for ex in (names["exchange"], names["dlx"]):
            try:
                ch.exchange_delete(exchange=ex)
            except Exception:
                pass
        conn.close()
    except Exception:
        pass


@pytest.mark.skipif(not os.getenv("AMQP_URL"), reason=_SKIP_REASON)
def test_enqueue_consume_ack_happy_path() -> None:
    import pika  # noqa: F401  # lazily, inside the test body per the queue's contract

    if not _broker_reachable():
        pytest.skip("AMQP_URL set but broker unreachable")

    names = _ephemeral_names()
    q = _make_queue(names)
    seen: list = []
    handlers = {"probe": lambda payload: seen.append(payload)}
    try:
        assert q.enqueue("probe", {"n": 1}) is True
        processed = q.consume(handlers=handlers, max_messages=1, inactivity_timeout=5.0)
        assert processed == 1
        assert seen == [{"n": 1}]
        # Acked -> work queue drained, nothing dead-lettered.
        assert _msg_count(names["queue"]) == 0
        assert _msg_count(names["dlq"]) == 0
    finally:
        q.close()
        _teardown(names)


@pytest.mark.skipif(not os.getenv("AMQP_URL"), reason=_SKIP_REASON)
def test_handler_failure_retries_then_dlq() -> None:
    import pika

    if not _broker_reachable():
        pytest.skip("AMQP_URL set but broker unreachable")

    names = _ephemeral_names()
    # max_retries=2: attempt1 -> requeue (attempts=1), attempt2 -> DLQ (attempts=2).
    q = _make_queue(names, max_retries=2)
    calls: list = []

    def boom(payload):
        calls.append(payload)
        raise RuntimeError("always fails")

    handlers = {"boom": boom}
    try:
        assert q.enqueue("boom", {"k": "v"}) is True

        # Retries are republished to the SAME work queue, so drain in a bounded
        # loop until the poison lands in the DLQ (robust to how many redeliveries
        # surface per consume() call).
        deadline = time.time() + 15.0
        while _msg_count(names["dlq"]) < 1 and time.time() < deadline:
            q.consume(handlers=handlers, max_messages=5, inactivity_timeout=1.0)

        assert _msg_count(names["dlq"]) == 1
        assert len(calls) == 2  # exactly max_retries handler invocations
        assert _msg_count(names["queue"]) == 0

        # Inspect the dead-lettered envelope: reason header + final attempt count.
        conn = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
        try:
            ch = conn.channel()
            method, props, body = ch.basic_get(queue=names["dlq"], auto_ack=True)
            assert method is not None
            env = json.loads(body)
            assert env["type"] == "boom"
            assert env["attempts"] == 2
            assert (props.headers or {}).get("x-dlq-reason") == "max-retries"
        finally:
            conn.close()
    finally:
        q.close()
        _teardown(names)


@pytest.mark.skipif(not os.getenv("AMQP_URL"), reason=_SKIP_REASON)
def test_at_least_once_redelivery_until_success() -> None:
    """Fail once, succeed on redelivery: the task is retried (at-least-once) and
    ultimately completes WITHOUT dead-lettering. Because the real handlers are
    idempotent, a redelivery carrying the identical payload converges safely."""
    import pika  # noqa: F401

    if not _broker_reachable():
        pytest.skip("AMQP_URL set but broker unreachable")

    names = _ephemeral_names()
    q = _make_queue(names, max_retries=5)
    calls: list = []

    def flaky(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise RuntimeError("transient failure on first delivery")
        return {"ok": True}

    handlers = {"flaky": flaky}
    try:
        assert q.enqueue("flaky", {"id": "x"}) is True

        deadline = time.time() + 15.0
        while len(calls) < 2 and time.time() < deadline:
            q.consume(handlers=handlers, max_messages=5, inactivity_timeout=1.0)

        assert len(calls) == 2  # delivered at least twice -> a retry happened
        assert calls == [{"id": "x"}, {"id": "x"}]  # identical payload each delivery
        # Second delivery succeeded -> acked -> nothing left, nothing dead-lettered.
        assert _msg_count(names["queue"]) == 0
        assert _msg_count(names["dlq"]) == 0
    finally:
        q.close()
        _teardown(names)

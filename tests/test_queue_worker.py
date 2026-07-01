"""Consumer dispatch state-machine tests for the background task queue.

These drive ``RabbitMQTaskQueue._handle_delivery`` / ``.consume`` against a FAKE
channel, so the full retry -> DLQ ladder, unknown-type/decode-error routing, and
the consume loop's stop conditions are verified WITHOUT a broker -- they run in
the default suite.

``pika`` is not installed in the default env, so it is never imported at module
top level. The few tests that exercise the publish paths (retry / DLQ) inject a
minimal fake ``pika`` module (just ``BasicProperties``) via the ``fake_pika``
fixture, so ``_publish`` builds its properties without the real dependency.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from app.queue.task_queue import RabbitMQTaskQueue, _encode, _new_envelope


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _FakeMethod:
    """Stand-in for pika's Basic.Deliver (only delivery_tag is read)."""

    def __init__(self, delivery_tag: int = 1) -> None:
        self.delivery_tag = delivery_tag


class _FakeChannel:
    """Records publishes/acks/nacks so the dispatch decisions can be asserted."""

    def __init__(self) -> None:
        self.is_open = True
        self.published: list[dict] = []
        self.acks: list = []
        self.nacks: list = []

    def basic_publish(self, exchange, routing_key, body, properties=None, mandatory=False) -> None:
        # mandatory kw mirrors real pika's signature; the producer now passes
        # mandatory=True so an unroutable publish surfaces as UnroutableError.
        self.published.append(
            {"exchange": exchange, "routing_key": routing_key, "body": body, "properties": properties}
        )

    def basic_ack(self, delivery_tag) -> None:
        self.acks.append(delivery_tag)

    def basic_nack(self, delivery_tag, requeue) -> None:
        self.nacks.append((delivery_tag, requeue))


class _ConsumeFakeChannel(_FakeChannel):
    """Adds the surface ``consume()`` drives: basic_qos, a consume() generator
    (yielding preloaded deliveries then a ``(None, None, None)`` inactivity
    sentinel forever), and cancel()."""

    def __init__(self, deliveries) -> None:
        super().__init__()
        self._deliveries = list(deliveries)
        self.qos = None
        self.cancelled = False

    def basic_qos(self, prefetch_count) -> None:
        self.qos = prefetch_count

    def consume(self, queue, inactivity_timeout=None):
        for d in self._deliveries:
            yield d
        while True:  # exhausted -> emulate an idle broker
            yield (None, None, None)

    def cancel(self) -> None:
        self.cancelled = True


@pytest.fixture
def fake_pika(monkeypatch: pytest.MonkeyPatch):
    """Inject a minimal ``pika`` into sys.modules so ``_publish`` (which does
    ``import pika`` for BasicProperties) works without the real dependency.
    monkeypatch.setitem restores/removes the entry on teardown."""
    mod = types.ModuleType("pika")

    class BasicProperties:
        def __init__(self, delivery_mode=None, content_type=None, headers=None) -> None:
            self.delivery_mode = delivery_mode
            self.content_type = content_type
            self.headers = headers

    mod.BasicProperties = BasicProperties  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pika", mod)
    return mod


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _queue(**over) -> RabbitMQTaskQueue:
    """A queue with a dead url -- these tests never connect (no _ensure_channel)."""
    kwargs = dict(
        exchange="vp.test.ex",
        queue="vp.test.q",
        routing_key="vp.test.rk",
        dlx="vp.test.dlx",
        dlq="vp.test.dlq",
        max_retries=3,
        prefetch=10,
    )
    kwargs.update(over)
    return RabbitMQTaskQueue("amqp://unused/", **kwargs)


def _body(task_type: str, payload=None, attempts: int = 0) -> bytes:
    env = _new_envelope(task_type, payload)
    env["attempts"] = attempts
    return _encode(env)


# --------------------------------------------------------------------------- #
# Success path                                                                 #
# --------------------------------------------------------------------------- #
def test_success_acks_without_republish() -> None:
    q = _queue()
    ch = _FakeChannel()
    seen: list = []
    q._handle_delivery(ch, _FakeMethod(7), None, _body("probe", {"n": 1}), {"probe": seen.append})
    assert seen == [{"n": 1}]
    assert ch.acks == [7]
    assert ch.published == []  # success -> no retry, no DLQ
    assert ch.nacks == []


# --------------------------------------------------------------------------- #
# Retry -> DLQ ladder                                                          #
# --------------------------------------------------------------------------- #
def test_failure_republishes_with_incremented_attempts(fake_pika) -> None:
    q = _queue(max_retries=3)
    ch = _FakeChannel()

    def boom(_payload):
        raise RuntimeError("x")

    q._handle_delivery(ch, _FakeMethod(1), None, _body("boom", {}, attempts=0), {"boom": boom})

    # attempts 0 -> 1 (< max_retries): republish to the WORK exchange, then ack.
    assert ch.acks == [1]
    assert ch.nacks == []
    assert len(ch.published) == 1
    pub = ch.published[0]
    assert pub["exchange"] == q.exchange
    assert pub["routing_key"] == q.routing_key
    assert pub["properties"].delivery_mode == 2  # persistent
    assert pub["properties"].headers["x-attempts"] == 1
    assert json.loads(pub["body"])["attempts"] == 1


def test_retry_ladder_terminates_in_dlq(fake_pika) -> None:
    """Walk the full 0 -> 1 -> 2 -> DLQ sequence, one delivery at a time, each acked."""
    q = _queue(max_retries=3)

    def boom(_payload):
        raise RuntimeError("boom")

    handlers = {"boom": boom}

    # attempts=0 -> republish attempts=1 to the work exchange.
    ch = _FakeChannel()
    q._handle_delivery(ch, _FakeMethod(1), None, _body("boom", {}, 0), handlers)
    assert ch.published[0]["exchange"] == q.exchange
    assert ch.published[0]["properties"].headers["x-attempts"] == 1
    assert ch.acks == [1]

    # attempts=1 -> republish attempts=2 to the work exchange.
    ch = _FakeChannel()
    q._handle_delivery(ch, _FakeMethod(2), None, _body("boom", {}, 1), handlers)
    assert ch.published[0]["exchange"] == q.exchange
    assert ch.published[0]["properties"].headers["x-attempts"] == 2
    assert ch.acks == [2]

    # attempts=2 -> new=3 == max_retries -> DLQ via the fanout DLX (routing_key "").
    ch = _FakeChannel()
    q._handle_delivery(ch, _FakeMethod(3), None, _body("boom", {}, 2), handlers)
    assert len(ch.published) == 1
    dlq_pub = ch.published[0]
    assert dlq_pub["exchange"] == q.dlx
    assert dlq_pub["routing_key"] == ""
    assert dlq_pub["properties"].headers["x-dlq-reason"] == "max-retries"
    env = json.loads(dlq_pub["body"])
    assert env["attempts"] == 3
    assert "error" in env  # the failing repr is captured for the DLQ record
    assert ch.acks == [3]


def test_retry_publish_failure_nacks_and_requeues(fake_pika) -> None:
    """If the retry republish itself fails, the delivery is nack+requeued (not
    acked), so the broker redelivers it instead of the task being lost."""

    class _PublishBoomChannel(_FakeChannel):
        def basic_publish(self, *a, **k):
            raise RuntimeError("publish failed")

    q = _queue(max_retries=3)
    ch = _PublishBoomChannel()

    def boom(_payload):
        raise RuntimeError("handler")

    q._handle_delivery(ch, _FakeMethod(4), None, _body("boom", {}, 0), {"boom": boom})
    assert ch.nacks == [(4, True)]
    assert ch.acks == []


# --------------------------------------------------------------------------- #
# DLQ routing for non-retryable deliveries                                    #
# --------------------------------------------------------------------------- #
def test_unknown_type_routes_to_dlq(fake_pika) -> None:
    q = _queue()
    ch = _FakeChannel()
    called: list = []
    q._handle_delivery(ch, _FakeMethod(1), None, _body("nope", {"a": 1}, 0), {"probe": called.append})
    assert called == []  # unregistered type -> handler never runs
    assert len(ch.published) == 1
    assert ch.published[0]["exchange"] == q.dlx
    assert ch.published[0]["properties"].headers["x-dlq-reason"] == "no-handler"
    assert ch.acks == [1]


def test_malformed_json_routes_to_dlq_body_preserved(fake_pika) -> None:
    q = _queue()
    ch = _FakeChannel()
    bad = b"{not valid json"
    q._handle_delivery(ch, _FakeMethod(9), None, bad, {"probe": lambda p: None})
    assert len(ch.published) == 1
    pub = ch.published[0]
    assert pub["exchange"] == q.dlx
    assert pub["properties"].headers["x-dlq-reason"] == "decode-error"
    assert pub["body"] == bad  # original bytes preserved into the DLQ
    assert ch.acks == [9]


# --------------------------------------------------------------------------- #
# consume() loop stop conditions                                              #
# --------------------------------------------------------------------------- #
def test_consume_stops_at_max_messages() -> None:
    q = _queue()
    seen: list = []
    deliveries = [
        (_FakeMethod(1), None, _body("probe", {"i": 0})),
        (_FakeMethod(2), None, _body("probe", {"i": 1})),
        (_FakeMethod(3), None, _body("probe", {"i": 2})),
    ]
    ch = _ConsumeFakeChannel(deliveries)
    q._channel = ch  # is_open=True -> _ensure_channel returns it without connecting

    processed = q.consume(handlers={"probe": seen.append}, max_messages=2, inactivity_timeout=0.01)

    assert processed == 2
    assert seen == [{"i": 0}, {"i": 1}]  # stopped after 2, third never touched
    assert ch.acks == [1, 2]
    assert ch.qos == q.prefetch  # basic_qos(prefetch) applied
    assert ch.cancelled is True  # channel.cancel() in the finally block


def test_consume_stops_on_inactivity_timeout() -> None:
    q = _queue()
    seen: list = []
    deliveries = [(_FakeMethod(1), None, _body("probe", {"only": True}))]
    ch = _ConsumeFakeChannel(deliveries)  # then yields (None, None, None) -> break
    q._channel = ch

    # No max_messages: the loop relies on the inactivity sentinel to stop.
    processed = q.consume(handlers={"probe": seen.append}, inactivity_timeout=0.01)

    assert processed == 1
    assert seen == [{"only": True}]
    assert ch.acks == [1]
    assert ch.cancelled is True

"""Broker-agnostic background task queue (Feature: async background jobs).

Sibling of the Redis seams (``app/infra/redis_client.py``,
``app/infra/rate_limit.py``): a lazy, fail-soft, INERT-by-default piece of
infrastructure that degrades to a no-op the moment its backend is unconfigured
or unreachable — it must NEVER raise into a caller.

RED-LINE (interview-critical)
-----------------------------
This queue is for BACKGROUND work ONLY (memory curation, daily-summary
precompute, proactive nudges). It MUST NEVER sit in the realtime voice path
(VAD -> STT -> LLM -> TTS in ``app/ws/handler.py``). The only app-code enqueue
site is the ``chat_turn`` HTTP endpoint boundary, gated + try/excepted +
fire-and-forget. A broker outage must never break a chat or a voice turn.

Design notes
------------
* **Inert by default.** :func:`get_task_queue` returns ``None`` unless
  ``config.queue_enabled`` AND ``config.amqp_url`` is non-empty AND the broker
  is reachable. With ``AMQP_URL`` unset the default test suite is byte-identical
  to today: nothing connects and ``pika`` is never imported.
* **Lazy pika import.** ``import pika`` happens INSIDE the connect/publish
  methods, so ``import app.queue.task_queue`` works in a pika-less environment
  and a queue-disabled run never pays for the dependency.
* **Fail-soft, cache the outcome.** Mirrors ``redis_client.get_sync_redis``: a
  failed initial connect caches ``None`` so the hot path never re-pays a connect
  timeout; transient publish failures degrade to ``False`` and drop the channel
  so a later call can reconnect.
* **Durable & at-least-once.** A durable direct exchange (``vp.tasks``) routes a
  durable work queue (``vp.tasks.q``); messages are PERSISTENT
  (``delivery_mode=2``). The consumer uses manual ack + QoS prefetch. On handler
  failure the message is republished with an incremented ``attempts`` header up
  to ``queue_max_retries``, then dead-lettered through the DLX (``vp.tasks.dlx``)
  into the DLQ (``vp.tasks.dlq``).

JSON envelope: ``{id, type, payload, attempts, enqueued_at}``.
"""

from __future__ import annotations

import abc
import json
import logging
import re
import sys
import threading
import uuid
from datetime import UTC, datetime
from typing import Callable, Mapping

from app.config import config

logger = logging.getLogger(__name__)

# Fail-fast connection tuning: an unreachable broker should degrade to a no-op
# quickly instead of stalling the (gated) enqueue call. One attempt, no retry.
_SOCKET_TIMEOUT = 2.0
_CONNECT_ATTEMPTS = 1
# Heartbeat sized to comfortably exceed the slowest background job so a single
# in-flight handler can't starve heartbeat I/O on this blocking connection and
# get the broker to drop us mid-job (-> redelivery + duplicate work). The slowest
# jobs (curation/daily-summary) are bounded by the ~90s LLM timeout, so 2x that.
_HEARTBEAT_SECONDS = 180

_Handlers = Mapping[str, Callable[[dict], object]]


# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #
def _new_envelope(task_type: str, payload: dict | None) -> dict:
    """Build the canonical JSON envelope for a freshly enqueued task."""
    return {
        "id": str(uuid.uuid4()),
        "type": task_type,
        "payload": payload or {},
        "attempts": 0,
        "enqueued_at": datetime.now(UTC).isoformat(),
    }


def _encode(envelope: dict) -> bytes:
    return json.dumps(envelope, ensure_ascii=False).encode("utf-8")


def _redact(url: str) -> str:
    """Hide any ``user:password@`` credentials before logging a broker URL."""
    return re.sub(r"//[^@/]+@", "//***@", url or "")


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class TaskQueue(abc.ABC):
    """Broker-agnostic background task queue.

    Implementations are fail-soft: :meth:`enqueue` returns ``False`` (never
    raises) when the backend is unavailable, and :meth:`consume` dispatches to a
    ``{type: handler}`` registry with manual ack + retry/DLQ semantics.
    """

    @abc.abstractmethod
    def enqueue(self, task_type: str, payload: dict | None = None) -> bool:
        """Publish a task. Returns ``True`` iff it was handed to the broker."""

    @abc.abstractmethod
    def consume(
        self,
        handlers: _Handlers | None = None,
        *,
        max_messages: int | None = None,
        inactivity_timeout: float | None = None,
    ) -> int:
        """Consume + dispatch tasks; returns the number of deliveries processed.

        ``handlers`` defaults to the registry in ``app.queue.jobs``. With
        ``max_messages``/``inactivity_timeout`` unset this blocks forever (worker
        mode); both are set by tests for a bounded, deterministic drain.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the broker connection. Idempotent and never raises."""


# --------------------------------------------------------------------------- #
# RabbitMQ implementation (sync pika)
# --------------------------------------------------------------------------- #
class RabbitMQTaskQueue(TaskQueue):
    """Sync-``pika`` RabbitMQ queue with a durable topology and DLQ retries.

    Sync (not async) on purpose: the sole enqueue site — the ``chat_turn`` HTTP
    endpoint — is a sync ``def`` already running in FastAPI's threadpool, and a
    single non-awaited ``basic_publish`` is the whole enqueue. The consumer is a
    standalone blocking worker process.
    """

    def __init__(
        self,
        url: str,
        *,
        exchange: str,
        queue: str,
        routing_key: str,
        dlx: str,
        dlq: str,
        max_retries: int = 3,
        prefetch: int = 10,
    ) -> None:
        self.url = url
        self.exchange = exchange
        self.queue = queue
        self.routing_key = routing_key
        self.dlx = dlx
        self.dlq = dlq
        self.max_retries = max(1, int(max_retries))
        self.prefetch = max(1, int(prefetch))
        self._connection = None
        self._channel = None
        # pika's BlockingConnection/BlockingChannel is NOT thread-safe. The
        # producer is shared across FastAPI's threadpool (chat_turn is a sync
        # def), so serialize the whole enqueue critical section on this lock. The
        # consumer runs in its own single-threaded worker process and needs none.
        self._lock = threading.Lock()
        # Set by consume() when it can't start or drops mid-drain, so the worker
        # entrypoint chooses its exit code from this flag, not the delivery count.
        self._consume_aborted = False

    # -- connection / topology ------------------------------------------------ #
    def _ensure_channel(self):
        """Return an open channel, connecting + declaring topology on demand.

        Returns ``None`` (and logs) when pika is missing or the broker is
        unreachable — NEVER raises. ``pika`` is imported lazily here.
        """
        if self._channel is not None and self._channel.is_open:
            return self._channel

        try:
            import pika
        except Exception as exc:  # pragma: no cover - env without pika
            logger.warning("queue disabled: pika is not installed (%r)", exc)
            return None

        try:
            params = pika.URLParameters(self.url)
            params.socket_timeout = _SOCKET_TIMEOUT
            params.blocked_connection_timeout = _SOCKET_TIMEOUT
            params.connection_attempts = _CONNECT_ATTEMPTS
            params.retry_delay = 0
            params.heartbeat = _HEARTBEAT_SECONDS
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()
            self._declare_topology(self._channel)
            # Publisher confirms: make basic_publish synchronous w.r.t. the broker
            # so a lost/nacked publish RAISES (caught by callers -> nack+requeue)
            # instead of being silently dropped -- the at-least-once guarantee.
            self._channel.confirm_delivery()
            return self._channel
        except Exception as exc:
            logger.warning("queue disabled: broker unreachable at %s (%r)", _redact(self.url), exc)
            self._channel = None
            self._connection = None
            return None

    def _declare_topology(self, channel) -> None:
        """Idempotently declare the durable exchange/queue + DLX/DLQ topology."""
        # Dead-letter side first so the work queue can reference the DLX.
        channel.exchange_declare(exchange=self.dlx, exchange_type="fanout", durable=True)
        channel.queue_declare(queue=self.dlq, durable=True)
        channel.queue_bind(queue=self.dlq, exchange=self.dlx, routing_key="")

        # Main durable direct exchange + work queue bound by routing key. The
        # work queue also names the DLX so a reject/expire dead-letters cleanly.
        channel.exchange_declare(exchange=self.exchange, exchange_type="direct", durable=True)
        channel.queue_declare(
            queue=self.queue,
            durable=True,
            arguments={"x-dead-letter-exchange": self.dlx},
        )
        channel.queue_bind(queue=self.queue, exchange=self.exchange, routing_key=self.routing_key)

    def _publish(
        self,
        channel,
        exchange: str,
        routing_key: str,
        body: bytes,
        *,
        attempts: int | None = None,
        reason: str | None = None,
    ) -> None:
        """Publish a PERSISTENT message. Caller handles/limits exceptions."""
        import pika

        headers: dict = {}
        if attempts is not None:
            headers["x-attempts"] = attempts
        if reason is not None:
            headers["x-dlq-reason"] = reason
        props = pika.BasicProperties(
            delivery_mode=2,  # persistent
            content_type="application/json",
            headers=headers or None,
        )
        # mandatory=True + publisher confirms: an UNROUTABLE publish RAISES
        # UnroutableError (handled by callers) instead of being confirmed then
        # silently dropped -- required for the at-least-once guarantee.
        channel.basic_publish(
            exchange=exchange, routing_key=routing_key, body=body, properties=props, mandatory=True
        )

    # -- producer ------------------------------------------------------------- #
    def enqueue(self, task_type: str, payload: dict | None = None) -> bool:
        # Serialize the whole ensure-channel + publish + on-error drop: pika's
        # BlockingChannel is not thread-safe and this instance is shared across
        # FastAPI's threadpool. Fire-and-forget, so the (brief) lock never sits on
        # the realtime voice path.
        with self._lock:
            channel = self._ensure_channel()
            if channel is None:
                return False
            try:
                envelope = _new_envelope(task_type, payload)
                self._publish(channel, self.exchange, self.routing_key, _encode(envelope), attempts=0)
                return True
            except Exception as exc:
                logger.warning("enqueue(type=%s) failed (%r)", task_type, exc)
                # Drop a possibly-broken channel so the next call reconnects.
                self._channel = None
                self._connection = None
                return False

    # -- consumer ------------------------------------------------------------- #
    def consume(
        self,
        handlers: _Handlers | None = None,
        *,
        max_messages: int | None = None,
        inactivity_timeout: float | None = None,
    ) -> int:
        # Reset the abort flag each drain; the worker entrypoint (_run_worker)
        # consults it -- NOT the delivery count -- to pick its exit code.
        self._consume_aborted = False
        channel = self._ensure_channel()
        if channel is None:
            # Can't even start (pika missing / broker unreachable): a genuine
            # worker error, so signal a non-zero exit for a supervised restart.
            self._consume_aborted = True
            return 0
        if handlers is None:
            handlers = self._default_handlers()

        channel.basic_qos(prefetch_count=self.prefetch)

        # Guard for a mid-drain broker/channel/stream drop. pika is necessarily
        # importable here (a live channel exists); degrade to an empty tuple
        # (catches nothing) if it somehow is not, so this never raises in a
        # pika-less environment.
        try:
            from pika.exceptions import AMQPError as _AMQPError
        except Exception:
            _AMQPError = ()

        processed = 0
        try:
            for method, properties, body in channel.consume(self.queue, inactivity_timeout=inactivity_timeout):
                if method is None:
                    # inactivity_timeout elapsed with no message.
                    break
                self._handle_delivery(channel, method, properties, body, handlers)
                processed += 1
                if max_messages is not None and processed >= max_messages:
                    break
        except KeyboardInterrupt:  # pragma: no cover - worker Ctrl-C
            pass
        except _AMQPError as exc:  # pragma: no cover - broker/channel lost mid-drain
            # ConnectionClosed / ChannelClosed / StreamLostError while draining:
            # end the loop and return what we processed so the worker entrypoint
            # exits cleanly for a supervised restart instead of a bare traceback
            # (mirrors worker.main()'s except-Exception handling).
            self._consume_aborted = True
            logger.warning("consume aborted: broker/channel lost at %s (%r)", _redact(self.url), exc)
        finally:
            try:
                channel.cancel()
            except Exception:
                pass
        return processed

    def _handle_delivery(self, channel, method, properties, body, handlers: _Handlers) -> None:
        """Dispatch one delivery: ack on success; retry then DLQ on failure."""
        tag = method.delivery_tag

        try:
            envelope = json.loads(body)
        except Exception:
            logger.warning("dropping unparseable message -> DLQ")
            if self._to_dlq(channel, body if isinstance(body, (bytes, bytearray)) else b"{}", "decode-error"):
                self._safe_ack(channel, tag)
            else:  # DLQ publish failed: nack(requeue=False) so the work queue's DLX routes it -> DLQ (bounds the loop)
                self._safe_nack(channel, tag, requeue=False)
            return

        task_type = envelope.get("type", "")
        payload = envelope.get("payload") or {}
        attempts = int(envelope.get("attempts", 0) or 0)

        handler = handlers.get(task_type)
        if handler is None:
            logger.warning("no handler for type=%r -> DLQ", task_type)
            if self._to_dlq(channel, _encode(envelope), "no-handler"):
                self._safe_ack(channel, tag)
            else:  # DLQ publish failed: nack(requeue=False) so the work queue's DLX routes it -> DLQ (bounds the loop)
                self._safe_nack(channel, tag, requeue=False)
            return

        try:
            handler(payload)
        except Exception as exc:
            new_attempts = attempts + 1
            envelope["attempts"] = new_attempts
            if new_attempts >= self.max_retries:
                logger.warning("type=%s failed after %d attempt(s) -> DLQ (%r)", task_type, new_attempts, exc)
                envelope["error"] = repr(exc)
                if self._to_dlq(channel, _encode(envelope), "max-retries"):
                    self._safe_ack(channel, tag)
                else:  # DLQ publish failed: nack(requeue=False) so the work queue's DLX routes it -> DLQ (bounds the loop)
                    self._safe_nack(channel, tag, requeue=False)
            else:
                logger.info("type=%s attempt %d failed, requeueing (%r)", task_type, new_attempts, exc)
                try:
                    self._publish(
                        channel, self.exchange, self.routing_key, _encode(envelope), attempts=new_attempts
                    )
                except Exception as pub_exc:  # publish failed: nack->requeue instead of losing it
                    logger.warning("retry republish failed (%r); nack+requeue", pub_exc)
                    self._safe_nack(channel, tag, requeue=True)
                    return
                self._safe_ack(channel, tag)
            return

        self._safe_ack(channel, tag)

    def _to_dlq(self, channel, body: bytes, reason: str) -> bool:
        """Route a message to the DLQ via the (fanout) dead-letter exchange.

        Returns ``True`` iff the publish was accepted by the broker. On ``False``
        the caller nack+requeues instead of acking, so a redelivery can re-attempt
        the DLQ routing rather than silently dropping a poison message.
        """
        try:
            self._publish(channel, self.dlx, "", body, reason=reason)
            return True
        except Exception as exc:
            logger.warning("DLQ publish failed (%r)", exc)
            return False

    @staticmethod
    def _safe_ack(channel, delivery_tag) -> None:
        try:
            channel.basic_ack(delivery_tag=delivery_tag)
        except Exception as exc:  # pragma: no cover - broker/channel hiccup
            logger.warning("ack failed (%r)", exc)

    @staticmethod
    def _safe_nack(channel, delivery_tag, requeue: bool) -> None:
        try:
            channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        except Exception as exc:  # pragma: no cover - broker/channel hiccup
            logger.warning("nack failed (%r)", exc)

    @staticmethod
    def _default_handlers() -> _Handlers:
        """Lazy soft-dependency on the job registry (avoids an import cycle)."""
        from app.queue import jobs

        return jobs.get_registry()

    @property
    def available(self) -> bool:
        """True iff a channel could be established (connects lazily)."""
        return self._ensure_channel() is not None

    def close(self) -> None:
        try:
            if self._channel is not None and self._channel.is_open:
                self._channel.close()
        except Exception:
            pass
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass
        self._channel = None
        self._connection = None


# --------------------------------------------------------------------------- #
# Process-cached factory + fire-and-forget helpers
# --------------------------------------------------------------------------- #
_queue: TaskQueue | None = None
_attempted = False
# Guards first-call construction so concurrent threadpool callers don't race to
# build/connect two queues (double-checked against _attempted below).
_init_lock = threading.Lock()


def get_task_queue() -> TaskQueue | None:
    """Return a cached, connect-verified :class:`TaskQueue`, or ``None``.

    Returns ``None`` (and caches it) when the queue is disabled
    (``queue_enabled`` false / ``amqp_url`` empty) or the broker is unreachable —
    mirroring ``redis_client.get_sync_redis``. Never raises.
    """
    global _queue, _attempted
    if _attempted:
        return _queue

    # Double-checked lock: concurrent first calls from FastAPI's threadpool must
    # not race to build/connect two queues (which would leak a connection).
    # _attempted is set only AFTER _queue is finalized, so the lock-free fast path
    # above never observes a half-initialized cache.
    with _init_lock:
        if _attempted:
            return _queue

        if not config.queue_enabled or not config.amqp_url:
            _queue = None
            _attempted = True
            return None

        q = RabbitMQTaskQueue(
            config.amqp_url,
            exchange=config.queue_exchange,
            queue=config.queue_name,
            routing_key=config.queue_routing_key,
            dlx=config.queue_dlx,
            dlq=config.queue_dlq,
            max_retries=config.queue_max_retries,
            prefetch=config.queue_prefetch,
        )
        if q._ensure_channel() is None:
            # Broker unreachable / pika missing: cache the None so we don't re-pay
            # the connect timeout on the hot path.
            q.close()
            _queue = None
            _attempted = True
            return None

        _queue = q
        _attempted = True
        return q


def reset_task_queue_cache() -> None:
    """Clear the cached queue. Call after monkeypatching env/config in tests."""
    global _queue, _attempted
    if _queue is not None:
        try:
            _queue.close()
        except Exception:
            pass
    _queue = None
    _attempted = False


def queue_available() -> bool:
    """True iff a broker-backed queue could be built and connected."""
    return get_task_queue() is not None


def enqueue(task_type: str, payload: dict | None = None) -> bool:
    """Fire-and-forget enqueue for the HTTP boundary.

    No-op (returns ``False``) when the queue is disabled/unreachable, and swallows
    ALL errors so a broker outage can never break the calling turn.
    """
    try:
        q = get_task_queue()
        if q is None:
            return False
        return q.enqueue(task_type, payload)
    except Exception:
        return False


def _run_worker() -> int:  # pragma: no cover - manual worker entrypoint
    """Blocking consumer entrypoint: ``python -m app.queue.task_queue``.

    Returns a process exit code (NEVER the delivery count): ``0`` when the queue
    is intentionally inert (disabled / no ``AMQP_URL``) or after a clean stop, and
    ``1`` when the broker is configured-but-unreachable or the consume loop aborts
    on a mid-drain broker loss -- so a supervisor restarts us. Mirrors ``worker.main``.
    """
    logging.basicConfig(level=logging.INFO)
    # Inert by default: disabled or no AMQP_URL is the normal config, not an error.
    if not config.queue_enabled or not config.amqp_url:
        logger.warning("queue disabled (set QUEUE_ENABLED=1 and AMQP_URL); nothing to consume")
        return 0
    q = get_task_queue()
    if q is None:
        # Configured but broker unreachable: a real worker error -> non-zero exit.
        logger.error("queue configured but broker unreachable; exiting for restart")
        return 1
    logger.info("task-queue worker consuming from %s", config.queue_name)
    q.consume()
    # consume() swallows a mid-drain AMQPError and returns a count; consult the
    # abort flag (NOT the count) for the exit status.
    return 1 if getattr(q, "_consume_aborted", False) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_run_worker())

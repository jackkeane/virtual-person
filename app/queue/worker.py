"""Standalone background-job consumer worker (Feature: async background jobs).

Runnable as ``python -m app.queue.worker``. This is the long-lived process that
drains the durable work queue declared by ``app/queue/task_queue.py`` and runs
the registered job handlers (``app/queue/jobs.py``) OFF the realtime path.

RED-LINE (interview-critical)
-----------------------------
The queue carries BACKGROUND work ONLY (memory curation, daily-summary
precompute, proactive nudges). It MUST NEVER sit in the realtime voice path
(VAD -> STT -> LLM -> TTS in ``app/ws/handler.py``). This worker is a separate
process; nothing in the chat/voice turn ever waits on it. A broker outage must
never break a chat or a voice turn — the producer side is fire-and-forget, and
this consumer is the only thing that blocks on the broker.

Design notes
------------
* **Gated + inert by default.** Mirrors the Redis/queue seams: with the queue
  disabled or ``AMQP_URL`` unset there is nothing to consume, so ``main()`` logs
  a clear, actionable line and exits cleanly (0) instead of connecting. This is
  the normal "inert by default" state, not a failure.
* **Lazy pika, fail-soft connect.** ``import pika`` happens INSIDE
  :meth:`Worker.connect` (so ``import app.queue.worker`` works in a pika-less
  env such as the default CI test image), and the connect mirrors
  ``redis_client.get_sync_redis`` / ``RabbitMQTaskQueue._ensure_channel``: short
  socket timeouts, one attempt, catch-all -> return ``False`` and log rather than
  raise. Unlike the request-path seams, a configured-but-unreachable broker is a
  genuine worker error: ``main()`` exits non-zero so a supervisor restarts it.
* **Same topology + wire format as the producer.** The durable DLX/DLQ and the
  direct exchange + work queue (with ``x-dead-letter-exchange``) are declared
  idempotently exactly as in ``RabbitMQTaskQueue._declare_topology``, and the
  JSON envelope / headers (``x-attempts``, ``x-dlq-reason``, persistent
  ``delivery_mode=2``) match ``RabbitMQTaskQueue`` so retried messages this
  worker republishes stay byte-identical to producer-published ones.
* **At-least-once with a poison guard.** ``basic_qos(prefetch)`` + manual ack:
  a delivery is ACKed ONLY after its handler returns. On failure the envelope's
  ``attempts`` counter is incremented and the message is republished up to
  ``queue_max_retries`` total attempts, after which it is dead-lettered through
  the DLX into the DLQ. Undecodable bodies and unknown task types go straight to
  the DLQ. This is the same state machine as ``RabbitMQTaskQueue._handle_delivery``.
* **Graceful shutdown.** SIGINT/SIGTERM flip a stop flag; the consume loop wakes
  at least once per poll interval (via ``inactivity_timeout``), finishes any
  in-flight handler (so we never ack a half-done job), then cancels the consumer
  and closes the channel/connection.
* **Metrics, best-effort.** Per-outcome counters are emitted via the
  ``inc_job_*`` helpers, imported DEFENSIVELY (a missing metrics module or helper
  degrades to a no-op; the helpers are themselves non-raising). If
  ``QUEUE_METRICS_PORT`` is set, a Prometheus HTTP endpoint is started so a demo
  can scrape the worker's own process.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
from typing import Callable, Mapping

from app.config import config

logger = logging.getLogger(__name__)

# Fail-fast connection tuning (mirrors task_queue): an unreachable broker should
# surface quickly instead of hanging worker startup. One attempt, no retry.
_SOCKET_TIMEOUT = 2.0
_CONNECT_ATTEMPTS = 1
# Heartbeat sized to comfortably exceed the slowest background job so a single
# long-running handler on this blocking connection can't starve heartbeat I/O and
# get the broker to drop us mid-job (-> redelivery + duplicate work). The slowest
# jobs (curation/daily-summary) are bounded by the ~90s LLM timeout, so 2x that.
_HEARTBEAT_SECONDS = 180

# How often the (blocking) consume generator wakes when the queue is idle, so a
# shutdown signal is noticed within this bound even with no traffic. Seconds.
_POLL_INTERVAL = 1.0

_Handlers = Mapping[str, Callable[[dict], object]]


# --------------------------------------------------------------------------- #
# Defensive metrics binding
# --------------------------------------------------------------------------- #
# The queue counters (inc_job_processed/failed/retried/dead_lettered) live in
# app.observability.metrics and are themselves non-raising. We bind each one
# independently so the worker runs unchanged whether the module is present,
# absent, or only partially populated — a missing helper degrades to a no-op.
def _noop_metric() -> None:
    return None


try:  # pragma: no cover - observability is an optional dependency here
    from app.observability import metrics as _metrics_mod
except Exception:  # noqa: BLE001 - never let instrumentation break the worker
    _metrics_mod = None


def _bind_metric(name: str) -> Callable[[], None]:
    """Resolve a no-arg metrics helper by name, or a no-op if it is unavailable."""
    fn = getattr(_metrics_mod, name, None) if _metrics_mod is not None else None
    return fn if callable(fn) else _noop_metric


_inc_processed = _bind_metric("inc_job_processed")
_inc_failed = _bind_metric("inc_job_failed")
_inc_retried = _bind_metric("inc_job_retried")
_inc_dead_lettered = _bind_metric("inc_job_dead_lettered")


# --------------------------------------------------------------------------- #
# Wire helpers (kept identical to app.queue.task_queue's envelope format)
# --------------------------------------------------------------------------- #
def _encode(envelope: dict) -> bytes:
    """Serialize an envelope exactly as the producer does (byte-identical retries)."""
    return json.dumps(envelope, ensure_ascii=False).encode("utf-8")


def _redact(url: str) -> str:
    """Hide any ``user:password@`` credentials before logging a broker URL."""
    return re.sub(r"//[^@/]+@", "//***@", url or "")


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class Worker:
    """A blocking, sync-``pika`` consumer for the durable ``vp.tasks`` work queue.

    Owns its own connection/channel so it can weave signal-driven graceful
    shutdown and per-outcome metrics into the consume loop while reusing the
    exact durable topology, envelope format, and retry/DLQ policy of
    :class:`app.queue.task_queue.RabbitMQTaskQueue`.
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
        handlers: _Handlers | None = None,
    ) -> None:
        self.url = url
        self.exchange = exchange
        self.queue = queue
        self.routing_key = routing_key
        self.dlx = dlx
        self.dlq = dlq
        self.max_retries = max(1, int(max_retries))
        self.prefetch = max(1, int(prefetch))
        self._handlers = handlers
        self._connection = None
        self._channel = None
        self._stopping = False

    # -- connection / topology ------------------------------------------------ #
    def connect(self) -> bool:
        """Connect, declare topology, and set QoS. Returns ``False`` (never raises).

        Mirrors the ``redis_client`` / ``RabbitMQTaskQueue`` degrade pattern:
        ``pika`` is imported lazily here, the connect uses short timeouts and a
        single attempt, and any failure (missing pika, unreachable broker) is
        logged and reported as ``False`` rather than raised.
        """
        try:
            import pika
        except Exception as exc:  # pragma: no cover - env without pika
            logger.error("event=start_failed reason=pika-missing error=%r", exc)
            return False

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
            # so a lost/nacked retry-or-DLQ publish RAISES (caught -> nack+requeue)
            # instead of being silently dropped -- the at-least-once guarantee.
            self._channel.confirm_delivery()
            self._channel.basic_qos(prefetch_count=self.prefetch)
            logger.info("event=connected broker=%s", _redact(self.url))
            return True
        except Exception as exc:
            logger.error("event=broker_unreachable broker=%s error=%r", _redact(self.url), exc)
            self._channel = None
            self._connection = None
            return False

    def _declare_topology(self, channel) -> None:
        """Idempotently declare the durable exchange/queue + DLX/DLQ topology.

        Identical to ``RabbitMQTaskQueue._declare_topology`` so producer and
        consumer converge on the same broker objects regardless of who connects
        first.
        """
        # Dead-letter side first so the work queue can reference the DLX.
        channel.exchange_declare(exchange=self.dlx, exchange_type="fanout", durable=True)
        channel.queue_declare(queue=self.dlq, durable=True)
        channel.queue_bind(queue=self.dlq, exchange=self.dlx, routing_key="")

        # Main durable direct exchange + work queue bound by routing key; the
        # work queue names the DLX so a reject/expire dead-letters cleanly.
        channel.exchange_declare(exchange=self.exchange, exchange_type="direct", durable=True)
        channel.queue_declare(
            queue=self.queue,
            durable=True,
            arguments={"x-dead-letter-exchange": self.dlx},
        )
        channel.queue_bind(queue=self.queue, exchange=self.exchange, routing_key=self.routing_key)

    def _publish(
        self,
        exchange: str,
        routing_key: str,
        body: bytes,
        *,
        attempts: int | None = None,
        reason: str | None = None,
    ) -> None:
        """Publish a PERSISTENT message; headers match the producer. Caller limits exceptions."""
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
        self._channel.basic_publish(
            exchange=exchange, routing_key=routing_key, body=body, properties=props, mandatory=True
        )

    # -- consume loop --------------------------------------------------------- #
    def run(self) -> int:
        """Block draining the queue until stopped. Returns deliveries processed.

        Uses the generator consumer with an ``inactivity_timeout`` so a shutdown
        signal is observed within ``_POLL_INTERVAL`` even when the queue is idle.
        A stop request breaks the loop between deliveries (never mid-handler).
        """
        handlers = self._handlers
        if handlers is None:  # standalone/test convenience; main() passes these in
            from app.queue import jobs

            handlers = jobs.get_registry()

        processed = 0
        try:
            for method, properties, body in self._channel.consume(
                self.queue, inactivity_timeout=_POLL_INTERVAL
            ):
                if self._stopping:
                    break
                if method is None:
                    continue  # idle tick: no message this interval, re-check stop flag
                self._handle_delivery(method, properties, body, handlers)
                processed += 1
                if self._stopping:
                    break
        finally:
            # Stop the consumer; the broker requeues any prefetched-but-unacked
            # deliveries for redelivery (at-least-once). Mirrors task_queue.consume.
            try:
                self._channel.cancel()
            except Exception:
                pass
        return processed

    def _handle_delivery(self, method, properties, body, handlers: _Handlers) -> None:
        """Dispatch one delivery: ack on success; retry then DLQ on failure.

        Identical policy to ``RabbitMQTaskQueue._handle_delivery`` (attempts are
        tracked in the envelope body; ``queue_max_retries`` total attempts before
        the DLQ), with per-outcome metrics emitted at each terminal branch.
        """
        tag = method.delivery_tag

        try:
            envelope = json.loads(body)
        except Exception:
            logger.warning("event=dlq reason=decode-error")
            if self._to_dlq(body if isinstance(body, (bytes, bytearray)) else b"{}", "decode-error"):
                self._safe_ack(tag)
                _inc_dead_lettered()
            else:  # DLQ publish failed: nack(requeue=False) so the work queue's DLX routes it -> DLQ (bounds the loop)
                self._safe_nack(tag, requeue=False)
            return

        task_type = envelope.get("type", "")
        payload = envelope.get("payload") or {}
        attempts = int(envelope.get("attempts", 0) or 0)

        handler = handlers.get(task_type)
        if handler is None:
            logger.warning("event=dlq reason=no-handler type=%r", task_type)
            if self._to_dlq(_encode(envelope), "no-handler"):
                self._safe_ack(tag)
                _inc_dead_lettered()
            else:  # DLQ publish failed: nack(requeue=False) so the work queue's DLX routes it -> DLQ (bounds the loop)
                self._safe_nack(tag, requeue=False)
            return

        try:
            handler(payload)
        except Exception as exc:
            _inc_failed()
            new_attempts = attempts + 1
            envelope["attempts"] = new_attempts
            if new_attempts >= self.max_retries:
                logger.warning(
                    "event=dlq reason=max-retries type=%s attempts=%d error=%r",
                    task_type,
                    new_attempts,
                    exc,
                )
                envelope["error"] = repr(exc)
                if self._to_dlq(_encode(envelope), "max-retries"):
                    self._safe_ack(tag)
                    _inc_dead_lettered()
                else:  # DLQ publish failed: nack(requeue=False) so the work queue's DLX routes it -> DLQ (bounds the loop)
                    self._safe_nack(tag, requeue=False)
            else:
                logger.info(
                    "event=job_retry type=%s attempt=%d error=%r", task_type, new_attempts, exc
                )
                try:
                    self._publish(
                        self.exchange, self.routing_key, _encode(envelope), attempts=new_attempts
                    )
                except Exception as pub_exc:  # publish failed: nack->requeue instead of losing it
                    logger.warning("event=retry_republish_failed error=%r action=nack-requeue", pub_exc)
                    self._safe_nack(tag, requeue=True)
                    return
                self._safe_ack(tag)
                _inc_retried()
            return

        logger.info("event=job_ok type=%s", task_type)
        self._safe_ack(tag)
        _inc_processed()

    # -- broker helpers (mirror task_queue) ----------------------------------- #
    def _to_dlq(self, body: bytes, reason: str) -> bool:
        """Route a message to the DLQ via the (fanout) dead-letter exchange.

        Returns ``True`` iff the publish was accepted by the broker. On ``False``
        the caller nack+requeues instead of acking, so a redelivery can re-attempt
        the DLQ routing rather than silently dropping a poison message.
        """
        try:
            self._publish(self.dlx, "", body, reason=reason)
            return True
        except Exception as exc:
            logger.warning("event=dlq_publish_failed reason=%s error=%r", reason, exc)
            return False

    def _safe_ack(self, delivery_tag) -> None:
        try:
            self._channel.basic_ack(delivery_tag=delivery_tag)
        except Exception as exc:  # pragma: no cover - broker/channel hiccup
            logger.warning("event=ack_failed error=%r", exc)

    def _safe_nack(self, delivery_tag, requeue: bool) -> None:
        try:
            self._channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        except Exception as exc:  # pragma: no cover - broker/channel hiccup
            logger.warning("event=nack_failed error=%r", exc)

    # -- lifecycle ------------------------------------------------------------ #
    def stop(self) -> None:
        """Request a graceful stop (safe to call from a signal handler)."""
        self._stopping = True

    def close(self) -> None:
        """Release the channel + connection. Idempotent and never raises."""
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
# Entrypoint helpers
# --------------------------------------------------------------------------- #
def _install_signal_handlers(worker: Worker) -> None:
    """Route SIGINT/SIGTERM to a graceful stop (best-effort; main-thread only)."""

    def _on_signal(signum, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        logger.info("event=signal name=%s action=graceful-stop", name)
        worker.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):  # pragma: no cover - not in the main thread
            pass


def _maybe_start_metrics_server() -> None:
    """If ``QUEUE_METRICS_PORT`` is set, expose Prometheus metrics for scraping."""
    port_raw = os.getenv("QUEUE_METRICS_PORT", "").strip()
    if not port_raw:
        return
    try:
        port = int(port_raw)
    except ValueError:
        logger.warning("event=metrics_server_skipped reason=bad-port value=%r", port_raw)
        return
    try:
        from prometheus_client import start_http_server

        start_http_server(port)
        logger.info("event=metrics_server_started port=%d", port)
    except Exception as exc:  # pragma: no cover - optional; never fatal to the worker
        logger.warning("event=metrics_server_failed port=%d error=%r", port, exc)


def main() -> int:
    """Blocking worker entrypoint: ``python -m app.queue.worker``.

    Exit codes: ``0`` when the queue is intentionally inert (disabled or no
    ``AMQP_URL``) or after a clean graceful shutdown; ``1`` on a genuine failure
    (broker configured but unreachable, a broken job registry, or an unexpected
    consume-loop error) so a supervisor restarts the worker.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Gate: inert by default. With the queue disabled or AMQP_URL unset there is
    # nothing to consume — this is the normal configuration, not an error.
    if not config.queue_enabled:
        logger.warning("event=inert reason=queue-disabled hint=set-QUEUE_ENABLED=1-and-AMQP_URL")
        return 0
    if not config.amqp_url:
        logger.warning(
            "event=inert reason=no-amqp-url "
            "hint=set-AMQP_URL-e.g.-amqp://guest:guest@127.0.0.1:5672/"
        )
        return 0

    # Resolve the job registry up front so a broken import fails fast, before we
    # touch the broker. (Imported here, not at module top, so `import
    # app.queue.worker` stays free of the memory/proactivity chain.)
    try:
        from app.queue import jobs

        handlers = jobs.get_registry()
    except Exception as exc:
        logger.error("event=start_failed reason=registry-error error=%r", exc)
        return 1

    worker = Worker(
        config.amqp_url,
        exchange=config.queue_exchange,
        queue=config.queue_name,
        routing_key=config.queue_routing_key,
        dlx=config.queue_dlx,
        dlq=config.queue_dlq,
        max_retries=config.queue_max_retries,
        prefetch=config.queue_prefetch,
        handlers=handlers,
    )
    _install_signal_handlers(worker)

    if not worker.connect():
        # Configured but unreachable: a real worker error (unlike the request
        # path, a consumer that can't connect should exit non-zero to be restarted).
        return 1

    _maybe_start_metrics_server()

    logger.info(
        "event=worker_start queue=%s prefetch=%d max_retries=%d jobs=%d",
        config.queue_name,
        config.queue_prefetch,
        config.queue_max_retries,
        len(handlers),
    )
    processed = 0
    rc = 0
    try:
        processed = worker.run()
    except KeyboardInterrupt:  # pragma: no cover - Ctrl-C without a handler installed
        pass
    except Exception as exc:  # pragma: no cover - unexpected broker/channel failure
        logger.error("event=consume_error error=%r", exc)
        rc = 1
    finally:
        worker.close()
    logger.info("event=worker_stop processed=%d exit=%d", processed, rc)
    return rc


if __name__ == "__main__":  # pragma: no cover - manual worker entrypoint
    sys.exit(main())

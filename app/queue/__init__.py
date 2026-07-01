"""Background task queue (Feature: async background jobs).

BACKGROUND-ONLY and INERT BY DEFAULT. This package carries async background work
(memory curation, daily-summary precompute, proactive nudges) — it MUST NEVER
sit in the realtime voice path (VAD -> STT -> LLM -> TTS in app/ws/handler.py).
With ``AMQP_URL`` unset the queue is a no-op: nothing connects and ``pika`` is
never imported (see app/queue/task_queue.py). A broker outage must never break a
chat or a voice turn.

Only the lightweight transport API is re-exported here; import ``app.queue.jobs``
directly for the handler registry (it pulls in the memory/proactivity services).
"""

from app.queue.task_queue import (
    RabbitMQTaskQueue,
    TaskQueue,
    enqueue,
    get_task_queue,
    queue_available,
    reset_task_queue_cache,
)

__all__ = [
    "TaskQueue",
    "RabbitMQTaskQueue",
    "get_task_queue",
    "enqueue",
    "queue_available",
    "reset_task_queue_cache",
]

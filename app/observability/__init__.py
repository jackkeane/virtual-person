"""Observability package for the Ani realtime voice pipeline (Feature 3).

Exposes Prometheus collectors and tiny, never-raising instrumentation helpers
(see ``app.observability.metrics``). Instrumentation is in-process and O(1) so
the realtime hot path stays latency-neutral.
"""

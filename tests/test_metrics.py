"""Tests for the Prometheus /metrics endpoint and instrumentation helpers.

No Redis required: metrics live on the in-process default registry and are
independent of the Redis-gated session/cache/rate-limit features.
"""
from __future__ import annotations

import os

os.environ["AUDIT_LOG_PATH"] = ""  # disable file persistence in tests

from fastapi.testclient import TestClient

from app.config import config
from app.main import app
from app.observability import metrics as obs_metrics

client = TestClient(app)


def _sample(body: str, name: str) -> float:
    """Extract the numeric value of an unlabeled exposition line ``<name> <value>``."""
    prefix = name + " "
    for line in body.splitlines():
        if line.startswith(prefix):
            return float(line.split(" ", 1)[1])
    return 0.0


def test_metrics_endpoint_ok_and_content_type():
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text exposition content type.
    assert r.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in r.headers["content-type"]


def test_metrics_body_contains_expected_collectors():
    body = client.get("/metrics").text
    # Required by the spec.
    assert "vp_turns_total" in body
    assert "vp_ttfa_seconds" in body
    # The rest of the collectors are exposed too (TYPE/HELP lines always emit).
    for name in (
        "vp_vad_seconds",
        "vp_stt_seconds",
        "vp_tts_seconds",
        "vp_tts_cache_total",
        "vp_rate_limited_total",
    ):
        assert name in body


def test_observe_ttfa_changes_scrape():
    """observe_ttfa() then a re-scrape must reflect the new sample."""
    before_body = client.get("/metrics").text
    before_count = _sample(before_body, "vp_ttfa_seconds_count")
    before_sum = _sample(before_body, "vp_ttfa_seconds_sum")

    obs_metrics.observe_ttfa(0.5)

    after_body = client.get("/metrics").text
    after_count = _sample(after_body, "vp_ttfa_seconds_count")
    after_sum = _sample(after_body, "vp_ttfa_seconds_sum")

    assert after_count == before_count + 1
    assert after_sum == before_sum + 0.5
    assert before_body != after_body


def test_inc_turn_increments_counter():
    before = _sample(client.get("/metrics").text, "vp_turns_total")
    obs_metrics.inc_turn()
    after = _sample(client.get("/metrics").text, "vp_turns_total")
    assert after == before + 1


def test_tts_cache_labels_hit_and_miss():
    obs_metrics.inc_tts_cache(hit=True)
    obs_metrics.inc_tts_cache(hit=False)
    body = client.get("/metrics").text
    assert 'vp_tts_cache_total{result="hit"}' in body
    assert 'vp_tts_cache_total{result="miss"}' in body


def test_helpers_never_raise_on_bad_input():
    # Instrumentation must degrade to a no-op, never raise into the hot path.
    obs_metrics.observe_vad("not-a-number")  # type: ignore[arg-type]
    obs_metrics.observe_stt(None)  # type: ignore[arg-type]
    obs_metrics.observe_tts(object())  # type: ignore[arg-type]
    obs_metrics.inc_rate_limited()
    # Still scrapes fine afterward.
    assert client.get("/metrics").status_code == 200


def test_metrics_disabled_returns_404():
    prev = config.metrics_enabled
    config.metrics_enabled = False
    try:
        r = client.get("/metrics")
        assert r.status_code == 404
    finally:
        config.metrics_enabled = prev
    # Re-enabled afterwards.
    assert client.get("/metrics").status_code == 200

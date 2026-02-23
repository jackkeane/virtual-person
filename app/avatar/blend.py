from __future__ import annotations

import math


def clamp01(value: float, fallback: float = 0.0) -> float:
    if value is None or not math.isfinite(value):
        value = fallback
    return max(0.0, min(1.0, float(value)))


def exp_smooth(current: float, target: float, lam: float, dt_sec: float) -> float:
    """Deterministic exponential smoothing used by avatar blend logic.

    y(t+dt) = y + (target - y) * (1 - exp(-lam*dt))
    """
    if dt_sec <= 0 or lam <= 0:
        return float(target)
    alpha = 1.0 - math.exp(-lam * dt_sec)
    return float(current + (target - current) * alpha)


def blend_mouth(expression_open: float, speech_open: float, speech_weight: float, micro_open: float = 0.0) -> float:
    """Blend expression mouth baseline and speech mouth target with optional additive micro motion."""
    w = clamp01(speech_weight, 0.0)
    expr = clamp01(expression_open, 0.05)
    speech = clamp01(speech_open, 0.05)
    mixed = expr * (1.0 - w) + speech * w + float(micro_open)
    return clamp01(mixed, expr)

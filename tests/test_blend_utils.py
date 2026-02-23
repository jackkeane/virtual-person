import math

from app.avatar.blend import blend_mouth, clamp01, exp_smooth


def test_clamp01_bounds_and_fallback():
    assert clamp01(-1) == 0.0
    assert clamp01(2) == 1.0
    assert clamp01(float("nan"), 0.3) == 0.3


def test_exp_smooth_matches_closed_form_step():
    current = 0.2
    target = 0.8
    lam = 10.0
    dt = 0.1
    expected = current + (target - current) * (1 - math.exp(-lam * dt))
    assert exp_smooth(current, target, lam, dt) == expected


def test_exp_smooth_degenerate_cases():
    assert exp_smooth(0.2, 0.9, 0.0, 0.1) == 0.9
    assert exp_smooth(0.2, 0.9, 8.0, 0.0) == 0.9


def test_blend_mouth_speech_dominates_when_weight_is_one():
    assert blend_mouth(expression_open=0.15, speech_open=0.8, speech_weight=1.0) == 0.8


def test_blend_mouth_returns_to_expression_when_not_speaking():
    assert blend_mouth(expression_open=0.22, speech_open=0.9, speech_weight=0.0) == 0.22


def test_blend_mouth_applies_micro_motion_with_clamp():
    assert blend_mouth(0.2, 0.2, 0.0, micro_open=0.05) == 0.25
    assert blend_mouth(0.95, 0.95, 1.0, micro_open=0.2) == 1.0
